//! The thread pool that every parallel call in the crate runs on, rebuilt after `fork`.
//!
//! rayon's global pool starts its threads on first use, and `fork` copies only the
//! thread that calls it. A child forked from a process that already ran a parallel
//! call (a `multiprocessing` or `PyTorch` `DataLoader` worker reading `GeoTIFF` patches
//! after the parent read one) inherits a pool whose threads do not exist, and its first
//! parallel call waits forever for them. So the crate never uses the global pool:
//! [`install`] runs work on a pool keyed by process ID, which a child builds anew on
//! first use. The fetcher's Tokio runtime is keyed the same way.

use std::sync::{Mutex, PoisonError};

use rayon::{ThreadPool, ThreadPoolBuilder};

/// A pool and the process that built it.
struct ProcessPool {
    pid: u32,
    pool: &'static ThreadPool,
}

/// The current process's pool, or the one inherited from the parent before a child
/// builds its own. Pools are leaked: the process's own lives as long as the process
/// (as the global pool would), and an inherited one must not be dropped, since that
/// would signal threads that do not exist.
static POOL: Mutex<Option<ProcessPool>> = Mutex::new(None);

/// The pool of the process with ID `pid`, if one was built.
fn current(pid: u32) -> Option<&'static ThreadPool> {
    POOL.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
        .filter(|found| found.pid == pid)
        .map(|found| found.pool)
}

/// This process's pool, built on first use with as many threads as rayon's global pool
/// would have (`RAYON_NUM_THREADS`, or one per CPU).
fn pool() -> Result<&'static ThreadPool, String> {
    let pid = std::process::id();
    if let Some(pool) = current(pid) {
        return Ok(pool);
    }
    // Built without holding the lock: a fork from another thread while the lock is
    // held would leave it locked in the child.
    let built = ThreadPoolBuilder::new()
        .thread_name(|i| format!("mapcv-worker-{i}"))
        .build()
        .map_err(|e| {
            format!(
                "mapcv could not start its worker threads ({e}); free some memory or set \
                 RAYON_NUM_THREADS to a smaller number, then try again"
            )
        })?;
    let mut guard = POOL.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(found) = guard.as_ref().filter(|found| found.pid == pid) {
        // Another thread built one first; `built` is dropped (its threads stop).
        return Ok(found.pool);
    }
    let pool: &'static ThreadPool = Box::leak(Box::new(built));
    *guard = Some(ProcessPool { pid, pool });
    Ok(pool)
}

/// Run `op` on this process's thread pool; the parallel iterators inside it use that
/// pool. Called from inside the pool, `op` runs in place.
///
/// # Errors
/// Returns a message when the pool's threads cannot be started.
pub fn install<R: Send>(op: impl FnOnce() -> R + Send) -> Result<R, String> {
    Ok(pool()?.install(op))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn work_runs_on_one_pool_per_process_and_a_child_builds_its_own() {
        let first = pool().unwrap();
        assert!(std::ptr::eq(first, pool().unwrap()));
        let sum: u64 = install(|| (1..=100u64).into_par_iter().sum()).unwrap();
        assert_eq!(sum, 5050);
        // Inside `install`, rayon reports the pool's threads, not the global pool's.
        let threads = install(rayon::current_num_threads).unwrap();
        assert_eq!(threads, first.current_num_threads());
        // Nested calls run in place on the same pool.
        let nested = install(|| install(|| rayon::current_thread_index().is_some())).unwrap();
        assert_eq!(nested, Ok(true));

        // What a forked child sees: the entry names its parent's process ID. (One test,
        // since tests share the process and run in parallel.)
        let pid = std::process::id();
        let mine = first;
        let replaced = POOL.lock().unwrap().replace(ProcessPool {
            pid: pid.wrapping_add(1),
            pool: mine,
        });
        assert_eq!(replaced.map(|found| found.pid), Some(pid));
        assert!(current(pid).is_none());
        let rebuilt = pool().unwrap();
        assert!(!std::ptr::eq(rebuilt, mine));
        assert!(std::ptr::eq(current(pid).unwrap(), rebuilt));
        assert_eq!(
            install(|| (1..=100u64).into_par_iter().sum::<u64>()),
            Ok(5050)
        );
    }
}
