#!/usr/bin/env bash
# Run one fuzz target for a while, the way CI does.
#
#   fuzz/run.sh <target> [seconds]      # needs: rustup nightly, cargo install cargo-fuzz
#
# Targets: geotiff_open geotiff_structured chunk_decode kml_parse tile_decode.
# A hang (> 10 s) or an allocation past 2 GiB counts as a crash. The grown corpus is
# kept in fuzz/corpus/<target> (git-ignored); the committed seeds are fuzz/seeds.
set -euo pipefail

target="${1:?usage: fuzz/run.sh <target> [seconds]}"
seconds="${2:-60}"
cd "$(dirname "$0")/.."

seeds=""
dict=""
max_len=4096
case "$target" in
  geotiff_open) seeds=geotiff; dict=tiff.dict; max_len=16384 ;;
  geotiff_structured) max_len=4096 ;;
  chunk_decode) seeds=chunk; max_len=8192 ;;
  kml_parse) seeds=kml; dict=kml.dict; max_len=65536 ;;
  tile_decode) seeds=tile; max_len=65536 ;;
  *) echo "unknown fuzz target: $target" >&2; exit 2 ;;
esac

mkdir -p "fuzz/corpus/$target"
corpus=("fuzz/corpus/$target")
if [ -n "$seeds" ]; then
  corpus+=("fuzz/seeds/$seeds")
fi
options=(
  "-max_total_time=$seconds" "-rss_limit_mb=2048" "-timeout=10"
  "-max_len=$max_len" "-print_final_stats=1"
)
if [ -n "$dict" ]; then
  options+=("-dict=fuzz/dictionaries/$dict")
fi

# The decoders fan out over rayon; a full-size pool only adds scheduling noise to a
# single-process fuzzer (5x slower on a busy machine), and two threads still exercise
# the parallel code.
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"

exec cargo +nightly fuzz run "$target" "${corpus[@]}" -- "${options[@]}"
