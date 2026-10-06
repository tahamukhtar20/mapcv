//! Arbitrary bytes as a map tile (PNG, JPEG, WebP, GIF), decoded into windows of
//! a tile raster at several offsets, including partial overlaps.
#![no_main]

use libfuzzer_sys::fuzz_target;
use mapcv::tile_decoder::{decode_window, Tile};

type Case<'a> = (&'a [Tile<'a>], (u32, u32), (usize, usize), (usize, usize));

fuzz_target!(|data: &[u8]| {
    let single = [Tile { x: 5, y: 7, data }];
    let quad = [
        Tile { x: 0, y: 0, data },
        Tile { x: 1, y: 0, data },
        Tile { x: 0, y: 1, data },
        Tile { x: 1, y: 1, data },
    ];
    // (tiles, origin tile, window rows, window cols)
    let cases: [Case<'_>; 4] = [
        (&single, (5, 7), (0, 256), (0, 256)),
        (&single, (4, 6), (100, 300), (200, 400)),
        (&quad, (0, 0), (128, 384), (128, 384)),
        (&quad, (1, 1), (0, 300), (0, 300)),
    ];
    for (tiles, origin, rows, cols) in cases {
        let height = rows.1 - rows.0;
        let width = cols.1 - cols.0;
        let window = decode_window(tiles, origin, rows, cols).expect("window within the budget");
        assert_eq!(window.rgb.len(), height * width * 3);
        assert_eq!(window.valid.len(), height * width);
        assert!(window.undecoded.len() <= tiles.len());
    }
});
