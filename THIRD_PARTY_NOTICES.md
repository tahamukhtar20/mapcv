# Third-party notices

mapcv is licensed under the MIT License (see `LICENSE`). Parts of it are
derived from third-party code under the licenses below.

## GDAL

### What was derived

`src/rasterizer.rs` (the polygon rasterizer behind `mapcv.rasterize`) ports
the following GDAL code to Rust, operation for operation, so that its masks
match `rasterio.features.rasterize`:

| mapcv (`src/rasterizer.rs`) | GDAL 3.12.1 source |
|---|---|
| `Scratch::fill` (pixel-centre scanline fill) | `alg/llrasterize.cpp`: `GDALdllImageFilledPolygon()` |
| `walk_edge` (`all_touched` edge walk) | `alg/llrasterize.cpp`: `GDALdllImageLineAllTouched()` |
| `is_clockwise` and the ring reorientation in `Scratch::load` | `ogr/ogrlinestring.cpp`: `OGRLineString::isClockwise()`, used by `GDALCollectRingsFromGeometry()` in `alg/gdalrasterize.cpp` |
| `Affine::inverse`, `Affine::apply` | `alg/gdaltransformer.cpp`: `GDALInvGeoTransform()` and the geotransform step of `GDALGenImgProjTransform()` |

Source: GDAL v3.12.1, <https://github.com/OSGeo/gdal/tree/v3.12.1>.
`alg/llrasterize.cpp` notes that its polygon fill was originally adapted from
`gdImageFilledPolygon()` in libgd and relicensed under the GDAL MIT license.

### Notice

From `alg/llrasterize.cpp`:

```
Copyright (c) 2000, Frank Warmerdam <warmerdam@pobox.com>
Copyright (c) 2011, Even Rouault <even dot rouault at spatialys.com>
```

From `alg/gdalrasterize.cpp`:

```
Copyright (c) 2005, Frank Warmerdam <warmerdam@pobox.com>
Copyright (c) 2008-2013, Even Rouault <even dot rouault at spatialys.com>
```

From `ogr/ogrlinestring.cpp`:

```
Copyright (c) 1999, Frank Warmerdam
Copyright (c) 2008-2014, Even Rouault <even dot rouault at spatialys.com>
```

From `alg/gdaltransformer.cpp`:

```
Copyright (c) 2002, i3 - information integration and imaging
                         Fort Collin, CO
Copyright (c) 2008-2013, Even Rouault <even dot rouault at spatialys.com>
Copyright (c) 2021, CLS
```

These files are `SPDX-License-Identifier: MIT`, under the terms in GDAL's
`LICENSE.TXT`:

```
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```
