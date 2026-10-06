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

## jpeg-encoder (Independent JPEG Group)

The wheels link the `jpeg-encoder` crate (`(MIT OR Apache-2.0) AND IJG`), which
writes the JPEG patches. Its DCT code is ported from the Independent JPEG Group's
software, so this software is based in part on the work of the Independent JPEG
Group.

## COCO API (Microsoft COCO Toolbox)

`src/mapcv/_rle.py` writes the compressed run-length strings of COCO instance
segmentations. Its `counts_to_string` follows `rleToString()` in `common/maskApi.c` of
the COCO API (<https://github.com/cocodataset/cocoapi>), Simplified BSD licence; the
run-length computation around it is mapcv's own. pycocotools, which builds on the same
code, is used only in mapcv's tests, to check the output.

```
Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
```
