# Migrating to mapcv 0.2

## Imagery configuration

The imagery source is now a discriminated `imagery` block. Move the XYZ zoom out of `region` and replace `tiles` with `imagery`:

```yaml
region:
  west: 4.883
  south: 52.371
  east: 4.896
  north: 52.378

imagery:
  type: xyz
  zoom: 17
  source: osm
```

Configurations using `region.zoom` plus `tiles` still work in 0.2 and emit a deprecation warning. This compatibility path will be removed in 0.3.0.

## Google imagery preset

The built-in `google_satellite` preset has been removed. mapcv cannot determine whether a use of Google imagery is licensed for downloading, caching, or machine-learning dataset creation. Authorized custom XYZ templates remain supported through `imagery.url_template`; do not put credentials or signed query strings in committed YAML.

## EOPF output and manifests

EOPF Sentinel-2 L2A input requires `pip install "mapcv[zarr]"` and `writer.image_format: npy`. Each image patch has shape `(bands, height, width)` and decoded `float32` values. Band order is exactly the order configured in `imagery.bands`.

New datasets use manifest version 2. It records the source type, product identifier, bands, dtype, patch shape, CRS, and dataset affine transform. Patch rows and columns are global raster coordinates. Version 1 manifests remain readable by `mapcv split` and generation resume logic.
