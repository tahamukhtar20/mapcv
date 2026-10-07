# Imagery sources and licensing

mapcv is a tool, released under the MIT License without warranty. It does not provide imagery and grants no rights to any: the terms of the source you point it at are between you and its provider.

What mapcv does with imagery:

- **Caching.** Downloaded XYZ tiles stay in an on-disk cache for as long as the server's caching headers allow (7 days when it sends none). `imagery.cache: false` turns the cache off.
- **Requests.** `mapcv plan` reports the number of tiles before anything is fetched. Generated configs use `max_connections: 4`; mapcv retries with backoff, honours `Retry-After` and names itself in its user agent.
- **Credentials.** A key in `url_template` never reaches the manifest, logs or messages: only the host is shown. Remote GeoTIFF, STAC and EOPF URLs cannot carry credentials, and private stores are not supported.
- **Earth Engine** (`imagery.earth_engine`) runs on your own account and Cloud project. mapcv asks for a fresh map URL each run and never writes it, or its map ID, anywhere.
- **No Google Maps or OpenStreetMap tile presets.** mapcv ships no presets for tile servers whose operators ask not to be bulk-downloaded. OpenStreetMap *data* (ODbL) works well as labels.
