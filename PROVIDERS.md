# Imagery providers, licensing, and credentials

mapcv is software for processing imagery; it does not provide imagery or grant permission to download, cache, redistribute, or train models on provider data.

Before generating a dataset:

- review the provider's current terms, license, attribution requirements, quotas, and automated-access rules;
- use only an official endpoint and credentials you are authorized to use;
- keep API keys and signed URLs out of YAML, logs, manifests, examples, and source control;
- confirm that the resulting dataset and trained model may be stored, shared, and published for your use case.

The built-in XYZ names are conveniences, not endorsements or license guarantees. Free community tile servers are usually off-limits: the OpenStreetMap Foundation's servers, for example, forbid bulk downloading, so mapcv ships no OSM tile preset. OSM data itself is open (ODbL) and works well as labels. Custom XYZ URL templates remain supported, but mapcv never persists the template in the manifest. mapcv keeps downloaded XYZ tiles in an on-disk cache for as long as the server's caching headers allow (7 days when it sends none); set `imagery.cache: false` if the provider's terms do not allow local caching. EOPF URLs must be local or anonymously public and cannot include usernames, passwords, query strings, or fragments. Private-store authentication is intentionally not supported in 0.2.0.

Google Earth Engine imagery (`imagery.earth_engine`, the `gee` extra) runs on your own Earth Engine account and Cloud project, with your own login. Earth Engine's terms, its quota tiers for noncommercial projects, any billing for commercial use, and the license of each catalog dataset apply to you; mapcv only makes the tile requests your config asks for and is not responsible for how you use them. mapcv creates a fresh map URL each time it opens the imagery and never writes it, or its map ID, to the config, the manifest, the tile cache or messages.
