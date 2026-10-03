# Imagery providers, licensing, and credentials

mapcv is software for processing imagery; it does not provide imagery or grant permission to download, cache, redistribute, or train models on provider data.

Before generating a dataset:

- review the provider's current terms, license, attribution requirements, quotas, and automated-access rules;
- use only an official endpoint and credentials you are authorized to use;
- keep API keys and signed URLs out of YAML, logs, manifests, examples, and source control;
- confirm that the resulting dataset and trained model may be stored, shared, and published for your use case.

The built-in XYZ names are conveniences, not endorsements or license guarantees. Free community tile servers are usually off-limits: the OpenStreetMap Foundation's servers, for example, forbid bulk downloading, so mapcv ships no OSM tile preset. OSM data itself is open (ODbL) and works well as labels. Custom XYZ URL templates remain supported, but mapcv never persists the template in manifest v2. EOPF URLs must be local or anonymously public and cannot include usernames, passwords, query strings, or fragments. Private-store authentication is intentionally not supported in 0.2.0.
