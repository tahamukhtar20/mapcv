# Security policy

## Supported versions

Security fixes go into the latest minor release. Upgrade to it before reporting.

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a vulnerability

Please report vulnerabilities privately through [GitHub's private vulnerability reporting](https://github.com/tahamukhtar20/mapcv/security/advisories/new), not in public issues or pull requests.

Include what you can of:

- the affected version and platform;
- a config, label file or command that reproduces the problem;
- what an attacker gains, for example reading or writing files outside the dataset folder, running code, or exhausting memory.

You should get a reply within 7 days. Once the issue is confirmed, a fix is released as soon as it is ready and the advisory is published with credit to you, unless you prefer to stay anonymous.

## Scope

mapcv downloads imagery from the URLs in your config and parses label files you give it. Reports are in scope when crafted input (a label file, a tile response, a Zarr product or a manifest) makes mapcv crash, hang, use unbounded memory, write outside `writer.staging_dir`, or leak credentials from `url_template` into output files or logs.

Provider terms of service and imagery licensing are not security issues; see [PROVIDERS.md](PROVIDERS.md).
