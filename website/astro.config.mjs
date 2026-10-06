import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import starlightLlmsTxt from 'starlight-llms-txt';

export default defineConfig({
  site: 'https://tahamukhtar20.github.io',
  base: '/mapcv',
  // The changelog page renders ../CHANGELOG.md from the repository root.
  vite: {
    server: { fs: { allow: ['..'] } },
    build: {
      rollupOptions: {
        // Astro's MDX pipeline emits one harmless "use astro:head-inject" directive
        // warning per page (also on the pre-revamp site); keep other warnings visible.
        onwarn(warning, warn) {
          if (warning.code === 'MODULE_LEVEL_DIRECTIVE') return;
          warn(warning);
        },
      },
    },
  },
  // Pages that moved in the 0.2 docs revamp keep working from old links.
  redirects: {
    '/configuration': '/mapcv/reference/configuration/',
    '/cli': '/mapcv/reference/cli/',
    '/api': '/mapcv/reference/python-api/',
    '/examples': '/mapcv/tutorials/buildings-from-aerial-imagery/',
    '/migration': '/mapcv/project/migration/',
    '/providers': '/mapcv/project/providers/',
  },
  integrations: [
    starlight({
      title: 'mapcv',
      description:
        'Turn a region and polygon labels into a ready-to-train semantic-segmentation dataset.',
      favicon: '/favicon.svg',
      customCss: ['./src/styles/custom.css'],
      logo: {
        src: './src/assets/logo.svg',
        alt: 'mapcv',
      },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/tahamukhtar20/mapcv' },
      ],
      lastUpdated: false,
      disable404Route: true,
      plugins: [
        starlightLinksValidator({ errorOnLocalLinks: true }),
        // /llms.txt (index), /llms-full.txt (every page) and /llms-small.txt, built from the pages.
        starlightLlmsTxt({
          projectName: 'mapcv',
          description:
            'mapcv turns a region, imagery and labels into ready-to-train remote-sensing datasets: ' +
            'semantic segmentation, object detection (COCO, YOLO) and instance segmentation (COCO RLE). ' +
            'It is a GDAL-free Python and Rust library and CLI.',
          details: [
            '- Install with `pip install mapcv`; the CLI journey is `mapcv init`, `plan`, `generate`, `info`.',
            '- Imagery: XYZ tiles, Sentinel-2 L2A (EOPF Zarr) and GeoTIFF/COG. Labels: GeoJSON, KML, GeoPackage, Shapefile or GeoParquet polygons, or a label raster.',
            '- Always run `mapcv plan` before `mapcv generate`; it downloads nothing.',
            '- Imagery providers have terms the user is responsible for.',
            '- AI agents can drive mapcv through its MCP server (`mapcv mcp`); see the page "Use mapcv with AI agents".',
          ].join('\n'),
          optionalLinks: [
            {
              label: 'Source code and issues',
              url: 'https://github.com/tahamukhtar20/mapcv',
              description: 'Repository, issue tracker and releases',
            },
            {
              label: 'Imagery providers and licensing',
              url: 'https://github.com/tahamukhtar20/mapcv/blob/main/PROVIDERS.md',
              description: 'What to check before downloading tiles',
            },
          ],
          demote: ['project/changelog'],
          exclude: ['project/changelog', 'project/migration'],
        }),
      ],
      sidebar: [
        {
          label: 'Get started',
          items: [
            { label: 'Introduction', slug: 'introduction' },
            { label: 'Installation', slug: 'installation' },
            { label: 'Quickstart', slug: 'quickstart', badge: { text: '5 min', variant: 'tip' } },
          ],
        },
        {
          label: 'Tutorials',
          items: [
            { label: 'Buildings from aerial imagery', slug: 'tutorials/buildings-from-aerial-imagery' },
            { label: 'Land cover from Sentinel-2', slug: 'tutorials/land-cover-from-sentinel-2' },
            { label: 'Object detection datasets', slug: 'tutorials/object-detection' },
            { label: 'Instance segmentation datasets', slug: 'tutorials/instance-segmentation' },
            { label: 'Train a model on your dataset', slug: 'tutorials/train-a-model' },
          ],
        },
        {
          label: 'How-to guides',
          items: [
            { label: 'Use your dataset', slug: 'guides/use-your-dataset' },
            { label: 'Prepare labels', slug: 'guides/prepare-labels' },
            { label: 'Choose imagery and zoom', slug: 'guides/choose-imagery-and-zoom' },
            { label: 'Use your own GeoTIFF', slug: 'guides/use-your-own-geotiff' },
            { label: 'Use mapcv with AI agents', slug: 'guides/use-with-ai-agents' },
            { label: 'Large regions and resuming', slug: 'guides/large-regions-and-resuming' },
            { label: 'Splits without leakage', slug: 'guides/splits-without-leakage' },
            { label: 'Troubleshooting', slug: 'guides/troubleshooting' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'CLI', slug: 'reference/cli' },
            { label: 'Configuration', slug: 'reference/configuration' },
            { label: 'Python API', slug: 'reference/python-api' },
            { label: 'Dataset format', slug: 'reference/dataset-format' },
          ],
        },
        {
          label: 'Concepts',
          items: [
            { label: 'How mapcv works', slug: 'concepts/how-mapcv-works' },
            { label: 'Coordinates and grids', slug: 'concepts/coordinates-and-grids' },
            { label: 'Spatial leakage', slug: 'concepts/spatial-leakage' },
          ],
        },
        {
          label: 'Project',
          collapsed: true,
          items: [
            { label: 'Providers & licensing', slug: 'project/providers' },
            { label: 'Migrating', slug: 'project/migration' },
            { label: 'Changelog', slug: 'project/changelog' },
            { label: 'Contributing', slug: 'project/contributing' },
            { label: 'Citing mapcv', slug: 'project/citing' },
            { label: 'FAQ', slug: 'project/faq' },
          ],
        },
      ],
      credits: false,
    }),
  ],
});
