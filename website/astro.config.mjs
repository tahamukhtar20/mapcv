import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';

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
      plugins: [starlightLinksValidator({ errorOnLocalLinks: true })],
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
            { label: 'Train a model on your dataset', slug: 'tutorials/train-a-model' },
          ],
        },
        {
          label: 'How-to guides',
          items: [
            { label: 'Use your dataset', slug: 'guides/use-your-dataset' },
            { label: 'Prepare labels', slug: 'guides/prepare-labels' },
            { label: 'Choose imagery and zoom', slug: 'guides/choose-imagery-and-zoom' },
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
