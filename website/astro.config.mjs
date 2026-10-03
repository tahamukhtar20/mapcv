import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://tahamukhtar20.github.io',
  base: '/mapcv',
  integrations: [
    starlight({
      title: 'mapcv',
      favicon: '/favicon.svg',
      customCss: ['./src/styles/custom.css'],
      logo: {
        src: './src/assets/logo.svg',
        alt: 'mapcv',
        replacesTitle: true,
      },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/tahamukhtar20/mapcv' },
      ],
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Introduction', slug: 'introduction' },
            { label: 'Installation', slug: 'installation' },
            { label: 'Migrating to 0.2', slug: 'migration' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'Configuration', slug: 'configuration' },
            { label: 'CLI', slug: 'cli' },
            { label: 'API', slug: 'api' },
            { label: 'Provider Terms', slug: 'providers' },
          ],
        },
        {
          label: 'Examples',
          items: [
            { label: 'Examples', slug: 'examples' },
          ],
        },
      ],
      credits: false,
    }),
  ],
});
