import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    vite: {
      optimizeDeps: {
        include: ['mermaid'],
      },
    },
    title: 'Resin',
    description: 'Constructive media generation and manipulation',

    base: '/resin/',

    themeConfig: {
      nav: [
        { text: 'Guide', link: '/index' },
        { text: 'Design', link: '/philosophy' },
      ],

      sidebar: {
        '/': [
          {
            text: 'Guide',
            items: [
              { text: 'Overview', link: '/index' },
              { text: 'Getting Started', link: '/getting-started' },
            ]
          },
          {
            text: 'Design',
            items: [
              { text: 'Philosophy', link: '/philosophy' },
              { text: 'Prior Art', link: '/prior-art' },
              { text: 'Architecture', link: '/architecture' },
              { text: 'Cross-Domain Analysis', link: '/cross-domain-analysis' },
              { text: 'Domain Differences', link: '/domain-differences' },
            ]
          },
          {
            text: 'Domains',
            items: [
              { text: 'Meshes', link: '/domains/meshes' },
              { text: 'Audio', link: '/domains/audio' },
              { text: 'Textures', link: '/domains/textures' },
              { text: '2D Vector', link: '/domains/vector-2d' },
              { text: 'Rigging', link: '/domains/rigging' },
            ]
          },
        ]
      },

      socialLinks: [
        { icon: 'github', link: 'https://github.com/pterror/resin' }
      ],

      search: {
        provider: 'local'
      },

      editLink: {
        pattern: 'https://github.com/pterror/resin/edit/master/docs/:path',
        text: 'Edit this page on GitHub'
      },
    },
  }),
)
