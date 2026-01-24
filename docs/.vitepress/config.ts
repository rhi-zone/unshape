import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import fs from 'node:fs'
import path from 'node:path'

// Auto-generate sidebar items from a directory
function getSidebarItems(dir: string) {
  const fullPath = path.join(__dirname, '..', dir)
  if (!fs.existsSync(fullPath)) {
    return []
  }

  return fs
    .readdirSync(fullPath)
    .filter((file) => file.endsWith('.md') && file !== 'index.md')
    .map((file) => {
      const name = path.basename(file, '.md')
      // Convert kebab-case to Title Case
      const text = name
        .split('-')
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ')
      return { text, link: `/${dir}/${name}` }
    })
}

export default withMermaid(
  defineConfig({
    vite: {
      optimizeDeps: {
        include: ['mermaid'],
      },
    },
    title: 'Unshape',
    description: 'Constructive media generation and manipulation',

    base: '/unshape/',

    themeConfig: {
      nav: [
        { text: 'Guide', link: '/introduction' },
        { text: 'Design', link: '/philosophy' },
        { text: 'Rhizome', link: 'https://rhi-zone.github.io/' },
      ],

      sidebar: {
        '/': [
          {
            text: 'Guide',
            items: [
              { text: 'Introduction', link: '/introduction' },
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
              { text: 'Open Questions', link: '/open-questions' },
            ]
          },
          {
            text: 'Design Docs',
            collapsed: true,
            items: getSidebarItems('design'),
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
        { icon: 'github', link: 'https://github.com/rhi-zone/unshape' }
      ],

      search: {
        provider: 'local'
      },

      editLink: {
        pattern: 'https://github.com/rhi-zone/unshape/edit/master/docs/:path',
        text: 'Edit this page on GitHub'
      },
    },
  }),
)
