import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Cognitor Assistant',
  description: 'Assistente virtuale con classificazione di intenti usando PyTorch, FastText e NER.',
  lang: 'it',

  themeConfig: {
    logo: '/logo.svg',

    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guida', link: '/guide/' },
      { text: 'API', link: '/api/' },
      {
        text: 'Community',
        link: 'https://github.com/lucaterribili/cognitor-assistant/issues',
      },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Inizia Subito',
          items: [
            { text: 'Introduzione', link: '/guide/' },
            { text: 'Quick Start', link: '/guide/quickstart' },
            { text: 'Installazione', link: '/guide/installation' },
          ],
        },
        {
          text: 'Architettura',
          items: [
            { text: 'Overview', link: '/guide/architecture' },
            { text: 'Knowledge Base', link: '/guide/knowledge-base' },
            { text: 'Training', link: '/guide/training' },
          ],
        },
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'Overview', link: '/api/' },
            { text: 'Autenticazione', link: '/api/auth' },
            { text: 'Chatbot', link: '/api/chatbot' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/lucaterribili/cognitor-assistant' },
    ],

    footer: {
      message: 'Rilasciato sotto licenza MIT.',
      copyright: 'Copyright © 2025 Luca Terribili',
    },

    search: {
      provider: 'local',
    },
  },
})
