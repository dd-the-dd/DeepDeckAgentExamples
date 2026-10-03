import react from '@vitejs/plugin-react';
import vue from '@vitejs/plugin-vue';
import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [react(), vue()],
  build: {
    // The local workbench can be rebuilt while an Engine game is open. Keep
    // content-hashed chunks from the previous build so an existing tab can
    // still finish loading its lazy Pixi modules.
    emptyOutDir: false,
  },
  resolve: {
    alias: [
      {
        find: /^@deepdeck\/pixi$/,
        replacement: fileURLToPath(
          new URL('../../external/deepdeck-pixi/src/index.mjs', import.meta.url),
        ),
      },
      {
        find: /^pixi\.js$/,
        replacement: fileURLToPath(
          new URL('./node_modules/pixi.js/lib/index.mjs', import.meta.url),
        ),
      },
      {
        find: /^vue$/,
        replacement: fileURLToPath(
          new URL('./node_modules/vue/dist/vue.runtime.esm-bundler.js', import.meta.url),
        ),
      },
    ],
  },
  server: {
    port: 4173,
    proxy: {
      '/api': 'http://127.0.0.1:8765',
    },
    fs: {
      allow: ['../..'],
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test-setup.ts',
  },
});
