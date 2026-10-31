import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The dev server proxies API calls to uvicorn so the browser only ever talks to
    // one origin and CORS never comes into it while developing.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    // Responses carry base64 images, so the bundle itself stays small; this just keeps
    // the warning quiet if a chart library is added later.
    chunkSizeWarningLimit: 900,
  },
});
