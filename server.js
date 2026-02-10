// ABOUTME: Dev server with headers for high-resolution timing.
// ABOUTME: Run with `pnpm dev` to start on port 8080.

import express from "express";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();

app.use((req, res, next) => {
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  next();
});

app.use(express.static(__dirname));

app.listen(8080, () => {
  console.log("Dev server running at http://localhost:8080/demo/");
});
