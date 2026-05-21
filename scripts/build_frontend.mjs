import { mkdir, copyFile, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const root = process.cwd();
const outDir = path.join(root, "artifacts", "frontend");

async function main() {
  await mkdir(outDir, { recursive: true });

  const htmlPath = path.join(root, "webapp", "templates", "index.html");
  const jsPath = path.join(root, "webapp", "static", "app.js");
  const cssPath = path.join(root, "webapp", "static", "styles.css");

  await copyFile(htmlPath, path.join(outDir, "index.html"));
  await copyFile(cssPath, path.join(outDir, "styles.css"));

  // Lightweight minification for CI artifact output.
  const jsRaw = await readFile(jsPath, "utf8");
  const jsMin = jsRaw
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^\s*\/\/.*$/gm, "")
    .replace(/\n{2,}/g, "\n")
    .trim();
  await writeFile(path.join(outDir, "app.min.js"), jsMin, "utf8");

  const manifest = {
    built_at_utc: new Date().toISOString(),
    files: ["index.html", "styles.css", "app.min.js"],
  };
  await writeFile(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
  console.log(`Frontend build artifacts written to ${outDir}`);
}

main().catch((error) => {
  console.error("Frontend build failed:", error);
  process.exit(1);
});

