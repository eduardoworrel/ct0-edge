#!/bin/zsh
# Build + deploy num passo só — evita drift de bundle stale nos labs.
set -e
cd "$(dirname "$0")"
npx esbuild src/browser.ts --format=esm --target=es2022 --bundle --outfile=dist/browser.js
npx esbuild src/cai.ts     --format=esm --target=es2022 --bundle --outfile=dist/cai.js
npx esbuild src/webllm.ts  --format=esm --target=es2022 --bundle --outfile=dist/webllm.js
for d in /Users/pc/cai/alias3-mlc /Users/pc/cai/ct0-onnx-research/results; do
  [ -d "$d" ] || continue
  cp dist/browser.js "$d/ct0-browser.js"
  cp dist/cai.js     "$d/ct0-cai.js"
  cp dist/webllm.js  "$d/ct0-webllm.js"
  # carimba versão nos imports (cache-bust: evita Chrome servir módulo stale do cache)
  V=$(date +%s)
  for h in "$d"/*.html; do
    [ -f "$h" ] || continue
    perl -i -pe "s/(ct0-(?:browser|cai|webllm)\.js)\?v=[^\"']*/\$1?v=$V/g" "$h"
  done
  echo "deployed -> $d (browser.js $(wc -c < dist/browser.js)b, cai.js $(wc -c < dist/cai.js)b, v=$V)"
done
