#!/bin/zsh
# Build + deploy num passo só — evita drift de bundle stale nos labs.
#
# Os fontes de browser/cai pertencem a OUTROS repos (agent-browser-lite e cai-lite); o
# ct0-edge só carrega o model SDK. Este script junta os três para o lab consumir.
set -e
cd "$(dirname "$0")"
ABL="${ABL:-$PWD/../agent-browser-lite}"
CAIL="${CAIL:-$PWD/../cai-lite}"

[ -f "$ABL/dist/browser.js" ] || { echo "falta $ABL/dist/browser.js — rode npm run build no agent-browser-lite"; exit 1; }
[ -f "$CAIL/dist/agent.js" ]  || { echo "falta $CAIL/dist/agent.js — rode npm run build no cai-lite"; exit 1; }

npx esbuild src/webllm.ts --format=esm --target=es2022 --bundle --outfile=dist/webllm.js
cp "$ABL/dist/browser.js"  dist/browser.js
cp "$CAIL/dist/agent.js"   dist/cai.js

for d in ${LABS:-/Users/pc/cai/alias3-mlc /Users/pc/cai/ct0-onnx-research/results}; do
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
