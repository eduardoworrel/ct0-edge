
const nodeImport = (spec: string): Promise<any> => import(  spec);

export async function defaultCacheDir(): Promise<string> {
  const env = (globalThis as any).process?.env ?? {};
  if (env.CT0_CACHE) return env.CT0_CACHE;
  const os = await nodeImport("node:os"), path = await nodeImport("node:path");
  return path.join(env.XDG_CACHE_HOME ?? path.join(os.homedir(), ".cache"), "ct0-edge");
}

export async function localFile(url: string, cacheDir: string, onStatus: (m: string) => void = () => {}): Promise<string> {
  const { fileURLToPath } = await nodeImport("node:url");
  if (url.startsWith("file:")) return fileURLToPath(url);
  if (!/^https?:/i.test(url)) return url;
  const fs = await nodeImport("node:fs"), path = await nodeImport("node:path");
  const u = new URL(url);
  const dest = path.join(cacheDir, u.hostname, ...decodeURIComponent(u.pathname).split("/").filter((s: string) => s && s !== ".."));
  if (fs.existsSync(dest)) return dest;
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const part = `${dest}.part-${(globalThis as any).process?.pid ?? 0}`;
  const name = path.basename(dest);
  try {

    const res = await get(url);
    const total = Number(res.headers["content-length"]) || 0;
    let got = 0, shown = -1;
    res.on("data", (c: Uint8Array) => {
      got += c.length;
      const pct = total ? Math.floor((100 * got) / total / 10) * 10 : -1;
      if (pct !== shown && pct >= 0) { shown = pct; onStatus(`downloading ${name}: ${pct}% of ${(total / 1e9).toFixed(2)} GB`); }
    });
    const { pipeline } = await nodeImport("node:stream/promises");
    await pipeline(res, fs.createWriteStream(part));
    if (total && got !== total) throw new Error(`ct0-edge: download truncated: ${url} (${got} of ${total} bytes)`);
    fs.renameSync(part, dest);
  } catch (e) {
    try { fs.rmSync(part, { force: true }); } catch {  }
    throw e;
  }
  return dest;
}

async function get(url: string, hops = 0): Promise<any> {
  const mod = await nodeImport(url.startsWith("https:") ? "node:https" : "node:http");
  const res: any = await new Promise((resolve, reject) => mod.get(url, { headers: { "user-agent": "ct0-edge" } }, resolve).on("error", reject));
  if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location && hops < 10) { res.resume(); return get(new URL(res.headers.location, url).href, hops + 1); }
  if (res.statusCode !== 200) { res.resume(); throw new Error(`ct0-edge: download failed: ${url} -> HTTP ${res.statusCode}`); }
  return res;
}

export async function readLocalBytes(url: string): Promise<ArrayBuffer> {
  const { fileURLToPath } = await nodeImport("node:url");
  const fs = await nodeImport("node:fs");
  const b = fs.readFileSync(url.startsWith("file:") ? fileURLToPath(url) : url);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
