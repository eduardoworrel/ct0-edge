/**
 * ct0.ts — CT0 (alias3-micro-1026) SDK for the browser and Node. One entry point, `CT0.load({ runtime, device })`, over
 * four runtimes that share the `Backend` contract (see runtime.ts):
 *
 *   const ct0 = await CT0.load();                               // auto: best measured runtime for this machine
 *   const ct0 = await CT0.load({ runtime: "genai" });           // ONNX Runtime GenAI graph
 *   const ct0 = await CT0.load({ runtime: "decode-only" });     // ONNX Runtime Web / Node
 *   const ct0 = await CT0.load({ runtime: "transformers.js" }); // optimum-exported graph via transformers.js
 *   const ct0 = await CT0.load({ runtime: "webllm" });          // MLC
 *
 * Same API everywhere: ask() · askMany() · generate() · stream() · dispose(); ct0.runtime / ct0.device / ct0.fallbackReason.
 * Runtime packages are optional peers, imported only when the chosen runtime needs them.
 */
import type { Backend, GenerateOptions } from "./backend.js";
import { createDecodeOnly } from "./backends/decodeOnly.js";
import { createGenai, genaiFromSession, type GenaiOptions } from "./backends/genai.js";
import { transformersJsFromModel } from "./backends/transformersjs.js";
import { createWebllm, tryLoadRecovery, type Recovery } from "./webllm.js";
import { tokenOffsets, decidePos, optPositionsFor, type Probe } from "./positions.js";
import { ask as askImpl, askManyWith, type AskHeads, type AskOptions, type AskResult, type AskQuestion, type AskManyQuestion } from "./ask.js";
import { loadHead } from "./systemone.js";
import { checkOrtVersion } from "./ortVersion.js";
import {
  RUNTIME_DEFAULTS, defaultMethods, maxSafeTokensFor, resolveRuntimeChoice, detectPlatform, planRuntime, selectRuntime,
  importOptional, detectCuda, detectWebGPU, DEFAULT_REPOS, DEFAULT_FILES, type RuntimeName, type RuntimeChoice, type DeviceChoice, type Device, type Caps, type Plan, type AskMethod,
} from "./runtime.js";
import { defaultCacheDir, localFile, readLocalBytes } from "./nodeFiles.js";
export type { AskHeads, AskOptions, AskResult, AskQuestion, AskManyQuestion } from "./ask.js";
export { DEFAULT_METHOD, askMany, levelFromCuts } from "./ask.js";
export type { Backend, GenerateOptions, StateHandle } from "./backend.js";
export { resolveGenerateOptions, GENERATE_DEFAULTS } from "./backend.js";
export {
  RUNTIMES, RUNTIME_LIMITS, RUNTIME_DEFAULTS, MLC_Q4F32_LIB, GENAI_NODE_CPU_ERROR, maxSafeTokensFor, defaultMethods,
  resolveRuntimeChoice, detectPlatform, detectCuda, detectWebGPU, planRuntime, selectRuntime, packagesFor, importOptional,
  DEFAULT_REPOS, DEFAULT_FILES,
} from "./runtime.js";
export type { RuntimeName, RuntimeChoice, Device, DeviceChoice, Platform, Caps, Plan, SelectRuntimeDeps, SelectedRuntime } from "./runtime.js";

/** WebLLM (MLC) options for `runtime: "webllm"`. */
export interface MlcOptions {
  /** MLC build: `"q4f32_1"` (default, faithful; its lib is the official WebLLM prebuilt wasm) or `"q4f16_1"` (faster, less faithful; wasm in its repo). */
  build?: "q4f16_1" | "q4f32_1";
  /** Model base URL (default `https://huggingface.co/<build repo>/`). */
  modelUrl?: string;
  /** Model id in the WebLLM app config (default `alias3-micro-1026-<build>`). */
  modelId?: string;
  /** Model lib (wasm) URL (default: the build repo's `model-lib.wasm`, or the official prebuilt for q4f32_1). */
  modelLib?: string;
  /** Full WebLLM app config (replaces modelUrl / modelId / modelLib). */
  appConfig?: any;
  /** Hidden-state recovery (h = P·logits[rows]) that enables the head reads. Default: `recovery/` of the build's repo.
   *  `false` skips it (every mode reads the logits). */
  recovery?: Recovery | false;
  recoveryUrls?: { rows: string; P: string; format?: "f32" | "f16" };
  initProgressCallback?: (p: any) => void;
}

export interface LoadOptions {
  /** `"auto"` (default): the best measured runtime for this machine. `"genai"`, `"decode-only"`, `"transformers.js"`, `"webllm"` force one. */
  runtime?: RuntimeChoice;
  /** `"auto"` (default), `"webgpu"` / `"wasm"` (browser), `"cpu"` / `"cuda"` (Node). */
  device?: DeviceChoice;
  /** Tokenizer repo and decode-only repo (default `eduardoworrel/alias3-micro-1026-ONNX-decode-only`). */
  repo?: string;
  /** Self-hosted decode-only files (URL, or a directory in Node). Without `runtime`, `"auto"` then uses only decode-only. */
  baseUrl?: string;
  /** genai repo (default `eduardoworrel/alias3-micro-1026-ONNX-genai`). */
  genaiRepo?: string;
  /** Self-hosted genai files (replaces `genaiRepo`). */
  genaiBaseUrl?: string;
  /** genai graph inside its base (default `webgpu/int4/model.onnx` in the browser, `cuda/int4/model.onnx` in Node; data `<graph>.data`). */
  genaiModel?: string;
  /** Browser genai: check once that the graph runs on the WebGPU EP (default false: the probe costs an extra session). */
  verifyEp?: boolean | "force";
  /** Extra genai backend options (chunk, ...). */
  genai?: Omit<GenaiOptions, "ort">;
  /** decode-only graph (default `onnx/model_q4.onnx`, data `<graph>_data`). */
  decodeModel?: string;
  /** decode-only execution providers (overrides the ones derived from `device`). */
  providers?: string[];
  /** decode-only external data file name, if not `decodeModel + "_data"`. */
  externalDataFile?: string;
  /** decode-only initial KV past length (default 1, as in the reference implementation). */
  pastLen?: number;
  /** transformers.js model repo (default `eduardoworrel/alias3-micro-1026-ONNX`). */
  transformersRepo?: string;
  /** transformers.js dtype (default `"q4"`). */
  dtype?: string;
  /** WebLLM options. */
  mlc?: MlcOptions;
  /** Use these modules instead of importing them (custom builds, or bundlers that cannot resolve an optional import). */
  modules?: { ort?: any; transformers?: any; webllm?: any };
  /** Node: where downloaded graphs are cached (default `$CT0_CACHE` or `~/.cache/ct0-edge`). */
  cacheDir?: string;
  /** Load `heads/head.bin` and `heads/head_score.bin` from the runtime's repo (default true; webllm: when its recovery loaded). */
  heads?: boolean;
  /** Longest prompt `ask()` accepts (default per runtime: `RUNTIME_LIMITS`). */
  maxSafeTokens?: number;
  /** Do not require onnxruntime-web >= 1.30 (experiments only). */
  skipVersionCheck?: boolean;
  onStatus?: (msg: string) => void;
}

export class CT0 {
  /** Nominal model context (linear recurrence). */
  readonly maxContext = 262144;
  /** Longest prompt `ask()` accepts (default per runtime). */
  maxSafeTokens = 2048;
  private tok: any; private backend: Backend;
  private yesIds: number[] = []; private noIds: number[] = [];
  private constructor(tok: any, backend: Backend) { this.tok = tok; this.backend = backend; }
  private heads: AskHeads = {};
  private methods: Readonly<Record<"noul" | "choice" | "score", AskMethod>> = defaultMethods("decode-only");
  /** Runtime in use. */
  runtime!: RuntimeName;
  /** Device in use: "webgpu" | "wasm" | "cpu" | "cuda". */
  device!: Device;
  /** Why `"auto"` did not use genai (undefined when it did, or when the runtime was forced). */
  fallbackReason?: string;
  /** Releases the graph / session. */
  dispose(): Promise<void> { return this.backend.dispose(); }
  /** Hidden-state output name of the graph (null when absent). */
  get hiddenOutput(): string | null { return this.backend.hiddenName; }
  /** `ask()` default read per mode for this runtime. */
  get defaultMethod(): Readonly<Record<"noul" | "choice" | "score", AskMethod>> { return this.methods; }

  static async load(opts: LoadOptions = {}): Promise<CT0> {
    const status = opts.onStatus ?? (() => {});
    const platform = detectPlatform();
    // Self-hosted decode-only only (baseUrl without genai): "auto" does not reach for genai on the HF hub.
    const { choice } = resolveRuntimeChoice(opts, opts.baseUrl && !opts.genaiBaseUrl && !opts.genaiRepo ? "decode-only" : "auto");
    const device: DeviceChoice = opts.device ?? "auto";
    const webgpu = platform === "browser" ? await detectWebGPU() : false;
    // impossible pairs fail before anything is imported or downloaded
    planRuntime(choice, device, { platform, webgpu, cuda: true });

    const mods = opts.modules ?? {};
    const needsOrt = choice === "auto" || choice === "genai" || choice === "decode-only";
    const loadOrt = async (): Promise<any> => {
      if (mods.ort) return mods.ort;
      if (platform === "node") return importOptional("onnxruntime-node", () => import(/* webpackIgnore: true */ /* @vite-ignore */ NODE_ORT), `runtime "${choice}" in Node`);
      return importOptional("onnxruntime-web", () => import("onnxruntime-web/webgpu"), `runtime "${choice}" in the browser`);
    };
    let ort: any = null;
    let cuda = false;
    if (platform === "node" && (needsOrt || device === "cuda")) {
      try { ort = await loadOrt(); cuda = detectCuda(ort); }
      catch (e) { if (needsOrt) throw e; }
    }
    const caps: Caps = { platform, webgpu, cuda };

    status("loading tokenizer…");
    const tjs: any = mods.transformers ?? await importOptional("@huggingface/transformers", () => import("@huggingface/transformers"), "ct0-edge (tokenizer)");
    const env = tjs.env;
    const pageBase = (): string | undefined => {
      if (typeof location !== "undefined") return location.href;
      const cwd = (globalThis as any).process?.cwd?.();
      return cwd ? "file://" + encodeURI(cwd.replace(/\\/g, "/").replace(/^(?!\/)/, "/")) + "/" : undefined;
    };
    // a base that is not a URL resolves against the page (browser) or the working directory (Node)
    const absBase = (b: string) => new URL(b, pageBase()).href;
    const at = (p: string, b: string) => new URL(p, absBase(b)).href;
    const mlc = opts.mlc ?? {};
    const mlcBuild = RUNTIME_DEFAULTS.webllm.builds[mlc.build ?? RUNTIME_DEFAULTS.webllm.build];
    if (choice === "webllm" && !mlcBuild) throw new Error(`ct0-edge: unknown MLC build "${mlc.build}" (use "q4f32_1" or "q4f16_1")`);
    // webllm: the tokenizer of the MLC repo itself
    const repo = opts.repo ?? (choice === "webllm" ? mlcBuild.repo : opts.baseUrl ? undefined : DEFAULT_REPOS.decodeOnly);
    let tok: any, base: string;
    if (repo) {
      // Remote HF first: with allowLocalModels=true transformers.js tries ./models/<repo>/ first and, if the server answers
      // 404 instead of failing, the load hangs.
      env.allowRemoteModels = true; env.allowLocalModels = false;
      tok = await tjs.AutoTokenizer.from_pretrained(repo, { local_files_only: false });
      base = opts.baseUrl ?? `https://huggingface.co/${repo}/resolve/main/`;
    } else {
      base = absBase(opts.baseUrl!);
      env.allowRemoteModels = false; env.allowLocalModels = true;
      // transformers.js >= 4 tries the "local" path only when it is NOT an absolute URL; same origin -> pathname; Node -> directory
      const bu = new URL(base);
      if (bu.protocol === "file:") {
        const { fileURLToPath } = await import(/* webpackIgnore: true */ /* @vite-ignore */ NODE_URL);
        env.localModelPath = fileURLToPath(bu);
      } else env.localModelPath = typeof location !== "undefined" && bu.origin === location.origin ? bu.pathname : base;
      tok = await tjs.AutoTokenizer.from_pretrained("tok");
    }
    const genaiBase = opts.genaiBaseUrl ?? `https://huggingface.co/${opts.genaiRepo ?? DEFAULT_REPOS.genai}/resolve/main/`;
    const tjsRepo = opts.transformersRepo ?? RUNTIME_DEFAULTS["transformers.js"].repo;
    const tjsBase = `https://huggingface.co/${tjsRepo}/resolve/main/`;
    let cacheDir: string | undefined;
    const fileFor = async (url: string) => {
      if (platform !== "node") return url;
      cacheDir ??= opts.cacheDir ?? await defaultCacheDir();
      return localFile(url, cacheDir, status);
    };

    const loadDecodeOnly = async (dev: Device): Promise<Backend> => {
      ort ??= await loadOrt();
      if (platform === "browser" && !opts.skipVersionCheck) checkOrtVersion(ort.env?.versions?.web);
      const file = opts.decodeModel ?? RUNTIME_DEFAULTS["decode-only"].file;
      // the external .data name can differ from the .onnx (variants of one graph share weights)
      const dataPath = opts.externalDataFile ?? (file + "_data");
      const providers = opts.providers ?? [dev];
      // "./" or "/" = LOCAL path (next to the page / working directory), not resolved against the HF base
      const isLocal = /^(\.\/|\/)/.test(dataPath) || /^(\.\/|\/)/.test(file);
      const modelUrl = isLocal ? at(file, pageBase()!) : at(file, base);
      const dataUrl = isLocal ? at(dataPath, pageBase()!) : at(dataPath, base);
      status(`loading decode-only graph (${providers.join(",")})…`);
      let S: any;
      if (platform === "node") {
        await fileFor(dataUrl); // lands next to the graph, under the name the graph declares
        S = await ort.InferenceSession.create(await fileFor(modelUrl), { executionProviders: providers, logSeverityLevel: 3 });
      } else {
        S = await ort.InferenceSession.create(modelUrl, {
          executionProviders: providers as any,
          externalData: [{ path: dataPath.split("/").pop()!, data: dataUrl }],
        });
      }
      try { return await createDecodeOnly(S, tok, { ort, pastLen: opts.pastLen, skipVersionCheck: opts.skipVersionCheck }); }
      catch (e) { try { await S.release?.(); } catch { /* */ } throw e; }
    };
    const loadGenai = async (dev: Device): Promise<Backend> => {
      ort ??= await loadOrt();
      const file = opts.genaiModel ?? (dev === "cuda" ? RUNTIME_DEFAULTS.genai.files.cuda : RUNTIME_DEFAULTS.genai.files.webgpu);
      status(`loading genai int4 graph (${dev})…`);
      if (platform === "browser")
        return createGenai(at(file, genaiBase), at(file + ".data", genaiBase),
          { ...opts.genai, verifyEp: opts.verifyEp ?? opts.genai?.verifyEp ?? false, ort, skipVersionCheck: opts.skipVersionCheck ?? opts.genai?.skipVersionCheck });
      await fileFor(at(file + ".data", genaiBase));
      const S = await ort.InferenceSession.create(await fileFor(at(file, genaiBase)), { executionProviders: [dev], graphOptimizationLevel: "all", logSeverityLevel: 3 });
      try {
        const be = genaiFromSession(S, ort, opts.genai);
        if (opts.genai?.warmup ?? true) await be.prefill([248045], { positions: [] });
        return be;
      } catch (e) { try { await S.release?.(); } catch { /* */ } throw e; }
    };
    const loadTransformers = async (dev: Device): Promise<Backend> => {
      const dtype = opts.dtype ?? RUNTIME_DEFAULTS["transformers.js"].dtype;
      status(`loading transformers.js graph (${dtype}, ${dev})…`);
      env.allowRemoteModels = true; env.allowLocalModels = false;
      if (platform === "node" && opts.cacheDir) env.cacheDir = opts.cacheDir;
      const model = await tjs.AutoModelForCausalLM.from_pretrained(tjsRepo, { dtype, device: dev });
      return transformersJsFromModel(model, tjs.Tensor);
    };
    let mlcBase: string | null = null, mlcNote: string | undefined, webllmRecovery = false;
    const loadWebllm = async (): Promise<Backend> => {
      const webllm = mods.webllm ?? await importOptional("@mlc-ai/web-llm", () => import(/* webpackIgnore: true */ /* @vite-ignore */ WEBLLM_PKG), 'runtime "webllm"');
      const build = mlcBuild;
      const modelUrl = mlc.modelUrl ?? `https://huggingface.co/${build.repo}/`;
      const modelLib = mlc.modelLib ?? (/^https?:/.test(build.lib) ? build.lib : new URL("resolve/main/" + build.lib, modelUrl).href);
      mlcBase = new URL("resolve/main/", modelUrl).href;
      let recovery: Recovery | undefined = mlc.recovery || undefined;
      if (!recovery && mlc.recovery !== false) {
        status("loading WebLLM hidden-state recovery…");
        const files = mlc.recoveryUrls ?? build.recovery;
        const r = await tryLoadRecovery(mlcBase, files);
        recovery = r.recovery;
        if (!recovery) { mlcNote = `webllm recovery unavailable (${r.reason}): every question type reads the logits`; status(mlcNote); }
      }
      webllmRecovery = !!recovery;
      status(`loading WebLLM build ${mlc.build ?? RUNTIME_DEFAULTS.webllm.build}…`);
      return createWebllm(webllm, { modelId: mlc.modelId ?? build.modelId, modelUrl, libUrl: modelLib, appConfig: mlc.appConfig, tok,
        recovery, initProgressCallback: mlc.initProgressCallback });
    };
    const load = (p: Plan): Promise<Backend> => {
      switch (p.runtime) {
        case "genai": return loadGenai(p.device);
        case "decode-only": return loadDecodeOnly(p.device);
        case "transformers.js": return loadTransformers(p.device);
        case "webllm": return loadWebllm();
      }
    };

    const sel = await selectRuntime(choice, device, { caps, load, onStatus: status });
    const ct0 = new CT0(tok, sel.backend);
    ct0.runtime = sel.runtime; ct0.device = sel.device;
    ct0.fallbackReason = [sel.fallbackReason, sel.runtime === "webllm" ? mlcNote : undefined].filter(Boolean).join("; ") || undefined;
    ct0.methods = defaultMethods(sel.runtime, { webllmRecovery });
    ct0.maxSafeTokens = opts.maxSafeTokens ?? maxSafeTokensFor(sel.runtime, sel.device);
    status(`runtime: ${sel.runtime} (${sel.device})`);
    const headsBase = sel.runtime === "genai" ? genaiBase : sel.runtime === "decode-only" ? base
      : sel.runtime === "transformers.js" ? tjsBase : sel.runtime === "webllm" && webllmRecovery ? mlcBase : null;
    if ((opts.heads ?? true) && headsBase) {
      try { await ct0.loadHeads({ noul: at(DEFAULT_FILES.head, headsBase), score: at(DEFAULT_FILES.headScore, headsBase) }); }
      catch (e: any) { status("heads not loaded (" + (e?.message ?? e) + "); ask() with method head will throw"); }
    }
    const firstId = (s: string) => { const a = ct0.ids(s); return a.length ? a[0] : null; };
    ct0.yesIds = [...new Set(["Yes", " Yes", "yes", " yes", "YES"].map(firstId).filter((x): x is number => x != null))];
    ct0.noIds = [...new Set(["No", " No", "no", " no", "NO"].map(firstId).filter((x): x is number => x != null))];
    status("ready.");
    return ct0;
  }

  private ids(text: string): number[] { return this.tok(text, { add_special_tokens: false }).input_ids.tolist()[0].map(Number); }

  /** Counts the tokens of a text. */
  countTokens(text: string): number { return this.ids(text).length; }

  /** Truncates `text` to fit `maxSafeTokens - reserve` (keeps the start + end,
   *  where the information usually is). Calls `onTrim` if it cuts. */
  fitContext(text: string, opts: { reserve?: number; onTrim?: (i: { original: number; kept: number }) => void } = {}): string {
    const reserve = opts.reserve ?? 300;
    const budget = Math.max(64, this.maxSafeTokens - reserve);
    const toks = this.ids(text);
    if (toks.length <= budget) return text;
    const head = Math.floor(budget * 0.6), tail = budget - head;
    const kept = [...toks.slice(0, head), ...toks.slice(toks.length - tail)];
    opts.onTrim?.({ original: toks.length, kept: budget });
    return this.tok.decode(kept, { skip_special_tokens: true });
  }

  /**
   * chunk-watch: scans a LARGE text (over maxSafeTokens, e.g. huge HTML) in sliding windows,
   * running System 1 (`decide`) per chunk and accumulating the suspicious regions.
   * Covers documents that do not fit the context without truncating or losing signal.
   */
  async scanLong(text: string, question: string,
    opts: { windowTokens?: number; overlapTokens?: number; threshold?: number; onChunk?: (i: number, p: number, total: number) => void } = {}
  ): Promise<{ chunks: number; max: number; hits: { i: number; start: number; p: number; preview: string }[]; flagged: { i: number; start: number; p: number; preview: string }[] }> {
    const win = opts.windowTokens ?? (this.maxSafeTokens - 400);
    const overlap = opts.overlapTokens ?? 128;
    const thr = opts.threshold ?? 0.5;
    const toks = this.ids(text);
    const hits: { i: number; start: number; p: number; preview: string }[] = [];
    if (toks.length <= win) { const p = await this.decide(text, question); opts.onChunk?.(0, p, 1); hits.push({ i: 0, start: 0, p, preview: text.slice(0, 100) }); }
    else {
      const total = Math.ceil((toks.length - overlap) / (win - overlap));
      let i = 0, idx = 0;
      while (i < toks.length) {
        const chunk = this.tok.decode(toks.slice(i, i + win), { skip_special_tokens: true });
        const p = await this.decide(chunk, question);
        hits.push({ i: idx, start: i, p, preview: chunk.slice(0, 100) });
        opts.onChunk?.(idx, p, total); idx++; i += (win - overlap);
      }
    }
    return { chunks: hits.length, max: Math.max(...hits.map(h => h.p)), hits, flagged: hits.filter(h => h.p >= thr) };
  }

  /** Clean, condensed view of a page (for security agents): forms, inputs, buttons, links,
   *  HTML comments, inline scripts and localStorage, where vulnerabilities/secrets tend to hide, in little context. */
  static inspectPage(doc: any): any {
    const q = (s: string, f: (e: any) => any) => [...doc.querySelectorAll(s)].map(f);
    const comments: string[] = []; const w = doc.createTreeWalker(doc.body || doc.documentElement, 128 /* SHOW_COMMENT */);
    while (w.nextNode()) comments.push((w.currentNode.nodeValue || "").trim().slice(0, 180));
    const store: Record<string, string> = {};
    try { const ls = (doc.defaultView || window).localStorage; for (let i = 0; i < ls.length; i++) { const k = ls.key(i)!; store[k] = (ls.getItem(k) || "").slice(0, 80); } } catch {}
    return {
      title: doc.title, url: (doc.defaultView || {}).location?.search || "/",
      forms: q("form", (f: any) => ({ action: f.getAttribute("action"), method: f.method, inputs: [...f.querySelectorAll("input")].map((i: any) => ({ name: i.name, type: i.type, value: i.type === "hidden" ? i.value : undefined, autocomplete: i.autocomplete })) })),
      inputs: q("input,textarea", (i: any) => ({ sel: i.id ? "#" + i.id : i.name ? `[name=${i.name}]` : i.tagName.toLowerCase(), type: i.type })),
      buttons: q("button", (b: any) => ({ sel: b.id ? "#" + b.id : null, text: b.textContent.trim() })),
      links: q("a[href]", (a: any) => a.getAttribute("href")),
      // notable visible text (stack traces, error/debug banners), where credentials tend to leak
      texts: q("pre,code,[class*='debug'],[class*='error'],[class*='stack']", (e: any) => (e.textContent || "").trim().slice(0, 300)).filter(Boolean),
      comments, inlineScripts: q("script:not([src])", (s: any) => s.textContent.trim().slice(0, 220)).filter(Boolean), localStorage: store,
    };
  }

  /** Raw body HTML (fallback when a security review needs the full markup). */
  static pageSource(doc: any, max = 2500): string { return (doc.body || doc.documentElement).outerHTML.slice(0, max); }
  private turn(msg: string, system: string | undefined, think: boolean) {
    let t = ""; if (system) t += "<|im_start|>system\n" + system + "<|im_end|>\n";
    t += "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n";
    return t + (think ? "<think>\n" : "<think>\n\n</think>\n\n");
  }

  /** Single-turn generation: wraps `prompt` in a chat turn (System 2). */
  async *generate(prompt: string, opts: GenerateOptions = {}): AsyncGenerator<string> {
    yield* this.stream(this.turn(prompt, opts.system, opts.think ?? true), opts);
  }

  /**
   * Low-level streaming from a RAW prompt (no chat-turn wrapping). Use this to
   * drive multi-turn / agentic (ReAct) loops where you build the full prompt
   * yourself. `temperature <= 0` ⇒ greedy (deterministic tool calls).
   */
  async *stream(prompt: string, opts: GenerateOptions = {}): AsyncGenerator<string> {
    for await (const tid of this.backend.generate(this.ids(prompt), opts)) yield this.tok.decode([tid], { skip_special_tokens: true });
  }

  async decide(state: string, question: string): Promise<number> {
    const sys = "You are a security decision function. Given a STATE and a yes/no QUESTION, answer with exactly one word: Yes or No.";
    const text = "<|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\nSTATE:\n" + state +
      "\n\nQUESTION: " + question + "\nAnswer Yes or No.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const { lastLogits: L } = await this.backend.prefill(this.ids(text), { positions: [] });
    const all = [...this.yesIds, ...this.noIds];
    if (!all.length) return NaN;
    const mx = Math.max(...all.map((i) => L[i]));
    const Z = all.reduce((a, i) => a + Math.exp(L[i] - mx), 0);
    return this.yesIds.reduce((a, i) => a + Math.exp(L[i] - mx), 0) / Z;
  }

  /**
   * Loads the bilinear heads used by `ask(..., {method:"head"})`. Each source is an ArrayBuffer
   * (the .bin contents) or a URL. `noul` and `choice` come from the same original head.bin (if both
   * are given, `noul` wins); `score` is head_score.bin (required for score with method head; no fallback).
   */
  async loadHeads(src: { noul?: ArrayBuffer | string; choice?: ArrayBuffer | string; score?: ArrayBuffer | string }): Promise<void> {
    const get = async (x: ArrayBuffer | string) => {
      if (typeof x !== "string") return loadHead(x);
      if (x.startsWith("file:")) return loadHead(await readLocalBytes(x));
      const r = await fetch(x);
      if (!r.ok) throw new Error(`loadHeads: ${x} -> HTTP ${r.status}`);
      return loadHead(await r.arrayBuffer());
    };
    const nc = src.noul ?? src.choice;
    if (nc) this.heads.noulChoice = await get(nc);
    if (src.score) this.heads.score = await get(src.score);
  }

  /** Typed System-1 (noul / choice / score). Default read per mode: `ct0.defaultMethod` (ONNX runtimes: noul logits,
   *  choice head, score head_score.bin; webllm too, through recovery, else logits). `opts.method` overrides it. */
  ask(q: Probe & { levels?: number }, opts: AskOptions = {}): Promise<AskResult> {
    return askImpl(this.backend, this.tok, this.heads, q, this.askOpts(q, opts));
  }

  /** Several questions about one `state` (each question may omit `state`): same results as `ask()` for each, in order.
   *  genai and decode-only prefill the shared prefix (system turn + state) once and run only each question's suffix;
   *  transformers.js and webllm make independent `ask()` calls. */
  askMany(state: string, qs: AskManyQuestion[], opts: AskOptions = {}): Promise<AskResult[]> {
    return askManyWith(this.backend, this.tok, this.heads, state, qs, (q) => this.askOpts(q, opts));
  }

  private askOpts(q: AskQuestion, opts: AskOptions): AskOptions {
    return { ...opts, method: opts.method ?? this.methods[q.kind], maxSafeTokens: opts.maxSafeTokens ?? this.maxSafeTokens };
  }

  /**
   * Full prefill returning the hidden states the bilinear HEAD needs.
   * Requires a graph with a hidden-state output (`backend.hiddenName`). Positions come from
   * `positions.ts`: q at the '>' of `<decide>`, k at the last token and, for
   * choice, at the token covering the end of each `</opt>`.
   */
  async prefillHiddens(prompt: string): Promise<{ atDecide: Float32Array; atLast: Float32Array; atOpts: Float32Array[] } | null> {
    if (!this.backend.hiddenName) return null;
    const t = tokenOffsets(this.tok, prompt);
    const dp = decidePos(t.tokens), lp = t.ids.length - 1, op = optPositionsFor(prompt, t.offsets);
    const { hiddens } = await this.backend.prefill(t.ids, { positions: [dp, lp, ...op] });
    return { atDecide: hiddens[0], atLast: hiddens[1], atOpts: hiddens.slice(2) };
  }
}

/** Specifiers imported only at run time (never resolved by browser bundlers). */
const NODE_ORT = "onnxruntime-node";
const NODE_URL = "node:url";
const WEBLLM_PKG = "@mlc-ai/web-llm";
