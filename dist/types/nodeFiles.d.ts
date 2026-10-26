/** Default cache directory. */
export declare function defaultCacheDir(): Promise<string>;
/** Local path for `url`: `file:` URLs and plain paths as they are; http(s) downloaded into `cacheDir` (once). */
export declare function localFile(url: string, cacheDir: string, onStatus?: (m: string) => void): Promise<string>;
/** Bytes of a `file:` URL or a local path (fetch() in Node does not read files). */
export declare function readLocalBytes(url: string): Promise<ArrayBuffer>;
