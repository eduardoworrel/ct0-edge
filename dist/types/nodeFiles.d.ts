export declare function defaultCacheDir(): Promise<string>;
export declare function localFile(url: string, cacheDir: string, onStatus?: (m: string) => void): Promise<string>;
export declare function readLocalBytes(url: string): Promise<ArrayBuffer>;
