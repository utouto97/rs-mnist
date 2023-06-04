/* tslint:disable */
/* eslint-disable */
/**
* @param {Float32Array} w1
* @param {Float32Array} b1
* @param {Float32Array} w2
* @param {Float32Array} b2
* @param {Float32Array} w3
* @param {Float32Array} b3
* @param {Float32Array} inputs
* @returns {number}
*/
export function predict(w1: Float32Array, b1: Float32Array, w2: Float32Array, b2: Float32Array, w3: Float32Array, b3: Float32Array, inputs: Float32Array): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly predict: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => number;
  readonly __wbindgen_malloc: (a: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
