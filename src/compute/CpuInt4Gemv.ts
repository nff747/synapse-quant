export class CpuInt4Gemv {
  /**
   * Optimized GEMV (1 x K) * (K x N) -> 1 x N for autoregressive single-token decoding.
   */
  static vecMat(
    x: Float32Array,
    B_packed: Uint32Array,
    N: number,
    K: number,
    scale: number
  ): Float32Array {
    const y = new Float32Array(N);

    for (let c = 0; c < N; c++) {
      let sum = 0.0;
      for (let k = 0; k < K; k++) {
        const u32Idx = Math.floor(k / 8) * N + c;
        const shift = (k % 8) * 4;
        const raw = (B_packed[u32Idx] >> shift) & 0x0F;
        const signedVal = raw >= 8 ? raw - 16 : raw;
        sum += x[k] * (signedVal * scale);
      }
      y[c] = sum;
    }

    return y;
  }
}
