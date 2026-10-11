import { Int4Packer } from '../quant/Int4Packer';

export class CpuInt4Gemm {
  static matmul(
    A: Float32Array,
    B_packed: Uint32Array,
    M: number,
    N: number,
    K: number,
    scale: number
  ): Float32Array {
    const C = new Float32Array(M * N);

    for (let r = 0; r < M; r++) {
      for (let c = 0; c < N; c++) {
        let sum = 0.0;
        for (let k = 0; k < K; k++) {
          const u32Idx = Math.floor(k / 8) * N + c;
          const shift = (k % 8) * 4;
          const raw = (B_packed[u32Idx] >> shift) & 0x0F;
          const signedVal = raw >= 8 ? raw - 16 : raw;
          const bVal = signedVal * scale;
          sum += A[r * K + k] * bVal;
        }
        C[r * N + c] = sum;
      }
    }

    return C;
  }
}
