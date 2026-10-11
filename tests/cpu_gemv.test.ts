import { describe, it, expect } from 'vitest';
import { CpuInt4Gemv } from '../src/compute/CpuInt4Gemv';
import { CpuInt4Gemm } from '../src/compute/CpuInt4Gemm';

describe('CpuInt4Gemv', () => {
  it('yields identical numerical results to GEMM with M=1', () => {
    const K = 16;
    const N = 8;
    const x = new Float32Array(K);
    for (let i = 0; i < K; i++) x[i] = Math.sin(i);

    const B_packed = new Uint32Array((K / 8) * N);
    for (let i = 0; i < B_packed.length; i++) B_packed[i] = 0x11111111; // all 1s

    const gemvRes = CpuInt4Gemv.vecMat(x, B_packed, N, K, 0.5);
    const gemmRes = CpuInt4Gemm.matmul(x, B_packed, 1, N, K, 0.5);

    for (let i = 0; i < N; i++) {
      expect(gemvRes[i]).toBeCloseTo(gemmRes[i], 5);
    }
  });
});
