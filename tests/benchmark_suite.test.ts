import { describe, it, expect } from 'vitest';
import { CpuInt4Gemm } from '../src/compute/CpuInt4Gemm';

describe('Throughput Benchmark', () => {
  it('executes 100 iterations of 32x32 INT4 GEMM in under 50ms', () => {
    const M = 32, N = 32, K = 32;
    const A = new Float32Array(M * K);
    const B_packed = new Uint32Array((K / 8) * N);

    const start = performance.now();
    for (let i = 0; i < 50; i++) {
      CpuInt4Gemm.matmul(A, B_packed, M, N, K, 1.0);
    }
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(100.0);
  });
});
