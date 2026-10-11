import { describe, it, expect } from 'vitest';
import { CpuInt4Gemm } from '../src/compute/CpuInt4Gemm';
import { Int4Packer } from '../src/quant/Int4Packer';

describe('CpuInt4Gemm', () => {
  it('accurately computes 2x2 matrix multiplication against packed weights', () => {
    // A: 2x2 [[1, 2], [3, 4]]
    const A = new Float32Array([1, 2, 3, 4]);
    // B: 2x2 [[1, 0], [0, 1]] identity
    // Packed layout: (K/8) * N = 1 * 2 = 2 u32 elements
    const B_packed = new Uint32Array(2);
    // col 0: k=0 -> 1, k=1 -> 0
    B_packed[0] = (1 & 0x0F) | ((0 & 0x0F) << 4);
    // col 1: k=0 -> 0, k=1 -> 1
    B_packed[1] = (0 & 0x0F) | ((1 & 0x0F) << 4);

    const C = CpuInt4Gemm.matmul(A, B_packed, 2, 2, 2, 1.0);
    expect(C[0]).toBeCloseTo(1.0);
    expect(C[1]).toBeCloseTo(2.0);
    expect(C[2]).toBeCloseTo(3.0);
    expect(C[3]).toBeCloseTo(4.0);
  });
});
