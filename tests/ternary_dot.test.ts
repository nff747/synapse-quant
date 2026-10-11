import { describe, it, expect } from 'vitest';
import { BitPacker } from '../src/utils/BitPacker';
import { TernaryDotProduct } from '../src/ternary/TernaryDotProduct';

describe('TernaryDotProduct', () => {
  it('accurately computes vector inner product', () => {
    const weights = [1, -1, 0, 1, 1, 0, -1, 0];
    const { packed } = BitPacker.packTernary(weights);
    const x = new Float32Array([10, 20, 30, 40, 50, 60, 70, 80]);

    // expected: 10*1 + 20*(-1) + 30*0 + 40*1 + 50*1 + 60*0 + 70*(-1) + 80*0 = 10 - 20 + 40 + 50 - 70 = 10
    const res = TernaryDotProduct.dot(x, packed, 1.0);
    expect(res).toBeCloseTo(10.0);
  });
});
