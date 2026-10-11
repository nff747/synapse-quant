import { describe, it, expect } from 'vitest';
import { QuantDistortion } from '../src/metrics/QuantDistortion';

describe('QuantDistortion', () => {
  it('returns zero MSE and Infinite SQNR on identical signals', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    expect(QuantDistortion.mse(a, a)).toBe(0);
    expect(QuantDistortion.sqnr(a, a)).toBe(Infinity);
  });

  it('correctly calculates known SQNR', () => {
    const a = new Float32Array([10, 10, 10, 10]); // power = 400
    const b = new Float32Array([9, 9, 9, 9]);    // noise diff = 1 each, noise power = 4
    // 400 / 4 = 100 -> 10 * log10(100) = 20 dB
    expect(QuantDistortion.sqnr(a, b)).toBeCloseTo(20.0, 4);
  });
});
