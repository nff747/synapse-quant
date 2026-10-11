import { describe, it, expect } from 'vitest';
import { Fp8Format } from '../src/quant/Fp8Format';

describe('Fp8Format (E4M3)', () => {
  it('correctly encodes and decodes representative float values', () => {
    const values = [0.0, 1.0, -1.0, 2.5, 16.0, 128.0, 400.0];
    for (const v of values) {
      const byte = Fp8Format.floatToE4M3(v);
      const recovered = Fp8Format.e4m3ToFloat(byte);
      if (v === 0) {
        expect(recovered).toBe(0);
      } else {
        const relErr = Math.abs((recovered - v) / v);
        expect(relErr).toBeLessThan(0.15); // ~12.5% max quantization step for 3-bit mantissa
      }
    }
  });

  it('handles NaN representation correctly', () => {
    const byte = Fp8Format.floatToE4M3(NaN);
    expect(byte).toBe(0x7F);
    expect(Number.isNaN(Fp8Format.e4m3ToFloat(0x7F))).toBe(true);
  });
});
