import { describe, it, expect } from 'vitest';
import { Int2Packer } from '../src/quant/Int2Packer';

describe('Int2Packer', () => {
  it('quantizes and restores 16 weights in a single u32', () => {
    const data = new Float32Array(16);
    for (let i = 0; i < 16; i++) {
      data[i] = (i / 15.0) * 3.0; // [0, 3]
    }
    const { packed, scale, zeroPoint } = Int2Packer.quantizeAndPack(data);
    expect(packed.length).toBe(1);
    expect(scale).toBeCloseTo(1.0, 2);
    expect(zeroPoint).toBeCloseTo(0.0, 2);

    const unpacked = Int2Packer.unpack(packed, data.length, scale, zeroPoint);
    for (let i = 0; i < data.length; i++) {
      expect(Math.abs(unpacked[i] - data[i])).toBeLessThanOrEqual(0.6);
    }
  });
});
