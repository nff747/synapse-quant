import { describe, it, expect } from 'vitest';
import { Int4Packer } from '../src/quant/Int4Packer';

describe('Int4Packer', () => {
  it('correctly quantizes and unpacks 4-bit weights', () => {
    const original = new Float32Array([-7.0, -3.0, 0.0, 1.0, 4.0, 7.0, -1.0, 2.0]);
    const { packed, scale } = Int4Packer.quantizeAndPack(original);
    expect(packed.length).toBe(1);
    expect(scale).toBeCloseTo(1.0, 3);

    const unpacked = Int4Packer.unpack(packed, original.length, scale);
    for (let i = 0; i < original.length; i++) {
      expect(unpacked[i]).toBeCloseTo(original[i], 1);
    }
  });

  it('handles zero-padding when length is not multiple of 8', () => {
    const original = new Float32Array([1.5, -2.5, 3.0]);
    const { packed, scale } = Int4Packer.quantizeAndPack(original);
    expect(packed.length).toBe(1);
    const unpacked = Int4Packer.unpack(packed, original.length, scale);
    expect(unpacked.length).toBe(3);
    expect(unpacked[0]).toBeCloseTo(original[0], 0);
  });
});
