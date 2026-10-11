import { describe, it, expect } from 'vitest';
import { BlockQuantizer } from '../src/quant/BlockQuantizer';

describe('BlockQuantizer', () => {
  it('quantizes and dequantizes with group-size 64', () => {
    const size = 128;
    const data = new Float32Array(size);
    // Group 0 has small scale [0, 1]
    for (let i = 0; i < 64; i++) data[i] = Math.sin(i * 0.1);
    // Group 1 has large scale [0, 50]
    for (let i = 64; i < 128; i++) data[i] = 50.0 * Math.cos(i * 0.1);

    const q = BlockQuantizer.quantizeGroupedInt4(data, 64);
    expect(q.scales.length).toBe(2);
    expect(q.scales[0]).toBeLessThan(1.0);
    expect(q.scales[1]).toBeGreaterThan(5.0);

    const dequant = BlockQuantizer.dequantizeGroupedInt4(q);
    for (let i = 0; i < size; i++) {
      expect(Math.abs(dequant[i] - data[i])).toBeLessThanOrEqual(5.0);
    }
  });
});
