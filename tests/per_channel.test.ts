import { describe, it, expect } from 'vitest';
import { PerChannelInt8 } from '../src/quant/PerChannelInt8';

describe('PerChannelInt8', () => {
  it('normalizes each row with independent maximum scales', () => {
    const W = new Float32Array([
      1.27, -1.27,
      12.7, -12.7
    ]);
    const { qW, scales } = PerChannelInt8.quantize(W, 2, 2);
    expect(scales[0]).toBeCloseTo(0.01);
    expect(scales[1]).toBeCloseTo(0.1);
    expect(qW[0]).toBe(127);
    expect(qW[2]).toBe(127);
  });
});
