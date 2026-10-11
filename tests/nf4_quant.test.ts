import { describe, it, expect } from 'vitest';
import { Nf4Packer } from '../src/quant/Nf4Packer';

describe('Nf4Packer', () => {
  it('maps 0.0 exactly to centroid index 7', () => {
    expect(Nf4Packer.quantize(0.0)).toBe(7);
    expect(Nf4Packer.dequantize(7)).toBe(0.0);
  });

  it('correctly maps extremes', () => {
    expect(Nf4Packer.quantize(-0.95)).toBe(0);
    expect(Nf4Packer.quantize(0.99)).toBe(15);
  });
});
