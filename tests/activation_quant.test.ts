import { describe, it, expect } from 'vitest';
import { ActivationQuantizer } from '../src/smoothquant/ActivationQuantizer';

describe('ActivationQuantizer', () => {
  it('normalizes each row dynamically to [-127, 127]', () => {
    const X = new Float32Array([
      10.0, -10.0, 0.0,
      100.0, -50.0, 25.0
    ]);
    const { qX, scales } = ActivationQuantizer.quantizeRowWiseInt8(X, 2, 3);
    expect(scales[0]).toBeCloseTo(10.0 / 127.0, 4);
    expect(scales[1]).toBeCloseTo(100.0 / 127.0, 4);

    expect(qX[0]).toBe(127);
    expect(qX[1]).toBe(-127);
    expect(qX[3]).toBe(127);
  });
});
