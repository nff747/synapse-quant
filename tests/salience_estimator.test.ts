import { describe, it, expect } from 'vitest';
import { SalienceEstimator } from '../src/awq/SalienceEstimator';

describe('SalienceEstimator', () => {
  it('correctly calculates channel-wise activation scales', () => {
    // 2 rows, 3 columns
    const acts = new Float32Array([
      2.0, -4.0, 10.0,
      4.0, -2.0, 20.0
    ]);
    const scales = SalienceEstimator.computeActivationScales(acts, 2, 3);
    expect(scales[0]).toBeCloseTo(3.0);
    expect(scales[1]).toBeCloseTo(3.0);
    expect(scales[2]).toBeCloseTo(15.0);
  });

  it('scales weight columns proportionally', () => {
    const weights = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const scales = new Float32Array([2.0, 0.5]);
    const scaled = SalienceEstimator.applySalienceScaling(weights, scales, 2, 2);
    expect(scaled[0]).toBeCloseTo(2.0); // 1.0 * 2.0
    expect(scaled[1]).toBeCloseTo(1.0); // 2.0 * 0.5
    expect(scaled[2]).toBeCloseTo(6.0); // 3.0 * 2.0
    expect(scaled[3]).toBeCloseTo(2.0); // 4.0 * 0.5
  });
});
