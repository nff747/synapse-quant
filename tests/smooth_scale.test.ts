import { describe, it, expect } from 'vitest';
import { SmoothScaleSolver } from '../src/smoothquant/SmoothScaleSolver';

describe('SmoothScaleSolver', () => {
  it('balances scales equally when alpha is 0.5', () => {
    const actMax = new Float32Array([16.0, 4.0]);
    const weightMax = new Float32Array([4.0, 16.0]);
    const scales = SmoothScaleSolver.computeSmoothingScales(actMax, weightMax, 0.5);

    // sqrt(16) / sqrt(4) = 4 / 2 = 2.0
    expect(scales[0]).toBeCloseTo(2.0);
    // sqrt(4) / sqrt(16) = 2 / 4 = 0.5
    expect(scales[1]).toBeCloseTo(0.5);
  });
});
