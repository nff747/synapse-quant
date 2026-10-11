import { describe, it, expect } from 'vitest';
import { ClipOptimizer } from '../src/optimizer/ClipOptimizer';

describe('ClipOptimizer', () => {
  it('returns valid threshold bound', () => {
    const data = new Float32Array([1, 2, 1, 2, 1, 100]); // single extreme outlier
    const clip = ClipOptimizer.findBestClip(data, 5);
    expect(clip).toBeGreaterThan(0);
    expect(clip).toBeLessThanOrEqual(100);
  });
});
