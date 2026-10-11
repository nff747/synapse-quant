import { describe, it, expect } from 'vitest';
import { ScalarKMeans } from '../src/cluster/ScalarKMeans';

describe('ScalarKMeans', () => {
  it('converges to cluster modes for bimodal data', () => {
    const data = new Float32Array([-10, -9.8, -10.2, 5.0, 5.1, 4.9]);
    const { centroids } = ScalarKMeans.quantize(data, 2);
    const sorted = Array.from(centroids).sort((a, b) => a - b);
    expect(sorted[0]).toBeCloseTo(-10.0, 0);
    expect(sorted[1]).toBeCloseTo(5.0, 0);
  });
});
