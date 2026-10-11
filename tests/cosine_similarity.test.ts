import { describe, it, expect } from 'vitest';
import { CosineSimilarity } from '../src/metrics/CosineSimilarity';

describe('CosineSimilarity', () => {
  it('evaluates parallel vectors to 1.0 and orthogonal vectors to 0.0', () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([5, 0, 0]);
    const c = new Float32Array([0, 1, 0]);

    expect(CosineSimilarity.evaluate(a, b)).toBeCloseTo(1.0);
    expect(CosineSimilarity.evaluate(a, c)).toBeCloseTo(0.0);
  });
});
