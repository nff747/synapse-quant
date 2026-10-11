import { describe, it, expect } from 'vitest';
import { UnifiedQuantPipeline } from '../src/pipeline/UnifiedQuantPipeline';

describe('UnifiedQuantPipeline', () => {
  it('runs across all target precisions producing valid metric profiles', () => {
    const data = new Float32Array(64);
    for (let i = 0; i < 64; i++) data[i] = Math.sin(i * 0.2);

    for (const prec of ['ternary', 'int2', 'int4'] as const) {
      const res = UnifiedQuantPipeline.run(data, prec);
      expect(res.compressionRatio).toBeGreaterThanOrEqual(8.0);
      expect(res.cosineSim).toBeGreaterThan(0.7);
    }
  });
});
