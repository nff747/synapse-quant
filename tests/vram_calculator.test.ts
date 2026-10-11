import { describe, it, expect } from 'vitest';
import { VramCalculator } from '../src/telemetry/VramCalculator';

describe('VramCalculator', () => {
  it('computes realistic compression ratio for 7B INT4 model', () => {
    const params = 7_000_000_000;
    const fp32Bytes = params * 4; // 28 GB
    const int4Bytes = VramCalculator.estimateBytes(params, 4, 64);
    expect(int4Bytes).toBeLessThan(4.0 * 1024 * 1024 * 1024); // < 4 GB
    expect(VramCalculator.compressionRatio(params, 4, 64)).toBeGreaterThan(7.0);
  });
});
