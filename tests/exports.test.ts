import { describe, it, expect } from 'vitest';
import * as Synapse from '../src/index';

describe('Root Package Exports', () => {
  it('exposes all core classes and algorithms', () => {
    expect(Synapse.Int4Packer).toBeDefined();
    expect(Synapse.Int2Packer).toBeDefined();
    expect(Synapse.Fp8Format).toBeDefined();
    expect(Synapse.BlockQuantizer).toBeDefined();
    expect(Synapse.Nf4Packer).toBeDefined();
    expect(Synapse.CpuInt4Gemm).toBeDefined();
    expect(Synapse.CpuInt4Gemv).toBeDefined();
    expect(Synapse.QuantDistortion).toBeDefined();
  });
});
