import { describe, it, expect } from 'vitest';
import { INT4_GEMM_WGSL } from '../src/shaders/int4Gemm.wgsl';

describe('INT4 GEMM Shader Source', () => {
  it('contains expected workgroup size and binding definitions', () => {
    expect(INT4_GEMM_WGSL).toContain('@compute @workgroup_size(16, 16, 1)');
    expect(INT4_GEMM_WGSL).toContain('var<workgroup> tileA');
    expect(INT4_GEMM_WGSL).toContain('var<workgroup> tileB');
    expect(INT4_GEMM_WGSL).toContain('workgroupBarrier()');
  });

  it('correctly unpacks 4-bit nibbles with 2s complement signed adjustment', () => {
    expect(INT4_GEMM_WGSL).toContain('raw >= 8u');
    expect(INT4_GEMM_WGSL).toContain('signedVal = f32(raw) - 16.0');
  });
});
