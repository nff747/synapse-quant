import { describe, it, expect } from 'vitest';
import { INT2_GEMM_WGSL } from '../src/shaders/int2Gemm.wgsl';

describe('INT2 GEMM Shader Source', () => {
  it('contains zeroPoint and scale uniform bindings', () => {
    expect(INT2_GEMM_WGSL).toContain('zeroPoint: f32');
    expect(INT2_GEMM_WGSL).toContain('scale: f32');
    expect(INT2_GEMM_WGSL).toContain('(B_packed[u32Idx] >> shift) & 3u');
  });
});
