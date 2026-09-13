import { describe, it, expect } from 'vitest';
import { BitPacker } from '../src/utils/BitPacker';
import { BenchmarkSuite } from '../src/core/BenchmarkSuite';
import { ternaryGemmShader } from '../src/shaders/ternaryGemm.wgsl';
import { bitpackShader } from '../src/shaders/bitpack.wgsl';

describe('Synapse Quant - BitPacker Mathematical Correctness', () => {
  it('correctly encodes and packs 16 ternary weights into a single u32', () => {
    // 16 weights: pattern of [+1, 0, -1, 0, +1, -1, ...]
    const rawWeights = [
      1, 0, -1, 0,
      1, -1, 0, 1,
      -1, -1, 0, 0,
      1, 1, -1, 0
    ];

    const { packed } = BitPacker.packTernary(rawWeights);
    expect(packed.length).toBe(1);

    // Unpack and verify exact recovery
    const unpacked = BitPacker.unpackTernary(packed, 16, 1.0);
    for (let i = 0; i < 16; i++) {
      expect(unpacked[i]).toBe(rawWeights[i]);
    }
  });

  it('handles arbitrary length arrays with zero-padding', () => {
    const rawWeights = [1, -1, 1, 0, -1];
    const { packed } = BitPacker.packTernary(rawWeights);
    expect(packed.length).toBe(1);

    const unpacked = BitPacker.unpackTernary(packed, 5, 1.0);
    expect(Array.from(unpacked)).toEqual([1, -1, 1, 0, -1]);
  });

  it('verifies 16x VRAM compression factor over FP32', () => {
    const weights1024 = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) {
      weights1024[i] = (i % 3) - 1; // -1, 0, 1
    }

    const { packed } = BitPacker.packTernary(weights1024);
    // 1024 weights / 16 = 64 u32 words = 256 bytes
    expect(packed.length).toBe(64);
    expect(packed.byteLength).toBe(256);

    const fp32Bytes = 1024 * 4; // 4096 bytes
    expect(fp32Bytes / packed.byteLength).toBe(16);
  });
});

describe('Synapse Quant - WGSL Shader Syntax Verification', () => {
  it('validates ternary GEMM compute shader structure', () => {
    expect(ternaryGemmShader).toContain('@compute @workgroup_size(16, 16)');
    expect(ternaryGemmShader).toContain('struct MatrixDimensions');
    expect(ternaryGemmShader).toContain('packedWeightsB');
    // Check branchless addition/subtraction accumulation
    expect(ternaryGemmShader).toContain('accumulator += act;');
    expect(ternaryGemmShader).toContain('accumulator -= act;');
  });

  it('validates dynamic activation bit-packing shader structure', () => {
    expect(bitpackShader).toContain('@compute @workgroup_size(64)');
    expect(bitpackShader).toContain('packedWord |= (code << (i * 2u));');
  });
});

describe('Synapse Quant - Benchmark Specifications', () => {
  it('evaluates typical 7B/13B parameter LLM projection layer specs', () => {
    // Hidden dimension: 4096 x 4096
    const specs = BenchmarkSuite.evaluateSpecs(1, 4096, 4096);

    expect(specs.compressionRatio).toBeGreaterThan(14.0);
    expect(specs.fp32VramBytes).toBe(4096 * 4096 * 4); // 67.1 MB
    expect(specs.ternaryVramBytes).toBeLessThan(5 * 1024 * 1024); // < 5 MB
    expect(specs.estimatedSpeedup).toBeGreaterThan(5.0);
  });
});
