# Synapse-Quant Architecture & Kernel Specification

## Mixed Precision Matrix Tiling

Synapse-Quant provides ultra-high-throughput WebGPU / WGSL compute passes for edge LLM inference, reducing weights from FP32/FP16 down to INT4, INT2, and 1.58-bit Ternary formats.

### Workgroup Memory Layout
Each 16x16 compute tile loads activations $A$ and packed weights $B$ into workgroup shared memory:
- **INT4**: 8 weights packed per `u32`. Nibble extraction via bit shift `(B[idx] >> shift) & 0x0Fu`.
- **INT2**: 16 weights packed per `u32`.
- **Ternary**: 16 ternary values $\{-1, 0, 1\}$ packed per `u32` with 2-bit two's complement sign reconstruction.

### Quantization Pipeline Summary
1. **Activation-Aware Weight Quantization (AWQ)**: Column salience evaluation preserves top 1% critical weights.
2. **SmoothQuant**: Dynamic migration of activation peaks into weight matrices using optimal scale $\alpha = 0.5$.
3. **Group-wise Quantization**: Per-block scaling factor (group size 32, 64, or 128) suppresses dynamic range clipping errors.
