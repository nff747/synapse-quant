# ⚡ synapse-quant

> Hardware-Accelerated 1.58-Bit Ternary & INT2 Matrix GEMM Engine in WebGPU & WGSL.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![WebGPU](https://img.shields.io/badge/WebGPU-WGSL-red.svg)](https://www.w3.org/TR/webgpu/)

`synapse-quant` provides hardware-accelerated matrix-vector multiplication (GEMV/GEMM) compute shaders designed for 1.58-bit ternary ($\{-1, 0, 1\}$) and INT2 quantized Large Language Models (BitNet b1.58 architecture).

---

## 🚀 Key Features

- **Bit-Packed Kernels**: Packs four 2-bit weights into a single byte (`Uint8`), delivering 8x memory reduction over FP16.
- **Hardware WGSL Compute**: Direct WGSL compute pipeline with workgroup shared memory tiling and fused multiply-accumulate.
- **Cross-Platform**: Dual bindings in TypeScript and Python (`synapse_quant`).

---

## 🧪 Testing

```bash
# Python test suite
pytest tests/

# TypeScript / Vitest suite
npm test
```

---

## 📄 License

MIT © [nff747](https://github.com/nff747)
