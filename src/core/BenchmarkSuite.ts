export interface BenchmarkResult {
  matrixShape: { M: number; N: number; K: number };
  fp32VramBytes: number;
  ternaryVramBytes: number;
  compressionRatio: number;
  theoreticalGflops: number;
  estimatedSpeedup: number;
}

export class BenchmarkSuite {
  /**
   * Evaluates memory bandwidth and operational savings for BitNet 1.58-bit vs FP32.
   */
  static evaluateSpecs(M: number, N: number, K: number): BenchmarkResult {
    const weightCount = N * K;
    const fp32VramBytes = weightCount * 4; // 4 bytes per float32
    const ternaryVramBytes = Math.ceil(weightCount / 16) * 4 + N * 4; // 2 bits per weight + scale

    const compressionRatio = fp32VramBytes / ternaryVramBytes;
    const totalOps = 2 * M * N * K;
    const theoreticalGflops = totalOps / 1e9;

    // Memory-bound speedup factor based on reduced memory bandwidth consumption
    const estimatedSpeedup = Math.min(8.0, Number((compressionRatio * 0.45).toFixed(2)));

    return {
      matrixShape: { M, N, K },
      fp32VramBytes,
      ternaryVramBytes,
      compressionRatio: Number(compressionRatio.toFixed(2)),
      theoreticalGflops: Number(theoreticalGflops.toFixed(3)),
      estimatedSpeedup,
    };
  }
}
