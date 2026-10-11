export class VramCalculator {
  static estimateBytes(paramCount: number, bitsPerParam: number, groupSize = 64): number {
    const weightBits = paramCount * bitsPerParam;
    const numScales = Math.ceil(paramCount / groupSize);
    const scaleBits = numScales * 32; // FP32 scales
    return Math.ceil((weightBits + scaleBits) / 8);
  }

  static compressionRatio(paramCount: number, bitsPerParam: number, groupSize = 64): number {
    const fp32Bytes = paramCount * 4;
    const quantBytes = this.estimateBytes(paramCount, bitsPerParam, groupSize);
    return fp32Bytes / quantBytes;
  }
}
