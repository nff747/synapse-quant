/**
 * NormalFloat4 (NF4) optimal quantile-based 4-bit quantizer.
 * Centroids derived from standard normal distribution N(0, 1).
 */
export class Nf4Packer {
  static readonly CENTROIDS = new Float32Array([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0
  ]);

  static quantize(val: number): number {
    let bestIdx = 0;
    let minDiff = Math.abs(val - this.CENTROIDS[0]);
    for (let i = 1; i < 16; i++) {
      const diff = Math.abs(val - this.CENTROIDS[i]);
      if (diff < minDiff) {
        minDiff = diff;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  static dequantize(idx: number): number {
    return this.CENTROIDS[idx & 0x0F];
  }
}
