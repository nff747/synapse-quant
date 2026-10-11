import { BitPacker } from '../utils/BitPacker';

export class TernaryDotProduct {
  static dot(x: Float32Array, packedWeights: Uint32Array, scale = 1.0): number {
    const len = x.length;
    let sum = 0.0;

    for (let i = 0; i < len; i++) {
      const u32Index = Math.floor(i / 16);
      const shift = (i % 16) * 2;
      const code = (packedWeights[u32Index] >> shift) & 0b11;

      let w = 0.0;
      if (code === 0b01) w = 1.0;
      else if (code === 0b11) w = -1.0;

      sum += x[i] * w;
    }

    return sum * scale;
  }
}
