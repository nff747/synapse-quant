/**
 * INT2 2-Bit Quantizer.
 * Packs 16 2-bit weights into a single 32-bit unsigned integer (u32).
 * Discrete levels: [-2, -1, 0, 1] mapped to [0b00, 0b01, 0b10, 0b11].
 */
export class Int2Packer {
  static readonly WEIGHTS_PER_U32 = 16;
  static readonly BITS_PER_WEIGHT = 2;

  static quantizeAndPack(weights: ArrayLike<number>): { packed: Uint32Array; scale: number; zeroPoint: number } {
    const len = weights.length;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < len; i++) {
      const v = weights[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = Math.max(1e-6, max - min);
    const scale = range / 3.0;
    const invScale = 1.0 / scale;

    const u32Count = Math.ceil(len / this.WEIGHTS_PER_U32);
    const packed = new Uint32Array(u32Count);

    for (let i = 0; i < len; i++) {
      const u32Idx = Math.floor(i / this.WEIGHTS_PER_U32);
      const shift = (i % this.WEIGHTS_PER_U32) * this.BITS_PER_WEIGHT;

      const q = Math.max(0, Math.min(3, Math.round((weights[i] - min) * invScale)));
      packed[u32Idx] |= (q << shift);
    }

    return { packed, scale, zeroPoint: min };
  }

  static unpack(packed: Uint32Array, length: number, scale: number, zeroPoint: number): Float32Array {
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      const u32Idx = Math.floor(i / this.WEIGHTS_PER_U32);
      const shift = (i % this.WEIGHTS_PER_U32) * this.BITS_PER_WEIGHT;
      const q = (packed[u32Idx] >> shift) & 0b11;
      out[i] = zeroPoint + q * scale;
    }
    return out;
  }
}
