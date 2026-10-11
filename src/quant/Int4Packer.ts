/**
 * INT4 Symmetric Quantizer & Bit-Packer.
 * Packs 8 signed 4-bit integers [-8, 7] into a single 32-bit unsigned integer (u32).
 */
export class Int4Packer {
  static readonly WEIGHTS_PER_U32 = 8;
  static readonly BITS_PER_WEIGHT = 4;
  static readonly QMIN = -8;
  static readonly QMAX = 7;

  static quantizeAndPack(weights: ArrayLike<number>): { packed: Uint32Array; scale: number } {
    const len = weights.length;
    let maxAbs = 0;
    for (let i = 0; i < len; i++) {
      const abs = Math.abs(weights[i]);
      if (abs > maxAbs) maxAbs = abs;
    }
    const scale = maxAbs > 0 ? maxAbs / 7.0 : 1.0;
    const invScale = 1.0 / scale;

    const u32Count = Math.ceil(len / this.WEIGHTS_PER_U32);
    const packed = new Uint32Array(u32Count);

    for (let i = 0; i < len; i++) {
      const u32Idx = Math.floor(i / this.WEIGHTS_PER_U32);
      const shift = (i % this.WEIGHTS_PER_U32) * this.BITS_PER_WEIGHT;

      const q = Math.max(this.QMIN, Math.min(this.QMAX, Math.round(weights[i] * invScale)));
      const raw4bit = (q < 0 ? (q + 16) : q) & 0x0F;
      packed[u32Idx] |= (raw4bit << shift);
    }

    return { packed, scale };
  }

  static unpack(packed: Uint32Array, length: number, scale: number): Float32Array {
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      const u32Idx = Math.floor(i / this.WEIGHTS_PER_U32);
      const shift = (i % this.WEIGHTS_PER_U32) * this.BITS_PER_WEIGHT;
      const raw = (packed[u32Idx] >> shift) & 0x0F;
      const signedVal = raw >= 8 ? raw - 16 : raw;
      out[i] = signedVal * scale;
    }
    return out;
  }
}
