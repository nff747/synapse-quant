/**
 * Synapse Quant - 1.58-Bit (Ternary {-1, 0, 1}) Packing Utility
 * Compresses 16 ternary weights into a single 32-bit unsigned integer (u32).
 * 
 * Encoding:
 *   0b00 ->  0
 *   0b01 -> +1
 *   0b11 -> -1 (2's complement lower 2 bits)
 *   0b10 -> unused / zero
 */
export class BitPacker {
  static readonly WEIGHTS_PER_U32 = 16;
  static readonly BITS_PER_WEIGHT = 2;

  /**
   * Packs an array of ternary values {-1, 0, 1} into a Uint32Array.
   */
  static packTernary(weights: ArrayLike<number>): { packed: Uint32Array; scales: Float32Array } {
    const len = weights.length;
    const u32Count = Math.ceil(len / this.WEIGHTS_PER_U32);
    const packed = new Uint32Array(u32Count);
    
    // Compute per-channel or per-tensor scale factor (average absolute magnitude)
    let sumAbs = 0;
    for (let i = 0; i < len; i++) {
      sumAbs += Math.abs(weights[i]);
    }
    const scale = sumAbs > 0 ? sumAbs / len : 1.0;
    const scales = new Float32Array([scale]);

    for (let i = 0; i < len; i++) {
      const u32Index = Math.floor(i / this.WEIGHTS_PER_U32);
      const bitOffset = (i % this.WEIGHTS_PER_U32) * this.BITS_PER_WEIGHT;
      
      const w = weights[i];
      let code = 0b00;
      if (w > 0.3) {
        code = 0b01; // +1
      } else if (w < -0.3) {
        code = 0b11; // -1
      }

      packed[u32Index] |= (code << bitOffset);
    }

    return { packed, scales };
  }

  /**
   * Unpacks a packed Uint32Array back into an array of ternary numbers {-1, 0, 1}.
   */
  static unpackTernary(packed: Uint32Array, length: number, scale = 1.0): Float32Array {
    const unpacked = new Float32Array(length);

    for (let i = 0; i < length; i++) {
      const u32Index = Math.floor(i / this.WEIGHTS_PER_U32);
      const bitOffset = (i % this.WEIGHTS_PER_U32) * this.BITS_PER_WEIGHT;
      
      const code = (packed[u32Index] >> bitOffset) & 0b11;
      let val = 0.0;
      if (code === 0b01) {
        val = 1.0;
      } else if (code === 0b11) {
        val = -1.0;
      }
      unpacked[i] = val * scale;
    }

    return unpacked;
  }
}
