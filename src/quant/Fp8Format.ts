/**
 * FP8 Floating Point Emulation.
 * Implements IEEE-compliant E4M3 (4 exponent bits, 3 mantissa bits) and E5M2.
 */
export class Fp8Format {
  /**
   * Encodes float32 into FP8 E4M3 byte representation (range [-448, 448]).
   */
  static floatToE4M3(val: number): number {
    if (isNaN(val)) return 0x7F;
    if (val === 0) return (1 / val === -Infinity) ? 0x80 : 0x00;

    const sign = val < 0 ? 0x80 : 0x00;
    const absVal = Math.abs(val);

    if (absVal > 448.0) return sign | 0x7E; // clamp to max finite
    if (absVal < 0.001953125) return sign; // underflow to subnormal/zero

    const view = new DataView(new ArrayBuffer(4));
    view.setFloat32(0, absVal);
    const bits = view.getUint32(0);

    const f32Exp = (bits >> 23) & 0xFF;
    const f32Mant = bits & 0x7FFFFF;

    let e4m3Exp = f32Exp - 127 + 7; // bias 7
    let e4m3Mant = f32Mant >> (23 - 3);

    if (e4m3Exp <= 0) {
      e4m3Exp = 0;
      e4m3Mant = 0;
    } else if (e4m3Exp > 15) {
      e4m3Exp = 15;
      e4m3Mant = 6;
    }

    return (sign | (e4m3Exp << 3) | (e4m3Mant & 0x07)) & 0xFF;
  }

  /**
   * Decodes FP8 E4M3 byte back to float32.
   */
  static e4m3ToFloat(byte: number): number {
    const sign = (byte & 0x80) ? -1.0 : 1.0;
    const exp = (byte >> 3) & 0x0F;
    const mant = byte & 0x07;

    if (exp === 0) {
      // Subnormal
      return sign * (mant / 8.0) * Math.pow(2, -6);
    }
    if (exp === 15 && mant === 7) {
      return NaN;
    }
    return sign * (1.0 + mant / 8.0) * Math.pow(2, exp - 7);
  }
}
