export class QuantDistortion {
  /**
   * Computes Mean Squared Error (MSE) between original FP32 tensor and quantized reconstruction.
   */
  static mse(original: ArrayLike<number>, reconstructed: ArrayLike<number>): number {
    const len = original.length;
    let sum = 0.0;
    for (let i = 0; i < len; i++) {
      const diff = original[i] - reconstructed[i];
      sum += diff * diff;
    }
    return sum / len;
  }

  /**
   * Signal-to-Quantization-Noise Ratio (SQNR) in decibels (dB).
   */
  static sqnr(original: ArrayLike<number>, reconstructed: ArrayLike<number>): number {
    let signalPower = 0.0;
    let noisePower = 0.0;
    const len = original.length;

    for (let i = 0; i < len; i++) {
      const s = original[i];
      const n = s - reconstructed[i];
      signalPower += s * s;
      noisePower += n * n;
    }

    if (noisePower === 0) return Infinity;
    if (signalPower === 0) return 0;

    return 10.0 * Math.log10(signalPower / noisePower);
  }

  /**
   * Peak Signal-to-Noise Ratio (PSNR) in decibels (dB) given dynamic range maxVal.
   */
  static psnr(original: ArrayLike<number>, reconstructed: ArrayLike<number>, maxVal?: number): number {
    const errorMse = this.mse(original, reconstructed);
    if (errorMse === 0) return Infinity;

    let peak = maxVal;
    if (peak === undefined) {
      let maxAbs = 0.0;
      for (let i = 0; i < original.length; i++) {
        const absVal = Math.abs(original[i]);
        if (absVal > maxAbs) maxAbs = absVal;
      }
      peak = maxAbs > 0 ? maxAbs : 1.0;
    }

    return 20.0 * Math.log10(peak) - 10.0 * Math.log10(errorMse);
  }

  /**
   * Mean Absolute Error (MAE) L1 distortion.
   */
  static mae(original: ArrayLike<number>, reconstructed: ArrayLike<number>): number {
    const len = original.length;
    let sum = 0.0;
    for (let i = 0; i < len; i++) {
      sum += Math.abs(original[i] - reconstructed[i]);
    }
    return len > 0 ? sum / len : 0.0;
  }
}
