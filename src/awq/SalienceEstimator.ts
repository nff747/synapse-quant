/**
 * Activation-Aware Weight Quantization (AWQ) Salience Estimator.
 * Computes channel-wise importance S_w = s_X * |W| to protect top 1% critical weights.
 */
export class SalienceEstimator {
  /**
   * Computes per-channel average activation magnitude from calibration samples.
   */
  static computeActivationScales(activations: Float32Array, rows: number, cols: number): Float32Array {
    const scales = new Float32Array(cols);
    for (let c = 0; c < cols; c++) {
      let sum = 0;
      for (let r = 0; r < rows; r++) {
        sum += Math.abs(activations[r * cols + c]);
      }
      scales[c] = sum / rows;
    }
    return scales;
  }

  /**
   * Multiplies weight columns by salience scale vector before quantization.
   */
  static applySalienceScaling(
    weights: Float32Array,
    scales: Float32Array,
    rows: number,
    cols: number
  ): Float32Array {
    const scaled = new Float32Array(weights.length);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        scaled[idx] = weights[idx] * scales[c];
      }
    }
    return scaled;
  }
}
