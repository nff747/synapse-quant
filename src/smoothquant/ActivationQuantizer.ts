export class ActivationQuantizer {
  /**
   * Dynamically quantizes float activations to 8-bit signed integers [-128, 127] per token row.
   */
  static quantizeRowWiseInt8(
    X: Float32Array,
    rows: number,
    cols: number
  ): { qX: Int8Array; scales: Float32Array } {
    const qX = new Int8Array(rows * cols);
    const scales = new Float32Array(rows);

    for (let r = 0; r < rows; r++) {
      let maxAbs = 0.0;
      for (let c = 0; c < cols; c++) {
        const abs = Math.abs(X[r * cols + c]);
        if (abs > maxAbs) maxAbs = abs;
      }
      const scale = maxAbs > 0 ? maxAbs / 127.0 : 1.0;
      scales[r] = scale;
      const invScale = 1.0 / scale;

      for (let c = 0; c < cols; c++) {
        const val = X[r * cols + c];
        qX[r * cols + c] = Math.max(-128, Math.min(127, Math.round(val * invScale)));
      }
    }

    return { qX, scales };
  }
}
