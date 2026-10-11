export class PerChannelInt8 {
  static quantize(weights: Float32Array, rows: number, cols: number): { qW: Int8Array; scales: Float32Array } {
    const qW = new Int8Array(rows * cols);
    const scales = new Float32Array(rows);

    for (let r = 0; r < rows; r++) {
      let maxAbs = 0.0;
      for (let c = 0; c < cols; c++) {
        const abs = Math.abs(weights[r * cols + c]);
        if (abs > maxAbs) maxAbs = abs;
      }
      const scale = maxAbs > 0 ? maxAbs / 127.0 : 1.0;
      scales[r] = scale;
      const invScale = 1.0 / scale;

      for (let c = 0; c < cols; c++) {
        qW[r * cols + c] = Math.max(-128, Math.min(127, Math.round(weights[r * cols + c] * invScale)));
      }
    }

    return { qW, scales };
  }
}
