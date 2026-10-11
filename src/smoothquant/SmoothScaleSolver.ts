/**
 * SmoothQuant Migration Scale s_j = max(|X_j|)^alpha / max(|W_j|)^(1 - alpha)
 */
export class SmoothScaleSolver {
  static computeSmoothingScales(
    actMax: Float32Array,
    weightMax: Float32Array,
    alpha = 0.5
  ): Float32Array {
    const len = actMax.length;
    const scales = new Float32Array(len);

    for (let j = 0; j < len; j++) {
      const a = Math.max(1e-5, actMax[j]);
      const w = Math.max(1e-5, weightMax[j]);
      scales[j] = Math.pow(a, alpha) / Math.pow(w, 1.0 - alpha);
    }

    return scales;
  }
}
