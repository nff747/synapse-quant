export class ScalarKMeans {
  static quantize(weights: Float32Array, k = 4, maxIters = 20): { centroids: Float32Array; indices: Uint8Array } {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < weights.length; i++) {
      if (weights[i] < min) min = weights[i];
      if (weights[i] > max) max = weights[i];
    }

    const centroids = new Float32Array(k);
    for (let i = 0; i < k; i++) {
      centroids[i] = min + (i / (k - 1)) * (max - min);
    }

    const indices = new Uint8Array(weights.length);

    for (let iter = 0; iter < maxIters; iter++) {
      // Assign
      for (let i = 0; i < weights.length; i++) {
        const w = weights[i];
        let bestIdx = 0;
        let bestDist = Math.abs(w - centroids[0]);
        for (let c = 1; c < k; c++) {
          const dist = Math.abs(w - centroids[c]);
          if (dist < bestDist) {
            bestDist = dist;
            bestIdx = c;
          }
        }
        indices[i] = bestIdx;
      }

      // Update centroids
      const counts = new Uint32Array(k);
      const sums = new Float32Array(k);
      for (let i = 0; i < weights.length; i++) {
        const idx = indices[i];
        sums[idx] += weights[i];
        counts[idx]++;
      }
      for (let c = 0; c < k; c++) {
        if (counts[c] > 0) {
          centroids[c] = sums[c] / counts[c];
        }
      }
    }

    return { centroids, indices };
  }
}
