import { QuantDistortion } from '../metrics/QuantDistortion';
import { Int4Packer } from '../quant/Int4Packer';

export class ClipOptimizer {
  static findBestClip(weights: Float32Array, steps = 10): number {
    let maxAbs = 0.0;
    for (let i = 0; i < weights.length; i++) {
      const a = Math.abs(weights[i]);
      if (a > maxAbs) maxAbs = a;
    }

    let bestClip = maxAbs;
    let minMse = Infinity;

    for (let s = 1; s <= steps; s++) {
      const ratio = 0.8 + (s / steps) * 0.2; // test 0.82 to 1.0
      const clip = maxAbs * ratio;

      const clipped = new Float32Array(weights.length);
      for (let i = 0; i < weights.length; i++) {
        clipped[i] = Math.max(-clip, Math.min(clip, weights[i]));
      }

      const { packed, scale } = Int4Packer.quantizeAndPack(clipped);
      const reconstructed = Int4Packer.unpack(packed, weights.length, scale);

      const mse = QuantDistortion.mse(weights, reconstructed);
      if (mse < minMse) {
        minMse = mse;
        bestClip = clip;
      }
    }

    return bestClip;
  }
}
