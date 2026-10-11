import { Int4Packer } from '../quant/Int4Packer';
import { Int2Packer } from '../quant/Int2Packer';
import { BitPacker } from '../utils/BitPacker';
import { QuantDistortion } from '../metrics/QuantDistortion';
import { CosineSimilarity } from '../metrics/CosineSimilarity';

export type QuantPrecision = 'ternary' | 'int2' | 'int4';

export interface QuantPipelineResult {
  precision: QuantPrecision;
  compressionRatio: number;
  mse: number;
  cosineSim: number;
}

export class UnifiedQuantPipeline {
  static run(weights: Float32Array, precision: QuantPrecision): QuantPipelineResult {
    let reconstructed: Float32Array;
    let ratio: number;

    if (precision === 'ternary') {
      const { packed, scales } = BitPacker.packTernary(weights);
      reconstructed = BitPacker.unpackTernary(packed, weights.length, scales[0]);
      ratio = 16.0;
    } else if (precision === 'int2') {
      const { packed, scale, zeroPoint } = Int2Packer.quantizeAndPack(weights);
      reconstructed = Int2Packer.unpack(packed, weights.length, scale, zeroPoint);
      ratio = 16.0;
    } else {
      const { packed, scale } = Int4Packer.quantizeAndPack(weights);
      reconstructed = Int4Packer.unpack(packed, weights.length, scale);
      ratio = 8.0;
    }

    const mse = QuantDistortion.mse(weights, reconstructed);
    const cosineSim = CosineSimilarity.evaluate(weights, reconstructed);

    return { precision, compressionRatio: ratio, mse, cosineSim };
  }
}
