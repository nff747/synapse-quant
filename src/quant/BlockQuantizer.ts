import { Int4Packer } from './Int4Packer';

export interface QuantizedBlockTensor {
  packed: Uint32Array;
  scales: Float32Array;
  groupSize: number;
  totalLength: number;
}

/**
 * Block-wise / Group-wise Quantizer (e.g. GroupSize 32, 64, 128).
 * Provides high dynamic range preservation for outlier activations.
 */
export class BlockQuantizer {
  static quantizeGroupedInt4(weights: Float32Array, groupSize = 64): QuantizedBlockTensor {
    const totalLength = weights.length;
    const numGroups = Math.ceil(totalLength / groupSize);
    const scales = new Float32Array(numGroups);

    // Calculate total u32s
    const u32PerGroup = Math.ceil(groupSize / Int4Packer.WEIGHTS_PER_U32);
    const packed = new Uint32Array(numGroups * u32PerGroup);

    for (let g = 0; g < numGroups; g++) {
      const start = g * groupSize;
      const end = Math.min(totalLength, start + groupSize);
      const slice = weights.subarray(start, end);

      const groupRes = Int4Packer.quantizeAndPack(slice);
      scales[g] = groupRes.scale;

      const groupOffset = g * u32PerGroup;
      packed.set(groupRes.packed, groupOffset);
    }

    return { packed, scales, groupSize, totalLength };
  }

  static dequantizeGroupedInt4(qTensor: QuantizedBlockTensor): Float32Array {
    const out = new Float32Array(qTensor.totalLength);
    const numGroups = Math.ceil(qTensor.totalLength / qTensor.groupSize);
    const u32PerGroup = Math.ceil(qTensor.groupSize / Int4Packer.WEIGHTS_PER_U32);

    for (let g = 0; g < numGroups; g++) {
      const start = g * qTensor.groupSize;
      const end = Math.min(qTensor.totalLength, start + qTensor.groupSize);
      const len = end - start;

      const groupOffset = g * u32PerGroup;
      const groupPacked = qTensor.packed.subarray(groupOffset, groupOffset + u32PerGroup);
      const groupScale = qTensor.scales[g];

      const unpackedSlice = Int4Packer.unpack(groupPacked, len, groupScale);
      out.set(unpackedSlice, start);
    }

    return out;
  }
}
