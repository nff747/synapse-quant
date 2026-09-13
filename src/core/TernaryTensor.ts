import { BitPacker } from '../utils/BitPacker';

export interface TensorShape {
  rows: number; // N (output features)
  cols: number; // K (input features, multiple of 16)
}

export class TernaryTensor {
  readonly device: GPUDevice;
  readonly shape: TensorShape;
  readonly packedBuffer: GPUBuffer;
  readonly scaleBuffer: GPUBuffer;
  readonly totalElements: number;
  readonly packedWordCount: number;

  constructor(device: GPUDevice, shape: TensorShape, weights?: ArrayLike<number>) {
    this.device = device;
    this.shape = shape;

    if (shape.cols % 16 !== 0) {
      throw new Error(`TernaryTensor cols must be divisible by 16 (got ${shape.cols})`);
    }

    this.totalElements = shape.rows * shape.cols;
    this.packedWordCount = Math.ceil(this.totalElements / BitPacker.WEIGHTS_PER_U32);

    // Create GPU Buffers
    this.packedBuffer = device.createBuffer({
      size: Math.max(16, this.packedWordCount * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.scaleBuffer = device.createBuffer({
      size: Math.max(16, shape.rows * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    if (weights) {
      this.upload(weights);
    }
  }

  upload(rawWeights: ArrayLike<number>): void {
    const { packed, scales } = BitPacker.packTernary(rawWeights);
    
    this.device.queue.writeBuffer(this.packedBuffer, 0, packed.buffer);
    
    // If single scale, broadcast to rows, else upload per-row scales
    if (scales.length === 1) {
      const fullScales = new Float32Array(this.shape.rows).fill(scales[0]);
      this.device.queue.writeBuffer(this.scaleBuffer, 0, fullScales.buffer);
    } else {
      this.device.queue.writeBuffer(this.scaleBuffer, 0, scales.buffer);
    }
  }

  destroy(): void {
    this.packedBuffer.destroy();
    this.scaleBuffer.destroy();
  }
}
