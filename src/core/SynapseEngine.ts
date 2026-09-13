import { ternaryGemmShader } from '../shaders/ternaryGemm.wgsl';
import { bitpackShader } from '../shaders/bitpack.wgsl';
import { TernaryTensor } from './TernaryTensor';

export interface GemmOptions {
  M: number; // Tokens / Batch
  N: number; // Output features
  K: number; // Input features
}

export class SynapseEngine {
  readonly device: GPUDevice;
  private gemmPipeline!: GPUComputePipeline;
  private packPipeline!: GPUComputePipeline;

  constructor(device: GPUDevice) {
    this.device = device;
    this.initPipelines();
  }

  private initPipelines(): void {
    const gemmModule = this.device.createShaderModule({
      label: 'Synapse 1.58-Bit GEMM Module',
      code: ternaryGemmShader,
    });

    this.gemmPipeline = this.device.createComputePipeline({
      label: 'Synapse GEMM Pipeline',
      layout: 'auto',
      compute: {
        module: gemmModule,
        entryPoint: 'main',
      },
    });

    const packModule = this.device.createShaderModule({
      label: 'Synapse BitPack Module',
      code: bitpackShader,
    });

    this.packPipeline = this.device.createComputePipeline({
      label: 'Synapse BitPack Pipeline',
      layout: 'auto',
      compute: {
        module: packModule,
        entryPoint: 'main',
      },
    });
  }

  /**
   * Executes C = A * B^T
   * where A is [M, K] activations, B is [N, K] packed ternary weights.
   */
  matmul(
    commandEncoder: GPUCommandEncoder,
    activationBuffer: GPUBuffer,
    weightTensor: TernaryTensor,
    outputBuffer: GPUBuffer,
    opts: GemmOptions
  ): void {
    const { M, N, K } = opts;
    const K_packed = K / 16;

    // Uniform buffer with matrix dimensions
    const dimsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      dimsBuffer,
      0,
      new Uint32Array([M, N, K, K_packed])
    );

    const bindGroup = this.device.createBindGroup({
      layout: this.gemmPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dimsBuffer } },
        { binding: 1, resource: { buffer: activationBuffer } },
        { binding: 2, resource: { buffer: weightTensor.packedBuffer } },
        { binding: 3, resource: { buffer: weightTensor.scaleBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
      ],
    });

    const pass = commandEncoder.beginComputePass({ label: 'Synapse Ternary GEMM Pass' });
    pass.setPipeline(this.gemmPipeline);
    pass.setBindGroup(0, bindGroup);

    // 16x16 workgroup tiles
    const workgroupsX = Math.ceil(N / 16);
    const workgroupsY = Math.ceil(M / 16);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
    pass.end();
  }
}
