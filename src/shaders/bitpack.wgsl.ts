/**
 * Synapse Quant - Dynamic On-GPU Activation Quantization & Bit-Packing Shader
 * Quantizes continuous activations to ternary values {-1, 0, 1} with dynamic thresholding.
 */
export const bitpackShader = /* wgsl */ `
struct PackParams {
  totalElements: u32,
  threshold: f32,
};

@group(0) @binding(0) var<uniform> params: PackParams;
@group(0) @binding(1) var<storage, read> inputFloats: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputPacked: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let wordIndex = global_id.x;
  let baseElement = wordIndex * 16u;

  if (baseElement >= params.totalElements) {
    return;
  }

  var packedWord: u32 = 0u;

  for (var i = 0u; i < 16u; i = i + 1u) {
    let elemIdx = baseElement + i;
    if (elemIdx < params.totalElements) {
      let val = inputFloats[elemIdx];
      var code: u32 = 0u; // 0b00 (zero)

      if (val > params.threshold) {
        code = 1u; // 0b01 (+1)
      } else if (val < -params.threshold) {
        code = 3u; // 0b11 (-1)
      }

      packedWord |= (code << (i * 2u));
    }
  }

  outputPacked[wordIndex] = packedWord;
}
`;
