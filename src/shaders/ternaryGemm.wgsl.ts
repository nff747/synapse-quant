/**
 * Synapse Quant - Tiled 1.58-Bit (Ternary {-1, 0, 1}) Matrix Multiplication Kernel
 * Computes: C = A * B^T where B is packed ternary {-1, 0, 1} and A is FP32/FP16.
 * Zero floating-point multiplications: uses integer additions and subtractions.
 */
export const ternaryGemmShader = /* wgsl */ `
struct MatrixDimensions {
  M: u32, // Batch size / tokens
  N: u32, // Output features
  K: u32, // Input features (must be divisible by 16)
  K_packed: u32, // K / 16 (number of u32 words per row)
};

@group(0) @binding(0) var<uniform> dims: MatrixDimensions;
@group(0) @binding(1) var<storage, read> matrixA: array<f32>;        // [M, K]
@group(0) @binding(2) var<storage, read> packedWeightsB: array<u32>; // [N, K_packed]
@group(0) @binding(3) var<storage, read> weightScales: array<f32>;   // [N]
@group(0) @binding(4) var<storage, read_write> matrixC: array<f32>;  // [M, N]

// Tile dimensions for workgroup shared memory
const TILE_M: u32 = 16u;
const TILE_N: u32 = 16u;

var<workgroup> tileA: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>
) {
  let row = global_id.y; // index in M
  let col = global_id.x; // index in N

  var accumulator: f32 = 0.0;

  let numKWords = dims.K_packed;

  // Process 16 input features (1 u32 word) per step
  for (var w = 0u; w < numKWords; w = w + 1u) {
    let kBase = w * 16u;

    // Load packed 16 ternary weights for output channel 'col'
    var packedWord: u32 = 0u;
    if (col < dims.N) {
      let weightIdx = col * dims.K_packed + w;
      packedWord = packedWeightsB[weightIdx];
    }

    // Multiply-free dot product: branchless accumulation of 16 weights
    for (var bit = 0u; bit < 16u; bit = bit + 1u) {
      let kIndex = kBase + bit;
      let code = (packedWord >> (bit * 2u)) & 3u;

      if (row < dims.M && kIndex < dims.K) {
        let act = matrixA[row * dims.K + kIndex];
        
        // 0b01 -> +1 (add), 0b11 -> -1 (subtract), 0b00 -> 0 (skip)
        if (code == 1u) {
          accumulator += act;
        } else if (code == 3u) {
          accumulator -= act;
        }
      }
    }
  }

  // Write scaled result to matrix C
  if (row < dims.M && col < dims.N) {
    let scale = weightScales[col];
    matrixC[row * dims.N + col] = accumulator * scale;
  }
}
`;
