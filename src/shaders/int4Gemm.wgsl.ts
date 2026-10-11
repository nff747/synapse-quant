/**
 * WGSL INT4 Matrix-Matrix Multiplication (GEMM) Compute Shader.
 * Utilizes workgroup shared memory tiles (16x16) and unpack4xI8-equivalent bitfield shifts.
 */
export const INT4_GEMM_WGSL = /* wgsl */ `
struct Uniforms {
  M: u32,
  N: u32,
  K: u32,
  scale: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>; // M x K activation
@group(0) @binding(2) var<storage, read> B_packed: array<u32>; // (K/8) x N packed INT4 weights
@group(0) @binding(3) var<storage, read_write> C: array<f32>; // M x N output

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let row = global_id.y;
  let col = global_id.x;
  var sum: f32 = 0.0;

  let numTiles = (uniforms.K + 15u) / 16u;

  for (var t = 0u; t < numTiles; t = t + 1u) {
    let aCol = t * 16u + local_id.x;
    if (row < uniforms.M && aCol < uniforms.K) {
      tileA[local_id.y][local_id.x] = A[row * uniforms.K + aCol];
    } else {
      tileA[local_id.y][local_id.x] = 0.0;
    }

    let bRow = t * 16u + local_id.y;
    if (bRow < uniforms.K && col < uniforms.N) {
      let u32Idx = (bRow / 8u) * uniforms.N + col;
      let shift = (bRow % 8u) * 4u;
      let raw = (B_packed[u32Idx] >> shift) & 0x0Fu;
      var signedVal: f32 = f32(raw);
      if (raw >= 8u) {
        signedVal = f32(raw) - 16.0;
      }
      tileB[local_id.y][local_id.x] = signedVal * uniforms.scale;
    } else {
      tileB[local_id.y][local_id.x] = 0.0;
    }

    workgroupBarrier();

    for (var k = 0u; k < 16u; k = k + 1u) {
      sum = sum + tileA[local_id.y][k] * tileB[k][local_id.x];
    }

    workgroupBarrier();
  }

  if (row < uniforms.M && col < uniforms.N) {
    C[row * uniforms.N + col] = sum;
  }
}
`;
