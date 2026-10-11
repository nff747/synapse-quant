export const INT2_GEMM_WGSL = /* wgsl */ `
struct Uniforms {
  M: u32,
  N: u32,
  K: u32,
  scale: f32,
  zeroPoint: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B_packed: array<u32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.y;
  let col = global_id.x;
  if (row >= uniforms.M || col >= uniforms.N) {
    return;
  }

  var sum: f32 = 0.0;
  for (var k = 0u; k < uniforms.K; k = k + 1u) {
    let u32Idx = (k / 16u) * uniforms.N + col;
    let shift = (k % 16u) * 2u;
    let q = (B_packed[u32Idx] >> shift) & 3u;
    let bVal = uniforms.zeroPoint + f32(q) * uniforms.scale;
    sum = sum + A[row * uniforms.K + k] * bVal;
  }

  C[row * uniforms.N + col] = sum;
}
`;
