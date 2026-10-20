// GB10 GPU read-bandwidth ceiling: grid-stride and blocked reads, three load flavours, and an MoE-like layer pattern.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
__device__ __forceinline__ uint4 ldm(const uint4* p, int mode) {
  uint4 v;
  if (mode == 0) return __ldg(p);
  if (mode == 1) asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  else asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
template <int U> __global__ void rd_stride(const uint4* __restrict__ a, size_t n, int mode, uint32_t* out) {
  uint32_t acc = 0; size_t s = (size_t)gridDim.x * blockDim.x, i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (U - 1) * s < n; i += U * s) { uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; u++) v[u] = ldm(a + i + u * s, mode);
#pragma unroll
    for (int u = 0; u < U; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w; }
  for (; i < n; i += s) { uint4 v = __ldg(a + i); acc ^= v.x ^ v.w; }
  if (acc == 0x9e3779b9u) out[0] = acc;
}
template <int U> __global__ void rd_block(const uint4* __restrict__ a, size_t n, int mode, uint32_t* out) {
  size_t per = (n + gridDim.x - 1) / gridDim.x, b = (size_t)blockIdx.x * per, e = b + per < n ? b + per : n; uint32_t acc = 0;
  size_t i = b + threadIdx.x;
  for (; i + (U - 1) * blockDim.x < e; i += (size_t)U * blockDim.x) { uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; u++) v[u] = ldm(a + i + u * blockDim.x, mode);
#pragma unroll
    for (int u = 0; u < U; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w; }
  for (; i < e; i += blockDim.x) { uint4 v = __ldg(a + i); acc ^= v.x ^ v.w; }
  if (acc == 0x9e3779b9u) out[0] = acc;
}
// one "layer": 6 routed-expert halves of 6.35 MiB and a shared-expert half of 10.55 MiB at scattered offsets
__global__ void rd_layer(const uint4* __restrict__ a, const size_t* off, const size_t* len, int nchunk, int mode, uint32_t* out) {
  int c = blockIdx.y; const uint4* p = a + off[c]; size_t n = len[c]; uint32_t acc = 0;
  size_t s = (size_t)gridDim.x * blockDim.x, i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + 3 * s < n; i += 4 * s) { uint4 v0 = ldm(p + i, mode), v1 = ldm(p + i + s, mode), v2 = ldm(p + i + 2 * s, mode), v3 = ldm(p + i + 3 * s, mode);
    acc ^= v0.x ^ v1.y ^ v2.z ^ v3.w; }
  for (; i < n; i += s) acc ^= __ldg(p + i).x;
  if (acc == 0x9e3779b9u) out[0] = acc;
}
int main() {
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); int sms = pr.multiProcessorCount;
  const size_t bytes = 768ull << 20, n = bytes / 16; uint4* a; uint32_t* out; CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&out, 64)); CK(cudaMemset(a, 1, bytes));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  struct R { double gbs; const char* k; int thr, bps, u, mode; }; std::vector<R> rs;
  auto timeit = [&](auto launch, double nbytes, int reps) { launch(); launch(); cudaDeviceSynchronize(); std::vector<float> t;
    for (int r = 0; r < reps; r++) { cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); t.push_back(ms); }
    std::sort(t.begin(), t.end()); return nbytes / (t[t.size() / 2] * 1e-3) / 1e9; };
  for (int thr : {256, 512}) for (int bps : {4, 8, 16, 32}) for (int mode = 0; mode < 3; mode++) {
    int g = sms * bps; if (thr * bps > 2048) continue;
    rs.push_back({timeit([&] { rd_stride<4><<<g, thr>>>(a, n, mode, out); }, bytes, 9), "stride-u4", thr, bps, 4, mode});
    rs.push_back({timeit([&] { rd_stride<8><<<g, thr>>>(a, n, mode, out); }, bytes, 9), "stride-u8", thr, bps, 8, mode});
    rs.push_back({timeit([&] { rd_block<4><<<g, thr>>>(a, n, mode, out); }, bytes, 9), "block-u4", thr, bps, 4, mode});
  }
  CK(cudaGetLastError());
  std::sort(rs.begin(), rs.end(), [](const R& x, const R& y) { return x.gbs > y.gbs; });
  printf("SMs %d, %d configs; best:\n", sms, (int)rs.size());
  for (int i = 0; i < 6; i++) printf("  %7.1f GB/s  %-9s thr %d blocks/SM %2d mode %d\n", rs[i].gbs, rs[i].k, rs[i].thr, rs[i].bps, rs[i].mode);
  for (auto& r : rs) if (r.thr == 256 && r.bps == 8 && r.mode == 0 && r.u == 4 && r.k[0] == 's') printf("  baseline __ldg stride-u4 256x8/SM: %.1f GB/s\n", r.gbs);
  // MoE-like: 40 layers x (6 x 6.35 MiB + 10.55 MiB) at scattered offsets, one launch per layer, back to back
  const int L = 40, C = 7; std::vector<size_t> off(L * C), len(L * C); size_t cur = 0, chunk[7] = {6658458, 6658458, 6658458, 6658458, 6658458, 6658458, 11062476};
  for (int l = 0; l < L; l++) for (int c = 0; c < C; c++) { size_t q = chunk[c] / 16; if (cur + q > n) cur = (cur * 7 + 4096) % (n / 2); off[l * C + c] = cur; len[l * C + c] = q; cur += q + 8192; }
  size_t *doff, *dlen; CK(cudaMalloc(&doff, off.size() * 8)); CK(cudaMalloc(&dlen, len.size() * 8)); cudaMemcpy(doff, off.data(), off.size() * 8, cudaMemcpyHostToDevice); cudaMemcpy(dlen, len.data(), len.size() * 8, cudaMemcpyHostToDevice);
  double layer_bytes = 0; for (int c = 0; c < C; c++) layer_bytes += chunk[c];
  for (int mode = 0; mode < 3; mode++) for (int bx : {8, 16, 32}) {
    double g = timeit([&] { for (int l = 0; l < L; l++) rd_layer<<<dim3(bx, C), 512>>>(a, doff + l * C, dlen + l * C, C, mode, out); }, layer_bytes * L, 7);
    printf("  MoE-like 40 layers (%.1f MiB each), mode %d, %2dx7 CTAs of 512: %.1f GB/s (%.1f us a layer)\n", layer_bytes / 1048576, mode, bx, g, layer_bytes / g / 1e3);
  }
  CK(cudaGetLastError()); return 0;
}
