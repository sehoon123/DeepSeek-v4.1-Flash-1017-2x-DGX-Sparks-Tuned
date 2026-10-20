// GPU <-> CPU round trip through pinned host memory (the TP gather's doorbell and arrival-flag path, d1 + d5):
// one GPU thread stores flag = i (st.volatile + __threadfence_system) and spins until the host thread's ack == i; the
// host thread spins on flag and answers. Also a GPU-only spin on a device flag for the floor. Prints p50/p90/mean.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

__global__ void gpu_side(volatile uint32_t* flag, volatile uint32_t* ack, int n, unsigned long long* t) {
  for (int i = 1; i <= n; i++) {
    unsigned long long a; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(a));
    *flag = i; __threadfence_system();
    while (*ack != (uint32_t)i) { }
    unsigned long long b; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(b));
    t[i - 1] = b - a;
  }
}

int main(int argc, char** argv) {
  int n = argc > 1 ? atoi(argv[1]) : 20000;
  uint32_t *hflag, *hack; unsigned long long* dt;
  CK(cudaHostAlloc(&hflag, 4096, cudaHostAllocMapped)); CK(cudaHostAlloc(&hack, 4096, cudaHostAllocMapped));
  *hflag = 0; *hack = 0;
  uint32_t *dflag, *dack; CK(cudaHostGetDevicePointer(&dflag, hflag, 0)); CK(cudaHostGetDevicePointer(&dack, hack, 0));
  CK(cudaMalloc(&dt, n * sizeof(unsigned long long)));
  std::atomic<bool> stop{false};
  std::thread host([&] {
    volatile uint32_t* f = hflag; volatile uint32_t* a = hack; uint32_t last = 0;
    while (!stop.load(std::memory_order_relaxed)) { uint32_t v = *f; if (v != last) { last = v; *a = v; } }
  });
  for (int rep = 0; rep < 3; rep++) {
    *hflag = 0; *hack = 0;
    gpu_side<<<1, 1>>>(dflag, dack, n, dt); CK(cudaDeviceSynchronize());
    std::vector<unsigned long long> t(n); CK(cudaMemcpy(t.data(), dt, n * 8, cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> s(t.begin() + n / 10, t.end()); std::sort(s.begin(), s.end());
    double m = 0; for (auto x : s) m += x; m /= s.size();
    printf("rep %d: round trip p50 %.2f us  p90 %.2f us  p99 %.2f us  mean %.2f us (n=%zu)\n", rep, s[s.size() / 2] / 1e3,
           s[s.size() * 9 / 10] / 1e3, s[s.size() * 99 / 100] / 1e3, m / 1e3, s.size());
    // host must have seen the last flag before the next rep resets
  }
  stop = true; host.join();
  return 0;
}
