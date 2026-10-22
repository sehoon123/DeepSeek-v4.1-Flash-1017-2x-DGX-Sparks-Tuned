// tf_ds_topk_i64_kernel (zig/patches/0034's "before") against the register and slices kernels it adds: the top-k sets
// equal on random rows (ties in the score bits included), and the time a launch. nvcc -O3 -std=c++17
// -gencode=arch=compute_121,code=sm_121 topk-bench.cu -o topk-bench && ./topk-bench
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
extern "C" __global__ void __launch_bounds__(1024) tf_ds_topk_i64_kernel(const long long* keys, long long ks, int n, int k,
                                                                           long long* top) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long prefix_s, mask_s;
    __shared__ int want_s, count_s;
    const long long* row = keys + (long long)blockIdx.x * ks;
    long long* out = top + (long long)blockIdx.x * k;
    if (threadIdx.x == 0) {
        prefix_s = 0;
        mask_s = 0;
        want_s = k;
        count_s = 0;
    }
    __syncthreads();
    for (int d = 7; d >= 0; --d) {
        for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        const unsigned long long prefix = prefix_s, mask = mask_s;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const unsigned long long u = (unsigned long long)row[i] ^ 0x8000000000000000ull;
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> (8 * d)) & 0xff], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int want = want_s;
            int b = 255;
            for (; b > 0; --b) {
                if ((int)hist[b] >= want) break;
                want -= (int)hist[b];
            }
            want_s = want;
            prefix_s = prefix | ((unsigned long long)b << (8 * d));
            mask_s = mask | (0xffull << (8 * d));
        }
        __syncthreads();
    }
    const unsigned long long kth = prefix_s;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const unsigned long long u = (unsigned long long)row[i] ^ 0x8000000000000000ull;
        if (u > kth) out[atomicAdd(&count_s, 1)] = row[i];   // fewer than k
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const unsigned long long u = (unsigned long long)row[i] ^ 0x8000000000000000ull;
        if (u == kth) {
            const int at = atomicAdd(&count_s, 1);
            if (at < k) out[at] = row[i];
        }
    }
}


// The same top-k set as tf_ds_topk_i64_kernel (the keys are unique, so the k-th largest key, and with it the set, is one
// whatever the order of the work): a row a block, the row's keys read once into registers (at most RK a thread), each
// 8-bit digit's histogram counted by warps into shared memory and its bins summed by one warp's prefix scan instead of
// one thread's loop; then every key above the k-th is written and copies of the k-th fill the rest.
template <int RK>
__device__ __forceinline__ void topk_regs(const long long* row, int n, int k, long long* out) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long prefix_s, mask_s;
    __shared__ int want_s, count_s;
    unsigned long long u[RK];
#pragma unroll
    for (int j = 0; j < RK; ++j) {
        const int i = threadIdx.x + j * blockDim.x;
        u[j] = i < n ? ((unsigned long long)row[i] ^ 0x8000000000000000ull) : 0ull;   // (0: below every key, never counted)
    }
    if (threadIdx.x == 0) { prefix_s = 0; mask_s = 0; want_s = k; count_s = 0; }
    for (int d = 7; d >= 0; --d) {
        for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        const unsigned long long prefix = prefix_s, mask = mask_s;
#pragma unroll
        for (int j = 0; j < RK; ++j) {
            const int i = threadIdx.x + j * blockDim.x;
            if (i < n && (u[j] & mask) == prefix) atomicAdd(&hist[(u[j] >> (8 * d)) & 0xff], 1u);
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            // bins from the top: lane l holds bins 255-8l .. 248-8l; the first bin where the count from the top reaches want
            const int lane = threadIdx.x;
            unsigned int c[8], s = 0;
#pragma unroll
            for (int q = 0; q < 8; ++q) { c[q] = hist[255 - 8 * lane - q]; s += c[q]; }
            unsigned int incl = s;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) { const unsigned int v = __shfl_up_sync(0xffffffffu, incl, off); if (lane >= off) incl += v; }
            const unsigned int excl = incl - s;
            const int want = want_s;
            const unsigned int hit = __ballot_sync(0xffffffffu, (int)incl >= want);
            const int first = __ffs(hit) - 1;   // the lane whose bins hold the k-th key (one exists: want <= the keys left)
            if (lane == first) {
                unsigned int acc = excl;
                int b = 255 - 8 * lane;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    if ((int)(acc + c[q]) >= want) { b = 255 - 8 * lane - q; break; }
                    acc += c[q];
                }
                want_s = want - (int)acc;
                prefix_s = prefix | ((unsigned long long)b << (8 * d));
                mask_s = mask | (0xffull << (8 * d));
            }
        }
        __syncthreads();
    }
    const unsigned long long kth = prefix_s;
#pragma unroll
    for (int j = 0; j < RK; ++j) {
        const int i = threadIdx.x + j * blockDim.x;
        if (i < n && u[j] > kth) out[atomicAdd(&count_s, 1)] = (long long)(u[j] ^ 0x8000000000000000ull);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < RK; ++j) {
        const int i = threadIdx.x + j * blockDim.x;
        if (i < n && u[j] == kth) {
            const int at = atomicAdd(&count_s, 1);
            if (at < k) out[at] = (long long)(u[j] ^ 0x8000000000000000ull);
        }
    }
}

extern "C" __global__ void __launch_bounds__(1024) tf_ds_topk_i64_regs_kernel(const long long* keys, long long ks, int n, int k,
                                                                                long long* top) {
    const long long* row = keys + (long long)blockIdx.x * ks;
    long long* out = top + (long long)blockIdx.x * k;
    if (n <= 1024) topk_regs<1>(row, n, k, out);
    else if (n <= 4096) topk_regs<4>(row, n, k, out);
    else if (n <= 16384) topk_regs<16>(row, n, k, out);
    else topk_regs<32>(row, n, k, out);   // n <= 32768 (the caller checks)
}


// n past 4096 with few rows: the row in slices of 4096 keys, each slice's top-k by its own block (the row's top-k set is
// within the slices' top-k sets: the keys are unique), then the top-k of those (slices * k keys) by a block a row.
extern "C" __global__ void __launch_bounds__(1024) tf_ds_topk_i64_slices_kernel(const long long* keys, long long ks, int n, int k,
                                                                                  int slice, long long* cand) {
    const int nslices = (n + slice - 1) / slice;
    const int r = blockIdx.x / nslices, sl = blockIdx.x % nslices;
    const int lo = sl * slice, m = min(slice, n - lo);
    long long* out = cand + ((long long)r * nslices + sl) * k;
    if (m <= k) {   // the whole slice is among its top-k
        for (int i = threadIdx.x; i < k; i += blockDim.x) out[i] = i < m ? keys[(long long)r * ks + lo + i] : (long long)0x8000000000000000ull;
        return;
    }
    topk_regs<4>(keys + (long long)r * ks + lo, m, k, out);
}

static void run_old(const long long* d, int rows, int n, int k, long long* o) { tf_ds_topk_i64_kernel<<<rows, 1024>>>(d, n, n, k, o); }
static long long* scratch = nullptr;
static void run_new(const long long* d, int rows, int n, int k, long long* o) {
    if (n > 4096 && rows * ((n + 4095) / 4096) <= 192) {
        const int ns = (n + 4095) / 4096;
        if (!scratch) cudaMalloc(&scratch, (size_t)48 * 16 * 512 * 8);
        tf_ds_topk_i64_slices_kernel<<<rows * ns, 1024>>>(d, n, n, k, 4096, scratch);
        tf_ds_topk_i64_regs_kernel<<<rows, 1024>>>(scratch, (long long)ns * k, ns * k, k, o);
    } else tf_ds_topk_i64_regs_kernel<<<rows, 1024>>>(d, n, n, k, o);
}
int main() {
    int bad = 0, cases = 0;
    std::mt19937_64 g(7);
    for (int n : {1, 2, 7, 64, 511, 512, 513, 610, 1024, 1025, 1094, 4096, 4097, 8191, 16384, 16385, 32768}) for (int ties : {0, 1, 2}) for (int k : {1, 64, 512}) {
        if (k > n) continue;
        const int rows = 6;
        std::vector<long long> h((size_t)rows * n);
        for (int r = 0; r < rows; ++r) for (int i = 0; i < n; ++i) {
            float s = std::normal_distribution<float>(0.f, 1.f)(g);
            if (ties == 1) s = std::round(s * 4.f) / 4.f;            // many equal scores (keys still unique: the index bits)
            if (ties == 2) s = (i % 3 == 0) ? 1.5f : -s * s;          // a third at one score, many negative
            unsigned int bits; memcpy(&bits, &s, 4);
            h[(size_t)r * n + i] = ((long long)(int)bits << 32) | (unsigned int)(0x7fffffff - i);
        }
        long long *d, *o1, *o2; cudaMalloc(&d, h.size() * 8); cudaMalloc(&o1, (size_t)rows * k * 8); cudaMalloc(&o2, (size_t)rows * k * 8);
        cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice);
        run_old(d, rows, n, k, o1); run_new(d, rows, n, k, o2); cudaDeviceSynchronize();
        std::vector<long long> a((size_t)rows * k), b((size_t)rows * k);
        cudaMemcpy(a.data(), o1, a.size() * 8, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), o2, b.size() * 8, cudaMemcpyDeviceToHost);
        for (int r = 0; r < rows; ++r) {
            std::sort(a.begin() + r * k, a.begin() + (r + 1) * k); std::sort(b.begin() + r * k, b.begin() + (r + 1) * k);
        }
        cases++; if (a != b) { bad++; printf("DIFF n %d k %d ties %d\n", n, k, ties); }
        cudaFree(d); cudaFree(o1); cudaFree(o2);
    }
    printf("set equality: %d of %d cases differ\n", bad, cases);
    for (int n : {610, 1094, 4096, 32768}) {
        const int rows = 6, k = 512;
        std::vector<long long> h((size_t)rows * n);
        for (int r = 0; r < rows; ++r) for (int i = 0; i < n; ++i) { float s = std::normal_distribution<float>(0.f, 1.f)(g); unsigned int bits; memcpy(&bits, &s, 4); h[(size_t)r * n + i] = ((long long)(int)bits << 32) | (unsigned int)(0x7fffffff - i); }
        long long *d, *o; cudaMalloc(&d, h.size() * 8); cudaMalloc(&o, (size_t)rows * k * 8); cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        float t[2];
        for (int v = 0; v < 2; ++v) {
            for (int w = 0; w < 20; ++w) (v ? run_new : run_old)(d, rows, n, k, o);
            cudaEventRecord(e0); for (int w = 0; w < 400; ++w) (v ? run_new : run_old)(d, rows, n, k, o); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); t[v] = 1000.f * ms / 400;
        }
        printf("n %6d: old %.1f us, new %.1f us\n", n, t[0], t[1]);
        cudaFree(d); cudaFree(o);
    }
    return 0;
}
