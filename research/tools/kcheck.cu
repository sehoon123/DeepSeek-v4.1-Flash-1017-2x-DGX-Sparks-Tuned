// kcheck.cu: this recipe's own kernels in dsv41_ops.cu (0034 topk, 0035 sampling candidates) against CPU references,
// launched as ops.zig launches them, for a plain run and for compute-sanitizer (memcheck, racecheck, initcheck,
// synccheck). Build: nvcc -arch=sm_121 -O3 -lineinfo --fmad=false --ftz=false -DKSRC='"dsv41_ops.cu"' kcheck.cu
// Args: [quick] (fewer, smaller cases: for racecheck) [allnan] (adds a row of NaN only to the candidates cases).
#include KSRC
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                                  \
    do {                                                                                                       \
        cudaError_t e_ = (x);                                                                                  \
        if (e_ != cudaSuccess) {                                                                               \
            fprintf(stderr, "CUDA error %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_));          \
            exit(2);                                                                                           \
        }                                                                                                      \
    } while (0)

static int fails = 0, cases = 0;
static bool quick = false, allnan = false;

static unsigned bits(float v) { unsigned b; memcpy(&b, &v, 4); return b; }
static float fromBits(unsigned b) { float v; memcpy(&v, &b, 4); return v; }

// ---- 0035: sampling candidates -------------------------------------------------------------------------------------
// Reference: a row's first C ids in the order "the larger value, then the lower id" (-0.0 equal to +0.0); a row
// holding a NaN is flagged (the host then draws it from the whole row, so only the flag is checked for it).
static void fillRow(float* row, int V, int pattern, std::mt19937& g) {
    std::normal_distribution<float> nd(0.f, 3.f);
    switch (pattern) {
    case 0: for (int i = 0; i < V; ++i) row[i] = nd(g); break;                                    // plain
    case 1: for (int i = 0; i < V; ++i) row[i] = std::round(nd(g) * 2.f) / 2.f; break;            // many ties
    case 2: for (int i = 0; i < V; ++i) row[i] = -INFINITY;                                       // 100 finite
            for (int j = 0; j < 100; ++j) row[(int)(g() % V)] = nd(g); break;
    case 3: for (int i = 0; i < V; ++i) row[i] = 1.0f; break;                                     // all equal
    case 4: for (int i = 0; i < V; ++i) { unsigned u = g() % 3; row[i] = u == 0 ? 0.0f : u == 1 ? -0.0f : -1.0f - (g() % 7); } break;
    case 5: for (int i = 0; i < V; ++i) row[i] = nd(g);                                           // a few NaN
            for (int j = 0; j < 3; ++j) row[(int)(g() % V)] = NAN; break;
    case 6: for (int i = 0; i < V; ++i) {                                                         // extremes, denormals
                unsigned u = g() % 6;
                row[i] = u == 0 ? 3.0e38f : u == 1 ? -3.0e38f : u == 2 ? fromBits(1u + g() % 1000u) : u == 3 ? -fromBits(1u + g() % 1000u) : u == 4 ? INFINITY : nd(g);
            } break;
    case 7: {                                                                                     // the slices' edges
                for (int i = 0; i < V; ++i) row[i] = -5.f + nd(g) * 0.01f;
                const int L = (V + TF_CAND_SLICES - 1) / TF_CAND_SLICES;
                for (int s = 0; s < TF_CAND_SLICES; ++s) {
                    const int lo = s * L;
                    if (lo < V) row[lo] = 10.f + s;
                    if (lo + L - 1 < V) row[lo + L - 1] = 10.f + s;   // a tie with lo: lo first
                }
                row[V - 1] = 100.f;
            } break;
    default: for (int i = 0; i < V; ++i) row[i] = NAN; break;                                     // all NaN
    }
}

static void candidatesCase(int rows, int V, long long ld, int C, std::mt19937& g) {
    std::vector<float> h((size_t)rows * ld, 0.f);
    std::vector<int> pat(rows);
    for (int r = 0; r < rows; ++r) {
        pat[r] = (allnan && r == rows - 1) ? 99 : r % 8;
        fillRow(&h[(size_t)r * ld], V, pat[r], g);
        for (long long i = V; i < ld; ++i) h[(size_t)r * ld + i] = 1e30f;   // the stride's padding: never a candidate
    }
    float* d;
    unsigned long long* keys;
    int* nans;
    unsigned* out;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMalloc(&keys, (size_t)rows * TF_CAND_SLICES * TF_CAND_MAX * 8));
    CK(cudaMalloc(&nans, (size_t)rows * TF_CAND_SLICES * 4));
    CK(cudaMalloc(&out, (size_t)rows * TF_CAND_REC * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    tf_ds_cands_slices_kernel<<<dim3(TF_CAND_SLICES, rows), 256>>>(d, ld, V, C, keys, nans);
    CK(cudaGetLastError());
    tf_ds_cands_merge_kernel<<<rows, 1024>>>(d, ld, C, keys, nans, out);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<unsigned> rec((size_t)rows * TF_CAND_REC);
    CK(cudaMemcpy(rec.data(), out, rec.size() * 4, cudaMemcpyDeviceToHost));
    for (int r = 0; r < rows; ++r) {
        ++cases;
        const float* row = &h[(size_t)r * ld];
        const unsigned* q = &rec[(size_t)r * TF_CAND_REC];
        bool hasNan = false;
        for (int i = 0; i < V; ++i) hasNan |= std::isnan(row[i]);
        if (hasNan) {
            if (q[2 * TF_CAND_MAX] != 1u) { ++fails; printf("FAIL cands rows %d V %d C %d row %d (pattern %d): NaN flag %u, expected 1\n", rows, V, C, r, pat[r], q[2 * TF_CAND_MAX]); }
            continue;
        }
        std::vector<int> idx(V);
        for (int i = 0; i < V; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + C, idx.end(), [&](int a, int b) {
            if (row[a] != row[b]) return row[a] > row[b];
            return a < b;
        });
        bool ok = q[2 * TF_CAND_MAX] == 0u;
        int bad = -1;
        for (int t = 0; t < C && ok; ++t) {
            if ((int)q[TF_CAND_MAX + t] != idx[t] || q[t] != bits(row[idx[t]])) { ok = false; bad = t; }
        }
        if (!ok) {
            ++fails;
            printf("FAIL cands rows %d V %d C %d row %d (pattern %d): flag %u, first wrong at %d: id %u value %08x, expected id %d value %08x\n",
                   rows, V, C, r, pat[r], q[2 * TF_CAND_MAX], bad, bad >= 0 ? q[TF_CAND_MAX + bad] : 0u, bad >= 0 ? q[bad] : 0u, bad >= 0 ? idx[bad] : -1, bad >= 0 ? bits(row[idx[bad]]) : 0u);
        }
    }
    CK(cudaFree(d)); CK(cudaFree(keys)); CK(cudaFree(nans)); CK(cudaFree(out));
}

// ---- 0034: topk of unique int64 keys (the indexer's) ---------------------------------------------------------------
// Reference: the k largest keys of a row as a multiset (the kernels write them in no particular order).
static std::vector<long long> makeKeys(int rows, int n, long long ks, int mode, std::mt19937& g) {
    std::vector<long long> h((size_t)rows * ks, 0);
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < n; ++i) {
            int hi = mode == 0 ? (int)g() : (int)(g() % 9) - 4;   // mode 1: few distinct scores (ties on the high bits)
            h[(size_t)r * ks + i] = (long long)(((unsigned long long)(unsigned)hi << 32) | (unsigned long long)(0xffffffffu - (unsigned)i));
        }
    return h;
}

static bool checkTop(const std::vector<long long>& h, long long ks, int rows, int n, int k, const std::vector<long long>& top, const char* what) {
    bool all = true;
    for (int r = 0; r < rows; ++r) {
        ++cases;
        std::vector<long long> ref(h.begin() + (size_t)r * ks, h.begin() + (size_t)r * ks + n);
        std::partial_sort(ref.begin(), ref.begin() + k, ref.end(), std::greater<long long>());
        ref.resize(k);
        std::sort(ref.begin(), ref.end());
        std::vector<long long> got(top.begin() + (size_t)r * k, top.begin() + (size_t)(r + 1) * k);
        std::sort(got.begin(), got.end());
        if (got != ref) {
            ++fails; all = false;
            printf("FAIL %s rows %d n %d k %d row %d\n", what, rows, n, k, r);
        }
    }
    return all;
}

static void topkRegsCase(int rows, int n, int k, long long ks, int mode, std::mt19937& g) {
    auto h = makeKeys(rows, n, ks, mode, g);
    long long *d, *t;
    CK(cudaMalloc(&d, h.size() * 8));
    CK(cudaMalloc(&t, (size_t)rows * k * 8));
    CK(cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice));
    if (n <= 32768) tf_ds_topk_i64_regs_kernel<<<rows, 1024>>>(d, ks, n, k, t);
    else tf_ds_topk_i64_kernel<<<rows, 1024>>>(d, ks, n, k, t);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<long long> top((size_t)rows * k);
    CK(cudaMemcpy(top.data(), t, top.size() * 8, cudaMemcpyDeviceToHost));
    checkTop(h, ks, rows, n, k, top, n <= 32768 ? "topk regs" : "topk (upstream, n > 32768)");
    CK(cudaFree(d)); CK(cudaFree(t));
}

// ops.zig topkI64's slices path: n past one slice (4096), few rows.
static void topkSlicesCase(int rows, int n, int k, std::mt19937& g) {
    const int slice = 4096, ns = (n + slice - 1) / slice;
    if (!(n > slice && n <= 32768 && rows * ns <= 192 && ns * k <= 32768 && rows * ns * k <= 192 * 512)) return;
    auto h = makeKeys(rows, n, n, 0, g);
    long long *d, *cand, *t;
    CK(cudaMalloc(&d, h.size() * 8));
    CK(cudaMalloc(&cand, (size_t)192 * 512 * 8));   // ops.zig topk_scratch: topk_scratch_keys
    CK(cudaMalloc(&t, (size_t)rows * k * 8));
    CK(cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice));
    tf_ds_topk_i64_slices_kernel<<<rows * ns, 1024>>>(d, n, n, k, slice, cand);
    CK(cudaGetLastError());
    tf_ds_topk_i64_regs_kernel<<<rows, 1024>>>(cand, (long long)ns * k, ns * k, k, t);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<long long> top((size_t)rows * k);
    CK(cudaMemcpy(top.data(), t, top.size() * 8, cudaMemcpyDeviceToHost));
    checkTop(h, n, rows, n, k, top, "topk slices");
    CK(cudaFree(d)); CK(cudaFree(cand)); CK(cudaFree(t));
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "quick")) quick = true;
        if (!strcmp(argv[i], "allnan")) allnan = true;
    }
    std::mt19937 g(20261024);
    const int V = 129280;   // DeepSeek-V4.1's vocabulary
    // candidates: the served shapes (count = top_k + 8 <= 64; rows up to ROUND_ROWS 48), strides V and V + 128
    for (int C : {1, 9, 28, 64}) {
        if (quick && C != 28 && C != 64) continue;
        candidatesCase(quick ? 9 : 48, V, V, C, g);
        if (!quick) candidatesCase(9, V, V + 128, C, g);
    }
    if (!quick) { candidatesCase(3, 4096, 4096, 64, g); candidatesCase(2, 131072, 131072, 64, g); candidatesCase(2, 64, 64, 64, g); }
    // topk: the registers path's four sizes and their edges, ties on the high bits, a row stride
    const int ns_[] = {1, 37, 1024, 1025, 4096, 4097, 16384, 16385, 32768};
    for (int n : ns_)
        for (int k : {1, 64, 512, 2048}) {
            if (k > n || (quick && n != 4097 && n != 32768 && n != 1025)) continue;
            topkRegsCase(quick ? 2 : 5, n, k, n, 0, g);
            if (!quick) topkRegsCase(3, n, k, n + 3, 1, g);
        }
    for (int n : {4097, 8192, 20000, 32768})
        for (int rows : {1, 2, 4, 8})
            for (int k : {64, 512, 1024, 2048}) {
                if (quick && (rows != 2 || k == 1024)) continue;
                topkSlicesCase(rows, n, k, g);
            }
    if (!quick) for (int n : {32769, 65536}) for (int k : {64, 2048}) topkRegsCase(2, n, k, n, 0, g);
    printf("%s: %d of %d rows as the reference\n", fails ? "FAIL" : "PASS", cases - fails, cases);
    return fails ? 1 : 0;
}
