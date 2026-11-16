// Sketch (not built by the Python package): tensor-parallel Linear whose
// epilogue pushes partial sums to peers with NVSHMEM one-sided puts, so the
// reduce overlaps the next layer's GEMM instead of a blocking all-reduce.
//
// Build (GPU node): nvcc -DHAVE_NVSHMEM -I$NVSHMEM_HOME/include \
//   -L$NVSHMEM_HOME/lib -lnvshmem -c matmul_nvshmem.cu
#pragma once
#ifdef HAVE_NVSHMEM
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// Each rank computes C_local = A @ B_shard, then one-sided puts its partial
// into every peer's symmetric buffer and signals; consumers spin on the
// signal only when they actually need the reduced tile (late binding =
// exposed comm shrinks to the tail, not the whole collective).
__global__ void matmul_nvshmem_put(const float* A, const float* B, float* sym_out,
                                   int* sym_flag, int M, int N, int K, int pe, int npes) {
  // ponytail: naive tile; swap in cuBLAS/CUTLASS GEMM and put only the epilogue.
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;
  for (int j = 0; j < N; ++j) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + j];
    for (int peer = 0; peer < npes; ++peer)
      nvshmem_float_p(sym_out + row * N + j, acc, peer);  // one-sided, no peer CPU/GPU sync
  }
  __threadfence_system();
  if (threadIdx.x == 0)
    for (int peer = 0; peer < npes; ++peer)
      nvshmem_int_p(sym_flag, pe + 1, peer);  // signal: tile from `pe` is ready
}
#endif  // HAVE_NVSHMEM
