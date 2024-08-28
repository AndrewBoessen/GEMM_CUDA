/*
 * Global Memory Coalescing
 *
 * group memory access of all threads in same warp together
 */

__global__ gemm_coalesce(int M, int N, int K, const float *A, const float *B, float *C) {
  // position of threads element in C
  const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE)
  const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE)

  if (x < M && y < N) {
    float tmp = 0.0 // holds dot product of row of A and col of B
    for (int i = 0; i < K; i++) {
      tmp += A[x * K + i] * B[i * N + y];
    }

    // update element in C
    C[x * N + y] += tmp
  }
}
