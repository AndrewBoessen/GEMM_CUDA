/*
 * Naive Implementation of a GEMM Kernel
 *
 * each element in the the matrix C is assigned one thread
 */

__global__ gemm_naive(int M, int N, int K, const float *A, const float *B, float *C){
  // position of threads element in C
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.0 // holds dot product of row of A and col of B
    for (int i = 0; i < K; i++) {
      tmp += A[x * K + i] * B[i * N + y];
    }

    // update element in C
    C[x * N + y] += tmp
  }
}
