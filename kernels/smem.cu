/*
 * Shared Memory Cache-Blocking
 *
 * loads blocks of A and B into shared memory and accumulates results in C
 */
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE> // BLOCKSIZE = 32
void __global__ gemm_smem(int M, int N, int K, const float *A, const float *B, float *C) {
  // allocate shared memory buffer for current block
  // SMEM contains blocks from A and B of BLOCKSIZE^2 elements
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // set starting position of A and B for current block
  // this is the corresponding row of A and column of B for the current block
  A += blockIdx.x * BLOCKSIZE * K;
  B += blockIdx.y * BLOCKSIZE;
  C += blockIdx.x * BLOCKSIZE * N + blockIdx.y * BLOCKSIZE;
  
  float tmp = 0.0;

  for (int blkId = 0; blkId < K; blkId += BLOCKSIZE) {
    // Load block data into shared memory
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
  
    // wait for SMEM to be filled
    __sycthread();

    // advance pointers onto next chunk
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;
  
    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
              Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}
