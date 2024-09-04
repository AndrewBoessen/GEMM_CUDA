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

  for (int blkId = 0; blkId < K; blk += BLOCKSIZE) {

  }
}
