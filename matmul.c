#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void matmul(float *A, float *B, float *C, int M, int N, int K,
            int threads_per_process, int mpi_rank, int mpi_world_size)
{
  // Set the tile size
  int tile_size = 64;

// Loop over the tiles of C
#pragma omp parallel for num_threads(threads_per_process)
  for (int i = 0; i < M; i += tile_size)
  {
    for (int k = 0; k < K; k += tile_size)
    {
      for (int j = 0; j < N; j += tile_size)
      {
        // Loop over the tiles of A and B
        for (int ii = i; ii < i + tile_size && ii < M; ii++)
        {
          for (int kk = k; kk < k + tile_size && kk < K; kk++)
          {
            for (int jj = j; jj < j + tile_size && jj < N; jj++)
            {
              C[ii * N + jj] += A[ii * K + kk] * B[kk * N + jj];
            }
          }
        }
      }
    }
  }
}
