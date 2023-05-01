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
          for (int jj = j; jj < j + tile_size && jj < N; jj += 16)
          {
            __m512 c = _mm512_load_ps(&C[ii * N + jj]);
            for (int kk = k; kk < k + tile_size && kk < K; kk++)
            {
              __m512 a = _mm512_broadcastss_ps(_mm_load_ss(&A[ii * K + kk]));
              __m512 b = _mm512_loadu_ps(&B[kk * N + jj]);
              c = _mm512_add_ps(c, _mm512_mul_ps(a, b));
            }
            _mm512_store_ps(&C[ii * N + jj], c);
          }
        }
      }
    }
  }
}
