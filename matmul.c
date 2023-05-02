#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void matmul(float *A, float *B, float *C, int M, int N, int K,
            int threads_per_process, int mpi_rank, int mpi_world_size)
{
  // Set the tile size
  int tile_size = 32;

  // Compute local sizes for A and C
  int local_M = M / mpi_world_size;
  int local_N = N;
  int local_K = K;

  // Allocate memory for local matrices
  float *local_A = (float*)aligned_alloc(32, local_M * local_K * sizeof(float));
  float *local_C = (float*)aligned_alloc(32, local_M * local_N * sizeof(float));

  // Scatter A across processes
  MPI_Scatter(A, local_M * local_K, MPI_FLOAT, local_A, local_M * local_K, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Broadcast B to all processes
  MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Initialize local_C to zero
  memset(local_C, 0, local_M * local_N * sizeof(float));

  // Loop over the tiles of local_C
#pragma omp parallel for num_threads(threads_per_process)
  for (int i = 0; i < local_M; i += tile_size)
  {
    for (int k = 0; k < local_K; k += tile_size)
    {
      for (int j = 0; j < local_N; j += tile_size)
      {
        // Loop over the tiles of local_A and B
        for (int ii = i; ii < i + tile_size && ii < local_M; ii++)
        {
          for (int jj = j; jj < j + tile_size && jj < local_N; jj += 16)
          {
            __m512 c = _mm512_load_ps(&local_C[ii * local_N + jj]);
            for (int kk = k; kk < k + tile_size && kk < local_K; kk++)
            {
              __m512 a = _mm512_broadcastss_ps(_mm_load_ss(&local_A[ii * local_K + kk]));
              __m512 b = _mm512_loadu_ps(&B[kk * local_N + jj]);
              c = _mm512_add_ps(c, _mm512_mul_ps(a, b));
            }
            _mm512_store_ps(&local_C[ii * local_N + jj], c);
          }
        }
      }
    }
  }

  // Gather local_C to C
  MPI_Gather(local_C, local_M * local_N, MPI_FLOAT, C, local_M * local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Free memory for local matrices
  free(local_A);
  free(local_C);
}
