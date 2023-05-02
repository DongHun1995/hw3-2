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
  // 타일링 64, 128다 해봤는데 32가 젤빠름
  int tile_size = 32;

  // 처음에 scatter 다하려다가 계산 이상하케 나와서 A matrix만 scatter
  int local_M = M / mpi_world_size;
  int local_N = N;
  int local_K = K;

  // scatter 하고 gather하는 메트릭스 A, C에 memory alligned, //  malloc하면 sagfault
  float *local_A = (float *)aligned_alloc(32, local_M * local_K * sizeof(float));
  float *local_C = (float *)aligned_alloc(32, local_M * local_N * sizeof(float));

  // 프로세스에 matrix A scatter
  MPI_Scatter(A, local_M * local_K, MPI_FLOAT, local_A, local_M * local_K, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // 프로세스에 matrix B Bcast
  MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // local_c 초기화
  memset(local_C, 0, local_M * local_N * sizeof(float));

  // Loop over the tiles of local_C
#pragma omp parallel for num_threads(threads_per_process)
  for (int i = 0; i < local_M; i += tile_size)
  {
    for (int k = 0; k < local_K; k += tile_size)
    {
      for (int j = 0; j < local_N; j += tile_size)
      {
        // 타일링 적용
        for (int ii = i; ii < i + tile_size && ii < local_M; ii++)
        {
          for (int jj = j; jj < j + tile_size && jj < local_N; jj += 16)
          { // simd 명령어 적용 C에 local c의 원소 16개 가져옴
            __m512 c = _mm512_load_ps(&local_C[ii * local_N + jj]);
            for (int kk = k; kk < k + tile_size && kk < local_K; kk++)
            { // local_A에 1개, B에 16개 가져와서 연산후 곱셈, 덧셈 수행
              __m512 a = _mm512_broadcastss_ps(_mm_load_ss(&local_A[ii * local_K + kk]));
              __m512 b = _mm512_load_ps(&B[kk * local_N + jj]);
              c = _mm512_add_ps(c, _mm512_mul_ps(a, b));
            }
            _mm512_store_ps(&local_C[ii * local_N + jj], c);
          }
        }
      }
    }
  }

  // 연산 끝난거 gather
  MPI_Gather(local_C, local_M * local_N, MPI_FLOAT, C, local_M * local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // 메모리 free
  free(local_A);
  free(local_C);
}
