#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

//
// General
//

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 1024

// Reduction kernel for sum
__global__ void reduction_kernel(double* sum, const double* a, long N) {
	__shared__ double smem[BLOCK_SIZE];
	int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx < N) smem[threadIdx.x] = a[idx];
	else smem[threadIdx.x] = 0;

	__syncthreads();
	if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
	__syncthreads();
	if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
	__syncthreads();
	if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
	__syncthreads();
	if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x + 64];
	__syncthreads();
	if (threadIdx.x <  32) {
		smem[threadIdx.x] += smem[threadIdx.x + 32];
		__syncwarp();
		smem[threadIdx.x] += smem[threadIdx.x + 16];
		__syncwarp();
		smem[threadIdx.x] += smem[threadIdx.x + 8];
		__syncwarp();
		smem[threadIdx.x] += smem[threadIdx.x + 4];
		__syncwarp();
		smem[threadIdx.x] += smem[threadIdx.x + 2];
		__syncwarp();
		if (threadIdx.x == 0) {
			sum[blockIdx.x] = smem[0] + smem[1];
		}
	}
}

// CPU matrix-vector multiplication implementation
// Assume A is square and is stored in row major: A[i, j] = A[N * i + j]
void matrix_vector_ref(double* Ax_ref, const double* A, const double* x, long N) {
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) {
    for (long j = 0; j < N; j++) {
      Ax_ref[i] += A[N * i + j] * x[j];
    }
  }
}

// Kernel for matrix-vector product per result entry
__global__ void matrix_vector_kernel(double* Ax, const double* A, const double* x, long N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    Ax[idx] = 0;
    for (long j = 0; j < N; j++) {
      Ax[idx] += A[idx * N + j] * x[j];
    }
  }
}

int main() {
  long N = (1UL<<10);

  double *A, *x, *A_d, *x_d, *Ax_ref, *Ax;

  // Initialize vector and matrix
  cudaMallocHost((void**)&A, N * N * sizeof(double));
  cudaMallocHost((void**)&x, N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = drand48();
    for long(j = 0; j < N; j++) {
      A[i * N + j] = drand48();
    }
  }

  // Get reference product
  cudaMallocHost((void**)&Ax_ref, N * sizeof(double));
  double tt = omp_get_wtime();
  matrix_vector_ref(Ax_ref, A, x, N);
  printf("CPU Bandwidth = %f GB/s\n", 2*N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  // Get GPU product
  cudaMalloc(&A_d, N*N*sizeof(double));
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMallocHost(&Ax, N*sizeof(double));

  cudaMemcpyAsync(A_d, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  tt = omp_get_wtime();
  matrix_vector(Ax, A, x, N);
  printf("GPU Bandwidth = %f GB/s\n", 2*N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double err = 0;
  for (long i = 0; i < N; i++) {
    err += (Ax_ref[i] - Ax[i]) * (Ax_ref[i] - Ax[i])
  }
  printf("Error = %f\n", err);

  // Cleanup
  cudaFree(A_d);
  cudaFree(x_d);
  cudaFreeHost(A);
  cudaFreeHost(x);
  cudaFreeHost(Ax);
  cudaFreeHost(Ax_ref);

  return 0;
}
