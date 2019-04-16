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

// CPU inner product implementation
void inner_product_ref(double* ip_ptr, const double* a, const double* b){
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) sum += a[i] * b[i];
  *ip_ptr = sum;
}

// GPU inner product kernel
__global__ void inner_product_kernel(double* ip, const double* a, const double* b, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) {
    smem[threadIdx.x] = a[idx] * b[idx];
  }

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) ip[blockIdx.x] = smem[0] + smem[1];
  }
}

// Wrapper for inner product kernel
void inner_product(double* ip, double* a, double* b, long N) {
  double *x_d, *y_d, *z_d;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&y_d, N*sizeof(double));

  // Extra memory buffer for reduction across thread-blocks
  long N_work = 1;
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
       i > 1;
       i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) {
    N_work += i;
  }
  cudaMalloc(&z_d, N_work*sizeof(double));

  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(z_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  double* ip_d = z_d;
  long Nb = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE);
  inner_product_kernel<<<Nb,BLOCK_SIZE>>>(ip_d, x_d, y_d, N);
  while (Nb > 1) {
    long this_N = Nb;
    Nb = (Nb + BLOCK_SIZE - 1) / (BLOCK_SIZE);
    inner_product_kernel<<<Nb,BLOCK_SIZE>>>(ip_d + Nb, ip_d, N);
  }

  cudaMemcpyAsync(&sum, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
}

int main() {
  long N = (1UL<<15);

  double *x;
  double *y;

  cudaMallocHost((void**)&x, N * sizeof(double));
  cudaMallocHost((void**)&y, N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = drand48();
    y[i] = drand48();
  }

  double ip_ref, ip;
  double tt = omp_get_wtime();
  inner_product(&ip_ref, x, y, N);
  printf("CPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  tt = omp_get_wtime();

  tt = omp_get_wtime();
  inner_product(&ip, x, y, N);
  tt = omp_get_wtime();
  printf("GPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  printf("Error = %f\n", fabs(ip - ip_ref));

  cudaFreeHost(x);
  cudaFreeHost(y);

  return 0;
}
