// Usage:
// For serial,
//   ./jacobi2D-omp 100 10000
// For parallel,
//   ./jacobi2D-omp 100 10000 1

#include <stdio.h>
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
	#include <omp.h>
#endif


// Swap two pointers
void swap(double** u1, double** u2) {
  double* tmp = *u1;
  *u1 = *u2;
  *u2 = tmp;
}

// Sum the neighboring coordinates of a vectorized array
double sum_of_neighbors(int N, double* u, int i, int j) {
  double val_left = 0.0, val_right = 0.0, val_up = 0.0, val_down = 0.0;

  if (i > 0) { 
    val_up = u[(i - 1) * N + j]; 
  }
  if (i < N - 1) { 
    val_down = u[(i + 1) * N + j]; 
  }
  if (j > 0) { 
    val_left = u[i * N + (j - 1)]; 
  }
  if (j < N - 1) { 
    val_right = u[i * N + (j + 1)]; 
  }

  return val_left + val_right + val_up + val_down;
}

// Multiplication by discretized negative Laplacian matrix
void A_times(int N, double* u, double* ret) {
  double h_sq = 1.0 / ((N + 1) * (N + 1));
  
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int flat_ix = i * N + j;

      double val_here = u[flat_ix];
      
      ret[flat_ix] = 
        1.0 / h_sq * 
        (4.0 * val_here - sum_of_neighbors(N, u, i, j));
    }
  }
}

// Compute the norm of the residual
double residual(int N, double* u, double* f) {
  // Multiply by objective matrix
  double* Au = (double*) malloc(N * N * sizeof(double));
  A_times(N, u, Au);

  // Accumulate squared norm
  double residual_norm_sq = 0.0;
  for (int i = 0; i < N * N; i++) {
    residual_norm_sq += (Au[i] - f[i]) * (Au[i] - f[i]);
  }

  free(Au);

  return sqrt(residual_norm_sq);
}

// Serial iteration
void jacobi_iters(int N, int iters, double* f, double* u_prev, double* u) {
  double h_sq = 1.0 / ((N + 1) * (N + 1));

  for (int ix = 0; ix < iters; ix++) {
    // Set u to the next iteration step
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        int flat_ix = i * N + j;
        u[flat_ix] = 0.25 * 
          (h_sq * f[flat_ix] + sum_of_neighbors(N, u_prev, i, j));
      }
    }

    // Swap u and u_prev
    swap(&u, &u_prev);
  }

  // Swap one more time so that u contains the latest update
  swap(&u, &u_prev);
}

// Parallel iteration
void jacobi_iters_par(int N, int iters, double* f, double* u_prev, double* u) {
  double h_sq = 1.0 / ((N + 1) * (N + 1));

  for (int ix = 0; ix < iters; ix++) {
    // Set u to the next iteration step
    #pragma omp parallel
    {

    #pragma omp for
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        int flat_ix = i * N + j;
        u[flat_ix] = 0.25 * 
          (h_sq * f[flat_ix] + sum_of_neighbors(N, u_prev, i, j));
      }
    }

    }

    // Swap u and u_prev
    swap(&u, &u_prev);
  }

  // Swap one more time so that u contains the latest update
  swap(&u, &u_prev);
}


int main(int argc, char** argv) {
  int N = atoi(argv[1]);
  int iters = atoi(argv[2]);
  
  Timer timer;

  // Keep two steps of u since Jacobi needs to do a full pass on the current
  // values.
  double* u_prev = (double*) malloc(N * N * sizeof(double));
  double* u = (double*) malloc(N * N * sizeof(double));
  double* f = (double*) malloc(N * N * sizeof(double));

  for (int i = 0; i < N * N; i++) {
    u_prev[i] = 0.0;
    u[i] = 0.0;
    f[i] = 1.0;
  }

  printf("Initial residual: %3f\n", residual(N, u, f));

  timer.tic();
  if (argc <= 3) {
    jacobi_iters(N, iters, f, u, u_prev);
  } else {
    jacobi_iters_par(N, iters, f, u, u_prev);
  }
  double time = timer.toc();

  printf("Final residual:   %3f\n", residual(N, u, f));
  printf("Time (seconds):   %3f\n", time);

  free(u);
  free(f);

  return 0;
}
