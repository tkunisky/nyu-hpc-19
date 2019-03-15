// Usage:
// $ g++ -O3 -std=c++11 Laplace.cpp && ./a.out 10000 Jacobi 
//
// The first argument is N and the second is the method. Optionally, a third
// parameter may be passed which sets the maximum number of iterations (5000
// by default):
// $ g++ -O3 -std=c++11 Laplace.cpp && ./a.out 10000 Jacobi 100

#include <stdio.h>
#include <math.h>
#include "utils.h"


// Computes \ell^2 norm of difference of u and v
double norm_diff(long n, double *u, double *v) {
  double ret = 0.0;
  
  for (long i = 0; i < n; i++) {
    ret += (u[i] - v[i]) * (u[i] - v[i]);
  }

  return sqrt(ret);
}


// Implements multiplication by discretized operator A
void MultA(long n, double *u, double *out) {
  double pre_factor = (n + 1) * (n + 1);

  out[0] = pre_factor * (2.0 * u[0] - u[1]);

  for (long i = 1; i < n - 1; i++) {
    out[i] = pre_factor * (2.0 * u[i] - u[i + 1] - u[i - 1]);
  }

  // -1.0 here is for boundary condition:
  out[n - 1] = pre_factor * (2.0 * u[n - 1] - u[n - 2] - 1.0);
}


// If u_old and u_new point to distinct arrays of double[n], then this 
// implements the Jacobi method, where u_new is the next iterate. If u_old
// and u_new point to the same array, then this implements the Gauss-Seidel
// method, where u is overwritten with the updated values.
void SolverStep(long n, double *f, double *u_old, double *u_new) {
  u_new[0] = 0.5 * (f[0] / (n + 1) / (n + 1) + u_old[1]);

  for (long i = 1; i < n - 1; i++) {
    u_new[i] = 0.5 * (f[i] / (n + 1) / (n + 1) + u_old[i - 1] + u_old[i + 1]);
  }

  // +1.0 here is for boundary condition:
  u_new[n - 1] = 0.5 * (f[n - 1] / (n + 1) / (n + 1) + u_old[n - 2] + 1.0);
}


int main(int argc, char** argv) {
  // Read arguments
  long n = atoi(argv[1]);
  long max_iters = 5000;
  if (argc >= 4) {
    max_iters = atoi(argv[3]);
  }
  bool is_jacobi = (strcmp(argv[2], "Jacobi") == 0);
  
  if (is_jacobi) {
    printf("Using method: Jacobi\n\n");
  } else {
    printf("Using method: Gauss-Seidel\n\n");
  }

  // Allocate memory, with either one vector for Gauss-Seidel or two for Jacobi
  double* f = (double*) malloc(n * sizeof(double));
  double* u_old = (double*) malloc(n * sizeof(double));
  double* lhs = (double*) malloc(n * sizeof(double));
  double* u_new;
  if (is_jacobi) {
    u_new = (double*) malloc(n * sizeof(double));
  } else {
    u_new = u_old;
  }

  // Initialize data
  for (long i = 0; i < n; i++) {
    u_old[i] = 0.0;
    u_new[i] = 0.0;
    f[i] = 1.0;
  }
  MultA(n, u_new, lhs);

  double init_res = norm_diff(n, lhs, f);
  double this_res = 0.0;

  printf("Iteration     Residual Norm \n");

  Timer t;
  t.tic();

  // Main loop
  for (long it = 0; it < max_iters; it++) {
    // Swap u_old and u_new pointers (in case of Jacobi method)
    double *tmp = u_old;
    u_old = u_new;
    u_new = tmp;
    
    SolverStep(n, f, u_old, u_new);
    MultA(n, u_new, lhs);
    this_res = norm_diff(n, lhs, f);
    printf("%9d     %5.6f\n", (int) it, this_res);
    
    if (init_res > 1e6 * this_res) {
      printf("Reached stopping condition; terminating.\n");

      break;
    }
  }

  double time = t.toc();

  printf("\nCumulative residual shrinkage factor: %f\n", init_res / this_res);
  printf("\nTime (seconds): %f\n", time);

  // Cleanup
  free(f);
  free(u_old);
  free(lhs);
  if (is_jacobi) {
    free(u_new);
  }

  return 0;
}
