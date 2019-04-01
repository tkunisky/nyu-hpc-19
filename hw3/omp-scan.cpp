#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  long num_threads;
  long segment_length;

  #pragma omp parallel shared(num_threads, segment_length)
  {
    num_threads = omp_get_num_threads();
    segment_length = (n + num_threads - 1) / num_threads;

    // Parallel iteration over array segments
    #pragma omp for
    for (long loop_ix = 0; loop_ix < num_threads; loop_ix++) {
      long segment_ix = omp_get_thread_num();
      long segment_start = segment_length * segment_ix;
      long segment_end = segment_length * (segment_ix + 1);
      if (segment_end > n) {
        segment_end = n;
      }

      // Compute prefix sums for each segment independently
      if (segment_start > 0) {
        prefix_sum[segment_start] = A[segment_start - 1];
      } else {
        prefix_sum[segment_start] = 0;
      }
      for (long i = segment_start + 1; i < segment_end; i++) {
        prefix_sum[i] = prefix_sum[i - 1] + A[i - 1];
      }
    }
  }

  // Final correction for preceding partial sums in each segment
  for (long segment_ix = 1; segment_ix < num_threads; segment_ix++) {
    long segment_start = segment_length * segment_ix;
    long segment_end = segment_length * (segment_ix + 1);
    if (segment_end > n) {
      segment_end = n;
    }

    for (long i = segment_start; i < segment_end; i++) {
      prefix_sum[i] += prefix_sum[segment_start - 1];
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
