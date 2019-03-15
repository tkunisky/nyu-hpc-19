/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

int main (int argc, char *argv[]) {
  int i, tid;
  float sum;

  for (i=0; i < VECLEN; i++) {
    a[i] = b[i] = 1.0 * i;
  }

  sum = 0.0;

  // SOLUTION:
  // Inline the function call intended to be run in parallel, since sum cannot
  // be shared and accessed within the function (unless we make it global,
  // which would be an alternative solution).

  #pragma omp parallel shared(sum,a,b) private(tid)
  {
    tid = omp_get_thread_num();
    #pragma omp for reduction(+:sum)
    for (i=0; i < VECLEN; i++) {
        sum = sum + (a[i]*b[i]);
        printf("  tid= %d i=%d\n",tid,i);
    }
  }

  printf("Sum = %f\n",sum);
}
