#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

void swap(double** u1, double** u2) {
  double* tmp = *u1;
  *u1 = *u2;
  *u2 = tmp;
}

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

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, rank_i, rank_j, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  int p_copy = p;
  int pN = 1;
  while (p_copy > 1) {
    p_copy = p_copy / 4;
    pN *= 2;
  }
  rank_i = rank / pN;
  rank_j = rank % pN;

  int N, lN, max_iters;
  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);
  lN = N / pN;

  if (rank == 0) {
    printf("lN = %d\n", lN);
  }

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", rank, p, processor_name);

  // --- Start timed section ---
  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  double* lu = (double*) calloc(sizeof(double), (lN + 2) * (lN + 2));
  double* lu_new = (double*) calloc(sizeof(double), (lN + 2) * (lN + 2));
  double* lf = (double*) calloc(sizeof(double), (lN + 2) * (lN + 2));
  double* side_buffer = (double*) calloc(sizeof(double), lN + 2);

  for (int i = 0; i < (lN + 2) * (lN + 2); i++) {
    lu[i] = 0;
    lu_new[i] = 0;
    lf[i] = 1;
  }

  double h_sq = 1.0 / ((N + 1) * (N + 1));

  // --- Jacobi method iteration ---

  for (int iter = 0; iter < max_iters; iter++) {
    // Jacobi step for local points
    for (int i = 1; i < lN + 1; i++) {
      for (int j = 1; j < lN + 1; j++) {
        int flat_ix = i * (lN + 2) + j;
        lu_new[flat_ix] = 0.25 *
          (h_sq * lf[flat_ix] + sum_of_neighbors(lN + 2, lu, i, j));
      }
    }

    // Communicate ghost values
    MPI_Status status_top, status_bottom, status_left, status_right;
    // Send/receive at top
    if (rank_i > 0) {
      MPI_Send(
        lu_new + (lN + 2), lN + 2, MPI_DOUBLE,
        pN * (rank_i - 1) + rank_j, 0, MPI_COMM_WORLD);
      MPI_Recv(
        lu_new, lN + 2, MPI_DOUBLE,
        pN * (rank_i - 1) + rank_j, 0, MPI_COMM_WORLD, &status_top);
    }
    // Send/receive at bottom
    if (rank_i < pN - 1) {
      MPI_Send(
        lu_new + lN * (lN + 2), lN + 2, MPI_DOUBLE,
        pN * (rank_i + 1) + rank_j, 0, MPI_COMM_WORLD);
      MPI_Recv(
        lu_new + (lN + 1) * (lN + 2), lN + 2, MPI_DOUBLE,
        pN * (rank_i + 1) + rank_j, 0, MPI_COMM_WORLD, &status_bottom);
    }
    // Send/receive at left
    if (rank_j > 0) {
      for (int m = 0; m < lN + 2; m++) {
        side_buffer[m] = lu_new[(lN + 2) * m + 1];
      }
      MPI_Send(
        side_buffer, lN + 2, MPI_DOUBLE,
        pN * rank_i + (rank_j - 1), 0, MPI_COMM_WORLD);

      MPI_Recv(
        side_buffer, lN + 2, MPI_DOUBLE,
        pN * rank_i + (rank_j - 1), 0, MPI_COMM_WORLD, &status_left);
      for (int m = 0; m < lN + 2; m++) {
        lu_new[(lN + 2) * m] = side_buffer[m];
      }
    }
    // Send/receive at right
    if (rank_j < pN - 1) {
      for (int m = 0; m < lN + 2; m++) {
        side_buffer[m] = lu_new[(lN + 2) * m + lN];
      }
      MPI_Send(
        side_buffer, lN + 2, MPI_DOUBLE,
        pN * rank_i + (rank_j + 1), 0, MPI_COMM_WORLD);

      MPI_Recv(
        side_buffer, lN + 2, MPI_DOUBLE,
        pN * rank_i + (rank_j + 1), 0, MPI_COMM_WORLD, &status_right);
      for (int m = 0; m < lN + 2; m++) {
        lu_new[(lN + 2) * m + (lN + 1)] = side_buffer[m];
      }
    }

    // Swap lu with lu_new
    swap(&lu, &lu_new);

    if (iter % 100 == 0) {
      double lres = 0.0;
      double gres = 0.0;
      double res_entry = 0.0;

      for (int i = 1; i < lN + 1; i++) {
        for (int j = 1; j < lN + 1; j++) {
          int flat_ix = i * (lN + 2) + j;

          res_entry = ((4.0 * lu_new[flat_ix] - sum_of_neighbors(lN + 2, lu, i, j)) / h_sq - lf[flat_ix]);
          lres += res_entry * res_entry;
        }
      }

      MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      if (rank == 0) {
        printf("Iteration %5d  :  Residual = %g\n", iter, sqrt(gres));
      }
    }
  }

  free(lu);
  free(lu_new);
  free(side_buffer);
  free(lf);

  // --- End timed portion ---
  MPI_Barrier(MPI_COMM_WORLD);
  double duration = MPI_Wtime() - start_time;
  if (rank == 0) {
    printf("Time elapsed: %f seconds\n\n", duration);
  }

  MPI_Finalize();

  return 0;
}
