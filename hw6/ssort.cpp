// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor
  if (argc != 2) {
    printf("Usage: ssort <N>\n");
    return 1;
  }
  int N = atoi(argv[1]);
  printf("Running with N = %d\n", N);

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  // --- Start timed portion ---
  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int* local_splitters = (int*) malloc((p - 1) * sizeof(int));
  int splitter_ix = 0;
  for (int arr_ix = N / p; arr_ix < N; arr_ix += N / p) {
    local_splitters[splitter_ix] = vec[arr_ix];
    splitter_ix += 1;
  }

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int total_splitters = p * (p - 1);
  int* merged_splitters = NULL;
  if (rank == 0) {
    merged_splitters = (int*) malloc(total_splitters * sizeof(int));
  }
  MPI_Gather(
    local_splitters, p - 1, MPI_INT,
    merged_splitters, p - 1, MPI_INT,
    0, MPI_COMM_WORLD);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  int* final_splitters = (int*) malloc((p - 1) * sizeof(int));
  if (rank == 0) {
    std::sort(merged_splitters, merged_splitters + total_splitters);

    splitter_ix = 0;
    for (int arr_ix = p - 1; arr_ix < total_splitters; arr_ix += p - 1) {
      final_splitters[splitter_ix] = merged_splitters[arr_ix];
      splitter_ix += 1;
    }
  }

  // root process broadcasts splitters to all other processes
  MPI_Bcast(final_splitters, p - 1, MPI_INT, 0, MPI_COMM_WORLD);

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
  int* sdispls = (int*) malloc(p * sizeof(int));
  sdispls[0] = 0;
  for (int i = 0; i < p - 1; i++) {
    sdispls[i + 1] =
      std::lower_bound(vec, vec + N, final_splitters[i]) - vec;
  }

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  int* num_sending_per_process = (int*) malloc(p * sizeof(int));
  for (int i = 0; i < p; i++) {
    if (i < p - 1) {
      num_sending_per_process[i] = sdispls[i + 1] - sdispls[i];
    } else {
      num_sending_per_process[i] = N - sdispls[i];
    }
  }

  int* num_receiving_per_process = (int*) malloc(p * sizeof(int));
  MPI_Alltoall(
    num_sending_per_process, 1, MPI_INT,
    num_receiving_per_process, 1, MPI_INT,
    MPI_COMM_WORLD);

  int* rdispls = (int*) malloc(p * sizeof(int));
  rdispls[0] = 0;
  for (int i = 1; i < p; i++) {
    rdispls[i] = rdispls[i - 1] + num_receiving_per_process[i - 1];
  }

  int total_receiving = 0;
  for (int i = 0; i < p; i++) {
    total_receiving += num_receiving_per_process[i];
  }
  int* final_vec = (int*) malloc(total_receiving * sizeof(int));

  MPI_Alltoallv(
    vec, num_sending_per_process, sdispls, MPI_INT,
    final_vec, num_receiving_per_process, rdispls, MPI_INT,
    MPI_COMM_WORLD);

  // do a local sort of the received data
  std::sort(final_vec, final_vec + total_receiving);

  // --- End timed portion ---
  MPI_Barrier(MPI_COMM_WORLD);
  double duration = MPI_Wtime() - start_time;
  if (rank == 0) {
    printf("Time elapsed: %f seconds\n\n", duration);
  }

  // every process writes its result to a file
  FILE* fd = NULL;
  char filename[256];
  snprintf(filename, 256, "output%02d.txt", rank);
  fd = fopen(filename, "w+");

  if (NULL == fd) {
    printf("Error opening file\n");
    return 1;
  }

  for (int i = 0; i < total_receiving; i++) {
    fprintf(fd, "%d\n", final_vec[i]);
  }
  fclose(fd);

  free(vec);
  free(local_splitters);
  if (rank == 0) {
    free(merged_splitters);
  }
  free(final_splitters);
  free(num_sending_per_process);
  free(num_receiving_per_process);
  free(sdispls);
  free(rdispls);
  free(final_vec);

  MPI_Finalize();

  return 0;
}
