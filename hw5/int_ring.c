#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double time_ring(long N, long data_size, MPI_Comm comm) {
  int rank, num_processes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &num_processes);

  int* msg = (int*) malloc(data_size * sizeof(int));
  if (rank == 0) {
    for (int i = 0; i < data_size; i++) {
      msg[i] = 0;
    }
  }

  int my_source, my_target;
  // Set my_source to be where this process expects to receive from and
  // my_target to be where this process will send to.
  my_source = rank - 1;
  if (rank == 0) {
    my_source = num_processes - 1;
  }
  my_target = rank + 1;
  if (rank == num_processes - 1) {
    my_target = 0;
  }

  // Start timed section
  MPI_Barrier(comm);
  double tt = MPI_Wtime();

  for (long repeat = 0; repeat < N; repeat++) {
    MPI_Status status;

    if (rank == 0) {
      MPI_Send(msg, data_size, MPI_INT, my_target, repeat, comm);
    }
    MPI_Recv(msg, data_size, MPI_INT, my_source, repeat, comm, &status);
    if (rank > 0) {
      // Even when sending a large array, only increment one entry so that
      // arithmetic time does not factor into network bandwidth
      msg[0] += rank;
      MPI_Send(msg, data_size, MPI_INT, my_target, repeat, comm);
    }
  }
  
  MPI_Barrier(comm);
  tt = MPI_Wtime() - tt;
  // End timed section

  if (rank == 0) {
    printf(
      "  Error: %ld\n", 
      msg[0] - N * num_processes * (num_processes - 1) / 2);
  }
  
  free(msg);
  
  return tt;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank;
  MPI_Comm_rank(comm, &rank);

  if (argc < 2) {
    printf("Usage: mpirun ./int_ring <N> \n");
    return 1;
  }
  long N = atoi(argv[1]);

  // Run small data test for latency estimation
  long data_size = 1;
  if (rank == 0) {
    printf("Data size %ld:\n", data_size);
  }
  double tt = time_ring(N, data_size, comm);
  if (rank == 0) {
    printf("  Latency:   %e ms\n", tt / N * 1000);
  }

  // Run large data test for bandwidth estimation
  data_size = 5 * 1e5;
  if (rank == 0) {
    printf("\nData size %ld:\n", data_size);
  }
  tt = time_ring(N, data_size, comm);
  if (rank == 0) {
    printf("  Bandwidth: %e GB/s\n", data_size * N / tt / 1e9);
  }

  MPI_Finalize();
  return 0;
}
