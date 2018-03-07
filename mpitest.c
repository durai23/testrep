#include <stdio.h>
#include <mpi.h>
int main ( int argc, char** argv )
{
int rank, size;
char processor_name [MPI_MAX_PROCESSOR_NAME];
int name_len;
// Initialize the MPI environment.
MPI_Init( &argc, &argv );
// Get the number of processes.
MPI_Comm_size ( MPI_COMM_WORLD, &size);
// Get the rank of the process.
MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
// Get the name of the processor.
MPI_Get_processor_name ( processor_name, &name_len );
// Print out.
printf( "Hello world from processor %s, rank %d out of %d processors.\n", processor_name,
rank, size);
// Finalize the MPI environment.
MPI_Finalize();
}
