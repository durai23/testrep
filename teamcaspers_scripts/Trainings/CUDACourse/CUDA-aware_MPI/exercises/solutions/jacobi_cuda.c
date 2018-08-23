/*
 * Copyright 2012 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef USE_MPI
#include <mpi.h>
#endif //USE_MPI
#include <omp.h>
#include <cuda_runtime.h>

/**
 * @brief Does one Jacobi iteration on u_d writing the results to
 *        unew_d on all interior points of the domain.
 *
 * The Jacobi iteration solves the poission equation with diriclet
 * boundary conditions and a zero right hand side and returns the max
 * norm of the residue, executes synchronously.
 *
 * @param[in] u_d            pointer to device memory holding the
 *                           solution of the last iteration including
 *                           boundary.
 * @param[out] unew_d        pointer to device memory were the updates
 *                           solution should be written
 * @param[in] n              number of points in y direction
 * @param[in] m              number of points in x direction
 * @param[in,out] residue_d  pointer to a single float value in device
 * 			     memory, needed a a temporary storage to
 * 			     calculate the max norm of the residue.
 * @return		     the residue of the last iteration
 */
float launch_jacobi_kernel(const float* const u_d, float* const unew_d,
                           const int n, const int m, float* const residue_d);

/**
 * @brief Copies all inner points from unew_d to u_d, executes
 *        asynchronously.
 *
 * @param[out] u_d    pointer to device memory holding the solution of
 * 		      the last iteration including boundary which
 * 		      should be updated with unew_d
 * @param[in] unew_d  pointer to device memory were the updated
 *                    solution is saved
 * @param[in] n       number of points in y direction
 * @param[in] m       number of points in x direction
 */
void launch_copy_kernel(float* const u_d, const float* const unew_d,
		                const int n, const int m);

void checkCUDAError(const char* action);
#define CUDA_CALL( call )		\
{								\
	call;						\
	checkCUDAError( #call );    \
}

int handle_command_line_arguments(int argc, char** argv);

int init(int argc, char** argv);

void finalize();

void start_timer();
void stop_timer();

void jacobi();

int n, m;
int n_global;

int rank = 0;
int size = 1;

int iter = 0;
int iter_max = 1000;

double starttime;
double runtime;

const float pi = 3.1415926535897932384626f;
const float tol = 1.0e-5f;
float residue = 1.0f;

float* u;
float* unew;
float* y0;

float* u_d;
float* unew_d;
float* residue_d;

#ifdef USE_MPI
float* sendBuffer;
float* recvBuffer;
#endif //USE_MPI
/********************************/
/****         MAIN            ***/
/********************************/
int main(int argc, char** argv)
{
	if ( init(argc, argv) )
	{
		return -1;
	}

#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif //USE_MPI
	start_timer();

	// Main calculation
	jacobi();

#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif //USE_MPI
	stop_timer();

	finalize();
}

/********************************/
/****        JACOBI           ***/
/********************************/
void jacobi()
{
	while (residue > tol && iter < iter_max)
	{
		residue = launch_jacobi_kernel(u_d, unew_d, n, m, residue_d);

#ifdef USE_MPI
		float globalresidue = 0.0f;
		MPI_Allreduce( &residue, &globalresidue, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD );
		residue = globalresidue;
#endif //USE_MPI
#ifdef USE_MPI
		if ( size == 2 )
		{
			MPI_Status status;
			if ( rank == 0)
			{
				MPI_Sendrecv( unew_d+(n-2)*m+1, m-2, MPI_FLOAT, 1, 0, u_d+(n-1)*m+1 , m-2, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status );
			}
			else
			{
				MPI_Sendrecv( unew_d + 1*m+1, m-2, MPI_FLOAT, 0, 0, u_d+0*m+1, m-2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status );
			}
		}
#endif //USE_MPI
		launch_copy_kernel(u_d, unew_d, n, m);

		if (rank == 0 && iter % 100 == 0)
			printf("%5d, %0.6f\n", iter, residue);

		iter++;
	}

	CUDA_CALL( cudaMemcpy( u, u_d, m*n*sizeof(float), cudaMemcpyDeviceToHost ));
}

/********************************/
/**** Initialization routines ***/
/********************************/

int init(int argc, char** argv)
{
	char *str = NULL;
	if ((str = getenv("MPI_LOCALRANKID")) != NULL)
	{
		rank = atoi(str);
	}

	CUDA_CALL( cudaSetDevice( rank ));
#ifdef USE_MPI
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if ( size != 1 && size != 2 )
	{
		if ( rank == 0)
		printf("Error: %s can only run with 1 or 2 processes!\n",argv[0]);
		return -1;
	}
#endif //USE_MPI

	if (handle_command_line_arguments(argc, argv)) {
		return -1;
	}

	u = (float*) malloc(n * m * sizeof(float));
	unew = (float*) malloc(n * m * sizeof(float));
	y0 = (float*) malloc(n * sizeof(float));

#ifdef OMP_MEMLOCALTIY
#pragma omp parallel for
	for( int j = 0; j < n; j++)
	{
		for( int i = 0; i < m; i++ )
		{
			unew[j *m+ i] = 0.0f;
			u[j *m+ i] = 0.0f;
		}
	}
#else
	memset(u, 0, n * m * sizeof(float));
	memset(unew, 0, n * m * sizeof(float));
#endif //OMP_MEMLOCALTIY

	// set boundary conditions
#pragma omp parallel for
	for (int i = 0; i < m; i++)
	{
		//Set top boundary condition only for rank 0 (rank responsible of the upper halve of the domain)
		if (rank == 0)
			u[0 * m + i] = 0.f;
		//Set bottom boundary condition only for rank 1 (rank responsible of the lower halve of the domain)
		if (rank == 0 || size == 1)
			u[(n - 1) * m + i] = 0.f;
	}

	int j_offset = 0;
	if (size == 2 && rank == 1)
	{
		j_offset = n - 2;
	}
	for (int j = 0; j < n; j++)
	{
		y0[j] = sinf(pi * (j_offset + j) / (n - 1));
		u[j * m + 0] = y0[j];
		u[j * m + (m - 1)] = y0[j] * expf(-pi);
	}

#pragma omp parallel for
	for (int i = 1; i < m; i++)
	{
		if (rank == 0)
			unew[0 * m + i] = 0.f;
		if (rank == 1 || size == 1)
			unew[(n - 1) * m + i] = 0.f;
	}
#pragma omp parallel for
	for (int j = 1; j < n; j++)
	{
		unew[j * m + 0] = y0[j];
		unew[j * m + (m - 1)] = y0[j] * expf(-pi);
	}

	CUDA_CALL( cudaMalloc( (void**)&u_d, n*m * sizeof(float) ));
	CUDA_CALL( cudaMalloc( (void**)&unew_d, n*m * sizeof(float) ));
	CUDA_CALL( cudaMalloc( (void**)&residue_d, sizeof(float) ));

	CUDA_CALL( cudaMemcpy( u_d, u, m*n*sizeof(float), cudaMemcpyHostToDevice ));
	CUDA_CALL( cudaMemcpy( unew_d, unew, m*n*sizeof(float), cudaMemcpyHostToDevice ));
	return 0;
}

int handle_command_line_arguments(int argc, char** argv)
{
	if (argc > 3)
	{
		if (rank == 0)
			printf("usage: %s [n] [m]\n", argv[0]);
		return -1;
	}

	n = 4096;
	if (argc >= 2)
	{
		n = atoi(argv[1]);
		if (n <= 0)
		{
			if (rank == 0)
				printf("Error: The number of rows (n=%i) needs to positive!\n",n);
			return -1;
		}
	}
	if (size == 2 && n % 2 != 0)
	{
		if (rank == 0)
			printf( "Error: The number of rows (n=%i) needs to be devisible by 2 if two processes are used!\n",n);
		return -1;
	}
	m = n;
	if (argc >= 3)
	{
		m = atoi(argv[2]);
		if (m <= 0)
		{
			if (rank == 0)
				printf( "Error: The number of columns (m=%i) needs to positive!\n", m);
			return -1;
		}
	}

	n_global = n;

	if (size == 2)
	{
		//Do a domain decomposition and add one row for halo cells
		n = n / 2 + 1;
	}

	if (rank == 0)
	{
		struct cudaDeviceProp devProp;
		CUDA_CALL( cudaGetDeviceProperties( &devProp, rank ));
		printf("Jacobi relaxation Calculation: %d x %d mesh with "
				"%d processes and one %s for each process (%d rows per process).\n", n_global, m,
				size, devProp.name,n);
	}

	return 0;
}

/********************************/
/****  Finalization routines  ***/
/********************************/

void finalize()
{
	CUDA_CALL( cudaDeviceSynchronize());
	CUDA_CALL( cudaFree( residue_d ));
	CUDA_CALL( cudaFree(unew_d));
	CUDA_CALL( cudaFree(u_d));

	free(y0);
	free(unew);
	free(u);

#ifdef USE_MPI
	MPI_Finalize();
#endif //USE_MPI
}

/********************************/
/****    Timing functions     ***/
/********************************/
void start_timer()
{
#ifdef USE_MPI
	starttime = MPI_Wtime();
#else
	starttime = omp_get_wtime();
#endif //USE_MPI
}

void stop_timer()
{
#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
	runtime = MPI_Wtime() - starttime;
#else
	runtime = omp_get_wtime() - starttime;
#endif //USE_MPI
	if (rank == 0)
		printf(" total: %f s\n", runtime);
}

