#include "../include/cudakernel.cuh"
#include <cstdio>
#include <chrono>
#include "../include/global.h"


/**
 * Sample CUDA device function which adds an element from array A and array B.
 *
 */
__global__ void cuda_kernel(double *A, double *B, double *C, int arraySize)
{
    // Get thread ID.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if thread is within array bounds.
    if ( tid < arraySize ) 
    {
        // Add a and b.
        C[tid] = A[tid] + B[tid];
    }
}

/**
 * Sample CUDA device function which computes the number of triangles (support) for each of the edge.
 *
 */

__global__ void cuda_computeSupport_kernel(int m, int *us, int *vs, int* flatarray, int n, int* indices, int* results)
{
	int from = blockDim.x * blockIdx.x + threadIdx.x;
  	int step = gridDim.x * blockDim.x;
	
	for(int i = from; i < m; i += step)
	{
		int u = us[i];
		int v = vs[i];
		
		int first1 = indices[u];
		int last1 = indices[u + 1];
		int first2 = indices[v];
		int last2 = indices[v + 1];

		int result = 0;

		while(first1 != last1 && first2 != last2)
		{
			if(flatarray[first1] >= n || flatarray[first2] >= n)
			{
				break;
			}
			if(flatarray[first1] < flatarray[first2])
			{
				++first1;
			}
			else if(flatarray[first1] > flatarray[first2])
			{
				++first2;
			}
			else
			{
				++result;
				++first1;
				++first2;
			}	
		}
		results[i] = result;
	}
		
}

/**
 * Wrapper function for the CUDA kernel function.
 * @param A Array A.
 * @param B Array B.
 * @param C Sum of array elements A and B directly across.
 * @param arraySize Size of arrays A, B, and C.
 */
void kernel(double *A, double *B, double *C, int arraySize) 
{

    // Initialize device pointers.
    double *d_A, *d_B, *d_C;

    // Allocate device memory.
    cudaMalloc((void**) &d_A, arraySize * sizeof(double));
    cudaMalloc((void**) &d_B, arraySize * sizeof(double));
    cudaMalloc((void**) &d_C, arraySize * sizeof(double));

    // Transfer arrays a and b to device.
    cudaMemcpy(d_A, A, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, arraySize * sizeof(double), cudaMemcpyHostToDevice);

    // Calculate blocksize and gridsize.
    dim3 blockSize(512, 1, 1);
    dim3 gridSize(512 / arraySize + 1, 1);

    // Launch CUDA kernel.
    cuda_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, arraySize);

    // Copy result array c back to host memory.
    cudaMemcpy(C, d_C, arraySize * sizeof(double), cudaMemcpyDeviceToHost);
}



int* graphKernel(const EdgeList& edges, int n, int* flatarray, int flatarraylength, int* indices)
{
	// Initialize device pointers.
    	int *d_us;
	int *d_vs;
	int *d_flatarray;
	int *d_indices;
	int *d_results;
	int *results;

	// Allocate device memory
	
	int m = edges.size();

	// right now, the code below for sending the edges by 2 different array is in preliminary stage, I will probably do a better approach in near future

	//int *us = new int[m];
	//int *vs = new int[m];

	//checking with pinned memory

	int *us, *vs;

	cudaMallocHost((void**)&us, m * sizeof(int));
	cudaMallocHost((void**)&vs, m * sizeof(int));
	cudaMallocHost((void**)&results, m * sizeof(int));
	
	for(int i = 0; i < m; i++)
	{
		us[i] = edges[i].first;
		vs[i] = edges[i].second;
	}
	
	// allocate memory in GPU for transfering the graph

	
	cudaMalloc((void**) &d_us, m * sizeof(int));
	cudaMalloc((void**) &d_vs, m * sizeof(int));
    	cudaMalloc((void**) &d_flatarray, flatarraylength * sizeof(int));
    	cudaMalloc((void**) &d_indices, (n+1) * sizeof(int));
	cudaMalloc((void**) &d_results, m * sizeof(int));

	// copy the CPU data to GPU
	cudaMemcpy(d_us, us, m * sizeof(int), cudaMemcpyHostToDevice);
    	cudaMemcpy(d_vs, vs, m * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_flatarray, flatarray, flatarraylength * sizeof(int), cudaMemcpyHostToDevice);
    	cudaMemcpy(d_indices, indices, (n+1) * sizeof(int), cudaMemcpyHostToDevice);

	// Calculate blocksize and gridsize.
    	dim3 blockSize(512, 1, 1);
    	dim3 gridSize(512 / m + 1, 1);

	auto tm1 = std::chrono::high_resolution_clock::now();

	cuda_computeSupport_kernel<<<gridSize, blockSize>>>(m, d_us, d_vs, d_flatarray, n, d_indices, d_results);
	
	auto tm2 = std::chrono::high_resolution_clock::now();

	onlyKernelTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tm2 - tm1).count();

	// cudaDeviceSynchronize();

	// Copy result array back to host memory.
    	cudaMemcpy(results, d_results, m * sizeof(int), cudaMemcpyDeviceToHost);
	
	return results;
}

