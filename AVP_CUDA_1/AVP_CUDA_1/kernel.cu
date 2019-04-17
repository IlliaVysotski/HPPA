
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>  
#include <cuda_runtime.h> 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <intrin.h>
#include <ctime>
#include <chrono>
#include <ratio>
#pragma comment(lib, "cudart")

#define N 4096
#define THREADS_PER_BLOCK 1024

__global__ void reduceBase(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

//__global__ void reduceBase(int *g_idata, int *g_odata) {
//		extern __shared__ int sdata[];
//		// each thread loads one element from global to shared mem
//		unsigned int tid = threadIdx.x;
//		unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
//		sdata[tid] = g_idata[i];
//		__syncthreads();
//		// do reduction in shared mem
//		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//			if (tid < s) {
//				sdata[tid] += sdata[tid + s];
//			}
//			__syncthreads();
//		}
//		// write result for this block to global mem
//		if (tid == 0) g_odata[blockIdx.x] = sdata[0];
//}

int main()
{
	int sumCPU = 0;
	int *a, *c; // host copies of a, c
	int *d_a, *d_c; // device copies of a, c
	size_t size = N * sizeof(int);

	// Alloc space for device copies of a, c
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_c, size);

	// Alloc space for host copies of a, c
	a = (int*)malloc(size);
	c = (int*)malloc(size);

	// Random input array init
	srand((time(NULL)));
	for (int i = 0; i < N; i++)
	{
		a[i] = rand() % 100000;
	}

	// Compute on CPU
	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (int i = 0; i < N; i++)
	{
		sumCPU += a[i];
	}
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	std::cout << "CPU time: " << time_span.count() << " seconds." << "\n";


	cudaEvent_t start, stop;
	// создание события для точки старта
	cudaEventCreate(&start);
	// создание события для точки завершения
	cudaEventCreate(&stop);
	// точка начала замера времени
	cudaEventRecord(start, 0);

	// Copy input to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

	//int blocks = N / THREADS_PER_BLOCK;
	//int toDo = 0;
	//int maxThreads = THREADS_PER_BLOCK; //threads per block
	//int threads = (N < maxThreads) ? N : maxThreads;
	//if (blocks > 1) toDo = 1 + blocks / maxThreads;
	//else toDo = 0;
	//for (int i = 0; i < toDo; i++) {
	//	threads = (blocks < maxThreads) ? blocks : maxThreads;
	//	blocks = blocks / threads;
	//	dim3 dimBlock(threads, 1, 1);
	//	dim3 dimGrid(blocks, 1, 1);
		// Launch reduceBase() kernel on GPU
	//	reduceBase << < dimGrid, dimBlock, THREADS_PER_BLOCK * sizeof(int) >> > (d_a, d_c);
		//reduceBase << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int) >> > (d_a, d_c);
	//}
	// Copy result back to host
	reduceBase << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int) >> > (d_a, d_c);
	cudaMemcpy(c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

	// точка завершения замера времени
	cudaEventRecord(stop, 0);
	// ожидание завершения выполнения задач на GPU
	cudaEventSynchronize(stop);
	float elapsedTime;
	// elapsedTime - затраченное время в миллисекундах
	cudaEventElapsedTime(&elapsedTime, start, stop);
		// уничтожение объектов событий
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("GPU time: %lf seconds\n", (double)(elapsedTime) / CLOCKS_PER_SEC);

	// Print results
	std::cout << "\nGPU result: " << c[0]+c[1] << "\n" << "CPU result: " << sumCPU << "\n";

	// Cleanup
	free(a); free(c);
	cudaFree(d_a); cudaFree(d_c);
	return 0;
}
