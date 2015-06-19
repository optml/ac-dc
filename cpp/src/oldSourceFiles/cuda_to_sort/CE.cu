//ulimit -s unlimited
//gcc -lm -std=c99 NRCDML1RegLog.c && ./a.out
//nvcc CE.cu  -arch sm_20 && ./a.out

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <cuda.h>
#include "device_functions.h"
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>

#include <cuda_runtime.h>
void checkCUDAError(const char *msg);

// Part 3 of 5: implement the kernel
__global__ void myFirstKernel(float *d_a) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	float tmp=0;
	for (i = 0; i < 1000; i++) {
		if (i > 500)
			tmp = tmp+ idx + i;
		else
			tmp =  idx + i;
	}
		d_a[idx]=tmp;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
void cudaTest1() {
	// pointer for host memory
	float *h_a;

	// pointer for device memory
	float *d_a;

	// define grid and block size
	int numBlocks = 10000 ;
	int numThreadsPerBlock = 1024 ;

	// Part 1 of 5: allocate host and device memory
	size_t memSize = numBlocks * numThreadsPerBlock * sizeof(float);
	h_a = (float *) malloc(memSize);
	cudaMalloc((float **) &d_a, memSize);

	// Part 2 of 5: launch kernel
	dim3 dimGrid( numBlocks);
	dim3 dimBlock( numThreadsPerBlock);
	clock_t t1, t2;
	t1 = clock();
	myFirstKernel<<< dimGrid, dimBlock >>>( d_a );
	cudaThreadSynchronize();
	t2 = clock();
	float diff = ((float) t2 - (float) t1) / 1000000.0F;
	checkCUDAError("kernel execution");
	printf("sorting:%f\n", diff);
	float pole[numBlocks * numThreadsPerBlock];
	t1 = clock();

	for (int i = 0; i < numBlocks * numThreadsPerBlock; i++) {
		float tmp=0;
		for (int j = 0; j < 1000; j++) {
			if (i > 500)
				tmp = tmp+  i + j;
			else
				tmp = i + j;
		}
		pole[i]=tmp;
	}

	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;

	printf("sorting:%f\n", diff);
	// block until the device has completed

	// check if kernel execution generated an error


	// Part 4 of 5: device to host copy
	cudaMemcpy(h_a, d_a, memSize, cudaMemcpyDeviceToHost);

	// Check for any CUDA errors
	checkCUDAError("cudaMemcpy");

	// Part 5 of 5: verify the data returned to the host is correct
	for (int i = 0; i < numBlocks; i++) {
		for (int j = 0; j < numThreadsPerBlock; j++) {
			assert(h_a[i * numThreadsPerBlock + j] == pole[i
					* numThreadsPerBlock + j]);
		}
	}

	// free device memory
	cudaFree(d_a);

	// free host memory
	free(h_a);

	// If the program makes it this far, then the results are correct and
	// there are no run-time errors.  Good work!
	printf("Correct!\n");

}

int main(void) {
	srand(1);
	printf("start solving\n");

	cudaTest1();

	printf("end solving\n");

}

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}
