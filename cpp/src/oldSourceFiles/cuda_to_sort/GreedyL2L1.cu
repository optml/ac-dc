//ulimit -s unlimited
//nvcc Cuda.cu  -arch sm_20 && ./a.out


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
#include <cuda_runtime.h>

void checkCUDAError(const char *msg);


void main(){





}


void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}
