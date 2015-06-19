//ulimit -s unlimited
//nvcc -lcublas GreedyL2L1.cu

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
#include "cublas.h"
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

void modify(float *m, int ldm, int n, int p, int q, float alpha, float beta) {
	cublasSscal(n - p + 1, alpha, &m[IDX2F(p,q,ldm)], ldm);
	cublasSscal(ldm - p + 1, beta, &m[IDX2F(p,q,ldm)], 1);
}
#define M 6
#define N 5

void checkCUDAError(const char *msg);






int main() {
	int i, j;
	cublasStatus stat;
	float* devPtrA;
	float* a = 0;
	a = (float *) malloc(M * N * sizeof(*a));
	if (!a) {
		printf("host memory allocation failed");
		return EXIT_FAILURE;
	}
	for (j = 1; j <= N; j++) {
		for (i = 1; i <= M; i++) {
			a[IDX2F(i,j,M)] = (i - 1) * M + j;
		}
	}
	cublasInit();
	stat = cublasAlloc(M * N, sizeof(*a), (void**) &devPtrA);

	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("device memory allocation failed");
		return EXIT_FAILURE;
	}
	cublasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);

	for (j = 1; j <= N; j++) {
		for (i = 1; i <= M; i++) {
			printf("%7.0f", a[IDX2F(i,j,M)]);
		}
		printf("\n");
	}

//	modify(devPtrA, M, N, 2, 3, 16.0f, 12.0f);

	cublasSscal(M,10,&devPtrA[IDX2F(2,2,M)],2);


	cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
	cublasFree(devPtrA);
	cublasShutdown();
	for (j = 1; j <= N; j++) {
		for (i = 1; i <= M; i++) {
			printf("%7.0f", a[IDX2F(i,j,M)]);
		}
		printf("\n");
	}
	return EXIT_SUCCESS;

}

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}
