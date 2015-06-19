#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Kernel that executes on the CUDA device
//__global__ void square_array(float *a, int N)
//{
//  int idx = blockIdx.x * blockDim.x + threadIdx.x;
//  if (idx<N) a[idx] = a[idx] * a[idx];
//}

// main routine that executes on the host
int main(void) {

	float lambda = 20;
	int n = 10;
	int p = 2;
	int m = 10;
	int i, j;
	float *A_h[n]; // host A matrix pointers
	int *IDX_h[n]; // host Aindex matrix pointers
	float *A_d[n];
	int *IDX_d[n];

	for (i = 0; i < n; i++) {
		float *vec;
		vec = (float *) malloc(p * sizeof(float));
		int *vec_idx;
		vec_idx = (int *) malloc(p * sizeof(int));

		for (j = 0; j < p; j++) {
			int idx = (int) (m * (rand() / (RAND_MAX + 1.0)));
			float val = (float) rand() / RAND_MAX;
			vec[j] = val;
			vec_idx[j] = idx;
		}
		A_h[i] = &vec[0];
		IDX_h[i] = &vec_idx[0];

		float *vec_d;
		int *vec_idx_d;

		cudaMalloc((void **) &vec_d, p * sizeof(float));
		cudaMalloc((void **) &vec_idx_d, p * sizeof(int));

//		CUDA_SAFE_CALL(cudaMalloc((void **) &vec_d, p * sizeof(float))); // Allocate array on device
//		CUDA_SAFE_CALL(cudaMalloc((void **) &vec_idx_d, p * sizeof(int))); // Allocate array on device

		cudaMemcpy(vec_d, vec, sizeof(float) * p, cudaMemcpyHostToDevice);
		cudaMemcpy(vec_idx_d, vec_idx, sizeof(int) * p, cudaMemcpyHostToDevice);




	}

	puts("idem generovat x_0\n"); /* prints !!!Hello World!!! */
	float x_h[n];
	float L_h[n];
	float Li_h[n];
	for (i = 0; i < n; i++) {
		float val = (float) rand() / RAND_MAX;
		x_h[i] = val;
		L_h[i] = 0;

	}

	float gradient_h[m];
	for (i = 0; i < m; i++) {
		gradient_h[i] = 0;
	}
	//	size_t size = m * sizeof(float);
	//	gradient_h = (float *)malloc(size);
	for (i = 0; i < n; i++) {
		float *vector = A_h[i];
		int *vector_idx = IDX_h[i];
		for (j = 0; j < p; j++) {
			L_h[i] += vector[j] * vector[j];
			gradient_h[vector_idx[j]] = gradient_h[vector_idx[j]] + (x_h[i])
					* (vector[j]);
		}
	}

	for (i = 0; i < m; i++) {
		printf("gradient value:   %d %f\n", i, gradient_h[i]);
	}

	//vypocet Li
	for (i = 0; i < n; i++) {
		Li_h[i] = 1 / L_h[i];
		printf("Li:   %d %f  xi:  %f\n", i, Li_h[i], x_h[i]);

	}

	for (i = 0; i < n; i++) {
		printf("x[%d] =  %f  \n", i, x_h[i]);
	}

	for (i = 0; i < 10; i++) {
		int coordinate = (int) (n * (rand() / (RAND_MAX + 1.0)));
		//		printf("Zvolil som suradnicu: %d \n", coordinate);

		float *vector = A_h[coordinate];
		int *vector_idx = IDX_h[coordinate];
		float alpha = 0;
		for (j = 0; j < p; j++) {
			alpha += vector[vector_idx[j]] * gradient_h[vector_idx[j]];
		}
		//		printf("alpha: %f \n", alpha);
		//minimize  alfa*d + 0.5 * L*d^2 + lambda |x+d|
		double delta = -(alpha + lambda) * Li_h[coordinate];
		if (x_h[coordinate] + delta < 0) {
			delta = -(alpha - lambda) * Li_h[coordinate];
			if (x_h[coordinate] + delta > 0) {
				delta = -x_h[coordinate];
			}
		}

		for (j = 0; j < p; j++) {
			gradient_h[vector_idx[j]] += delta * vector[j];
		}

		x_h[coordinate] += delta;

		//
		//
		//		    g(idx)=g(idx)+delta*A(idx,j);
		//


	}

	for (i = 0; i < n; i++) {
		printf("x[%d] =  %f  \n", i, x_h[i]);
	}

	//	size_t matrixsize = n*p * sizeof(float);
	//
	//	h_A = (float *)malloc(matrixsize);        // Allocate array on host
	//	cudaMalloc((void **) &d_A, matrixsize);   // Allocate array on device


	// Initialize host array and copy it to CUDA device
	//  for (int i=0; i<N; i++) a_h[i] = (float)i;
	//  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
	// Do calculation on device:
	//  int block_size = 4;
	//  int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	//  square_array <<< n_blocks, block_size >>> (a_d, N);
	// Retrieve result from device and store it in host array
	//  cudaMemcpy(a_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
	// Print results
	//  for (int i=0; i<N; i++) printf("%d %f\n", i, a_h[i]);
	// Cleanup
	//	free(h_A);
	//	cudaFree(d_A);
	cudaFree(A_d);
	cudaFree(IDX_d);

}

