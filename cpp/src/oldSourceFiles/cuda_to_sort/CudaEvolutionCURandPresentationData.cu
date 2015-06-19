//ulimit -s unlimited
//nvcc Cuda.cu  -arch sm_20 -o cuda.out && ./cuda.out

#define COLUMNLENGTH 2
#define NMAXITER 100
#define SerialExecution 1

//#define NMAXITERKernel  60000
//#define CUDACount  60000
//#define CUDACount 44642
//#define NMAXITERKernel 44642
#define CUDACount 90000
#define NMAXITERKernel 90000


//nvcc Cuda.cu  -arch sm_13 && ./a.out

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

//cudaEventRecord(start, 0);
//for (int i = 0; i < 2; ++i) {
//cudaMemcpyAsync(inputDev + i * size, inputHost + i * size,
//size, cudaMemcpyHostToDevice, stream[i]);
//MyKernel<<<100, 512, 0, stream[i]>>>
//(outputDev + i * size, inputDev + i * size, size);
//cudaMemcpyAsync(outputHost + i * size, outputDev + i * size,
//size, cudaMemcpyDeviceToHost, stream[i]);
//}
//cudaEventRecord(stop, 0);
//cudaEventSynchronize(stop);
//float elapsedTime;
//cudaEventElapsedTime(&elapsedTime, start, stop);
//They are destroyed this way:
//cudaEventDestroy(start);
//cudaEventDestroy(stop);


#include <cuda_runtime.h>

#define MAXFEATURES 47235000

#define MAXINSTANCES 100000000
// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);
/* an example of struct */
struct st_sortingByAbsWithIndex {
	double value;
	long int idx;
};

/* qsort struct comparision function (price double field) */
int struct_cmp_by_value(const void *a, const void *b) {
	struct st_sortingByAbsWithIndex *ia = (struct st_sortingByAbsWithIndex *) a;
	struct st_sortingByAbsWithIndex *ib = (struct st_sortingByAbsWithIndex *) b;
	double aa = ia->value;
	double bb = ib->value;
	if (aa * aa > bb * bb)
		return -1;
	else if (aa * aa < bb * bb)
		return 1;
	else
		return 0;

	/* double comparison: returns negative if b > a
	 and positive if a > b. We multiplied result by 100.0
	 to preserve decimal fraction */

}
float CS[] = { 0.062500, 0.125000, 0.250000, 0.500000, 1.000000, 2.000000,
		4.000000, 8.000000, 16.000000, 32.000000, 64.000000 };
int cscount = 11;

//#include <cuda.h>

// Kernel that executes on the CUDA device
//__global__ void square_array(float *a, int N)
//{
//  int idx = blockIdx.x * blockDim.x + threadIdx.x;
//  if (idx<N) a[idx] = a[idx] * a[idx];
//}

// main routine that executes on the host

//void RCDM(float** A, int IDX, float L, float b, float lambda, int n,
//		int N) {
//
//}
/* integer array printing function */
void print_float_array(const float *array, size_t len) {
	size_t i;

	printf("\n---------  \n");
	for (i = 0; i < len; i++)
		printf("%f | ", array[i]);
	putchar('\n');
	printf("\n---------  \n");
}
void print_int_array(const int *array, size_t len) {
	size_t i;

	for (i = 0; i < len; i++)
		printf("%d | ", array[i]);

	putchar('\n');
}

void generateRandomProblem(float ** A, int** R_Idx, int**C_Idx, int **C_Count,
		int n, int m, int ccount_min, int ccount_diff, int * nnzout) {
	*C_Idx = (int *) malloc(n * sizeof(int));
	*C_Count = (int *) malloc(n * sizeof(int));
	int i, j, idx, k;
	int nnz = 0;
	for (i = 0; i < n; i++) {
		(*C_Count)[i] = ((int) ((ccount_diff) * (rand() / (RAND_MAX + 1.0))))
				+ ccount_min;
		nnz += (*C_Count)[i];
	}
	(*nnzout) = nnz;
	*A = (float *) malloc(nnz * sizeof(float));
	*R_Idx = (int *) malloc(nnz * sizeof(int));
	nnz = 0;
	for (i = 0; i < n; i++) {
		(*C_Idx)[i] = nnz;
		for (j = 0; j < (*C_Count)[i]; j++) {
			int notfinished = 1;
			while (notfinished) {
				notfinished = 0;
				idx = ((int) ((m) * (rand() / (RAND_MAX + 1.0))));
				for (k = 0; k < j; k++) {
					if ((*R_Idx)[(*C_Idx)[i] + k] == idx) {
						notfinished = 1;
					}
				}
			}
			(*R_Idx)[nnz] = idx;
			(*A)[nnz] = (float) rand() / RAND_MAX;
			//			printf("A[%d][%d] = %f\n", nnz, (*R_Idx)[nnz], (*A)[nnz]);
			nnz++;
		}
	}
}

void generateNesterovProblem(float ** A, int** R_Idx, int**C_Idx,
		int **C_Count, float**b_out, float** xOpt, int n, int m,
		int ccount_min, int ccount_diff, int * nnzout) {
	int p = ccount_min;
	double A_h[n][p]; // host A matrix pointers
	printf("alokacia A done \n");
	long int IDX_h[n][p]; // host Aindex matrix pointers
	printf("alokacia I done \n");
	printf("alokacia x done \n");
	double optimalvalue = 0;
	int n_nonzero = 160000;
	double rho = 1;
	double sqrtofnonzeros = 400;
	int i, j, k;
	double tmp;
	printf("alokacia poli END\n");

	//Generovanie problemu-------------------------------------------------------------------
	for (i = 0; i < n; i++) {
		long int idx = 0;
		for (j = 0; j < p; j++) {
			int notfinished = 1;
			double val = (double) rand() / RAND_MAX;
			while (notfinished) {
				notfinished = 0;
				idx = ((long int) ((m) * (rand() / (RAND_MAX + 1.0))));
				for (k = 0; k < j; k++) {
					if (IDX_h[i][k] == idx) {
						notfinished = 1;
					}
				}
			}
			A_h[i][j] = 2 * val - 1;
			IDX_h[i][j] = idx;
		}
	}
	printf("Matrix B Generated\n");

	double* y;
	y = (double*) calloc(m, sizeof(double));
	tmp = 0;
	for (j = 0; j < m; j++) {
		y[j] = (double) rand() / RAND_MAX;
		tmp += y[j] * y[j];
	}
	for (j = 0; j < m; j++) {
		y[j] = y[j] / tmp;
	}
	printf("vector y Generated\n");

	struct st_sortingByAbsWithIndex* dataToSort;
	dataToSort = (struct st_sortingByAbsWithIndex*) calloc(m,
			sizeof(struct st_sortingByAbsWithIndex));

	for (i = 0; i < n; i++) {
		dataToSort[i].idx = i;
		dataToSort[i].value = 0;
	}
	printf("Struc created\n");
	for (i = 0; i < n; i++) {
		tmp = 0;
		for (j = 0; j < p; j++) {
			tmp += y[IDX_h[i][j]] * A_h[i][j];
		}
		dataToSort[i].value = tmp;
	}

	//Sorting B
	printf("SORTING START\n");

	size_t structs_len = sizeof(dataToSort)
			/ sizeof(struct st_sortingByAbsWithIndex);
	printf("SORTING 2\n");
	qsort(dataToSort, structs_len, sizeof(struct st_sortingByAbsWithIndex),
			struct_cmp_by_value);
	printf("SORTING END\n");
	//	return 1;
	double* x;
	x = (double*) calloc(n, sizeof(double));
	for (i = 0; i < n; i++) { // vytvaranie matice A
		int idx = dataToSort[i].idx;
		double alpha = 1;
		x[idx] = 0;
		if (i < n_nonzero) {
			alpha = (double) abs(1 / dataToSort[idx].value);
			x[idx] = ((double) rand() / RAND_MAX) * rho / (sqrtofnonzeros);
			if (dataToSort[idx].value < 0) {
				x[idx] = -x[idx];
			}
		} else if (dataToSort[idx].value > 0.1 || dataToSort[idx].value < -0.1) {
			alpha = (double) abs(1 / dataToSort[idx].value) * (double) rand()
					/ RAND_MAX;
		}
		for (j = 0; j < p; j++) {
			A_h[idx][j] = A_h[idx][j] * alpha;
		}
	}
	//	print_double_array(&L[0],n);
	//	print_double_array(&Li[0], 10);
	free(dataToSort);
	// Compute Li
	double* Li; // Lipschitz constants
	Li = (double*) calloc(n, sizeof(double));
	for (i = 0; i < n; i++) {
		Li[i] = 0;
		for (j = 0; j < p; j++) {
			Li[i] += A_h[i][j] * A_h[i][j];
		}
		Li[i] = 1 / Li[i];
	}
	// END compute Li

	for (i = 0; i < m; i++) {
		optimalvalue += y[i] * y[i];
	}
	optimalvalue = optimalvalue * 0.5;

	double* b;
	b = y;
	for (i = 0; i < n; i++) {
		for (j = 0; j < p; j++) {
			b[IDX_h[i][j]] += x[i] * A_h[i][j];
		}
	}
	for (i = 0; i < n; i++) {
		//		printf("optval %1.16f \n", optimalvalue);
		if (x[i] > 0)
			optimalvalue += x[i];
		else
			optimalvalue -= x[i];
	}
	printf("optval %1.16f \n", optimalvalue);

	// write to output
	*xOpt = (float *) malloc(n * sizeof(float));
	*b_out = (float *) malloc(m * sizeof(float));
	for (i = 0; i < n; i++) {
		(*xOpt)[i] = x[i];
	}
	for (j = 0; j < m; j++) {
		(*b_out)[j] = b[j];
	}

	*C_Idx = (int *) malloc(n * sizeof(int));
	*C_Count = (int *) malloc(n * sizeof(int));
	int nnz = 0;
	for (i = 0; i < n; i++) {
		(*C_Count)[i] = p;
		nnz += (*C_Count)[i];
	}
	(*nnzout) = nnz;
	*A = (float *) malloc(nnz * sizeof(float));
	*R_Idx = (int *) malloc(nnz * sizeof(int));
	nnz = 0;
	for (i = 0; i < n; i++) {
		(*C_Idx)[i] = nnz;
		for (j = 0; j < (*C_Count)[i]; j++) {
			(*R_Idx)[nnz] = IDX_h[i][j];
			(*A)[nnz] = A_h[i][j];
			//			printf("A[%d][%d] = %f\n", nnz, (*R_Idx)[nnz], (*A)[nnz]);
			nnz++;
		}
	}
}

void printMatrixA(float *A, int*R_Idx, int*C_Idx, int*C_Count, int n, int nnz) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count[i]; j++) {
			printf("A[%d][%d] = %f \n", i, R_Idx[C_Idx[i] + j], A[C_Idx[i] + j]);
		}
	}
}

void matrix_vector_product(float *A, int*R_Idx, int*C_Idx, int*C_Count, int n,
		int m, int nnz, float *x, float* b, float c) {
	int i, j;
	for (j = 0; j < m; j++)
		b[j] = 0;
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count[i]; j++) {
			b[R_Idx[C_Idx[i] + j]] += c * A[C_Idx[i] + j] * x[i];
		}
	}
}

__global__ void setupKernel(curandState *state, unsigned long seed) {
	//	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int id = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	curand_init((seed << 20) + id, id, 0, &state[id]);
}

__global__ void randKernel(curandState *state) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState localState = state[id];
	curand_uniform(&localState); //dummy call, just to be here
	state[id] = localState;
}

__global__ void RCDMKernel(float *A, int*R_Idx, int*C_Idx, int*C_Count, int* n,
		int* m, int* nnz, float* b, float*residuals, float*x, float * lambda,
		float* Li, int* NMAX, curandState* cstate) {
	int j, i, k;
	float partialDetivative, delta, tmp;
	int id = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	//	int id = (blockDim.x + blockIdx.x * blockDim.x + threadIdx.x);
	curandState localState = cstate[id];

	for (k = 0; k < NMAXITERKernel ; k++) {
		double d = curand_uniform_double(&localState);
		//		double u1 = curand_uniform_double(localState);
		int idx = (int) (d * n[0]);
		// LOAD A, R, residuals
		float ALocal[COLUMNLENGTH];
		int RIDX[COLUMNLENGTH];
		float* residualsAddress[COLUMNLENGTH];
		float xLocal = x[idx];
		float LiLocal = Li[idx];
		int cidx = C_Idx[idx];
		partialDetivative = 0;
		for (i = 0; i < COLUMNLENGTH; i++) {
			j = cidx + i;
			ALocal[i] = A[j];
			RIDX[i] = R_Idx[j];
			residualsAddress[i] = &residuals[RIDX[i]];
			partialDetivative += ALocal[i] * residuals[RIDX[i]];
			//			partialDetivative += A[j] * residuals[R_Idx[j]];
		}

		tmp = LiLocal * (partialDetivative + lambda[0]);
		if (xLocal > tmp) {
			delta = -tmp;
		} else {
			tmp = LiLocal * (partialDetivative - lambda[0]);
			if (xLocal < tmp) {
				delta = -tmp;
			} else {
				delta = -xLocal;
			}
		}
		atomicAdd(&x[idx], delta);
		//				atomicAdd(&x[idx], 1);
		for (i = 0; i < COLUMNLENGTH; i++) {

			atomicAdd(residualsAddress[i], ALocal[i] * delta);
		}

	}
	cstate[id] = localState;
}

__global__ void MersenRandomKernel(float *A, int*R_Idx, int*C_Idx, int*C_Count,
		int* n, int* m, int* nnz, float*residuals, float*x, float * lambda,
		float* Li, int* NMAX) {
	unsigned int j, i, k;
	float partialDetivative, delta, tmp;
	int const a = 16807;
	int const mersen = 2147483647; //ie 2**31-1
	long seed = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);

	for (k = 0; k < NMAXITER; k++) {
		seed = (long(seed * a))%mersen;
		long idx = seed % n[0];
		//		temp = seed * a;
		//		seed = (int) (temp - mersen * floor(temp * reciprocal_m));
		//		int idx = ratioNMersen * seed;
		//


		float ALocal[COLUMNLENGTH];
		int RIDX[COLUMNLENGTH];
		float* residualsAddress[COLUMNLENGTH];
		float xLocal = x[idx];
		float LiLocal = Li[idx];
		int cidx = C_Idx[idx];
		partialDetivative = 0;
		for (i = 0; i < COLUMNLENGTH; i++) {
			j = cidx + i;
			ALocal[i] = A[j];
			RIDX[i] = R_Idx[j];
			residualsAddress[i] = &residuals[RIDX[i]];
			partialDetivative += ALocal[i] * residuals[RIDX[i]];
			//			partialDetivative += A[j] * residuals[R_Idx[j]];
		}

		tmp = LiLocal * (partialDetivative + lambda[0]);
		if (xLocal > tmp) {
			delta = -tmp;
		} else {
			tmp = LiLocal * (partialDetivative - lambda[0]);
			if (xLocal < tmp) {
				delta = -tmp;
			} else {
				delta = -xLocal;
			}
		}
		//		atomicAdd(&x[idx], delta);
		atomicAdd(&x[idx], 1);

		for (i = 0; i < COLUMNLENGTH; i++) {

			atomicAdd(residualsAddress[i], ALocal[i] * delta);
		}

	}
}

__global__ void MersenRandomKernelDoubleGenerator(float *A, int*R_Idx,
		int*C_Idx, int*C_Count, int* n, int* m, int* nnz, float*residuals,
		float*x, float * lambda, float* Li, int* NMAX) {
	unsigned int j, i, k;
	float partialDetivative, delta, tmp;
	double const a = 16807;
	double const mersen = 2147483647; //ie 2**31-1
	double const reciprocal_m = 1.0 / mersen;
	//		double const ratioNMersen = n[0]*reciprocal_m;
	long seed = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	double temp;

	for (k = 0; k < NMAXITER; k++) {
		temp = seed * a;
		seed = (int) (temp - mersen * floor(temp * reciprocal_m));
		//		int idx = ratioNMersen * seed;
		int idx = n[0] * reciprocal_m * seed;

		float ALocal[COLUMNLENGTH];
		int RIDX[COLUMNLENGTH];
		float* residualsAddress[COLUMNLENGTH];
		float xLocal = x[idx];
		float LiLocal = Li[idx];
		int cidx = C_Idx[idx];
		partialDetivative = 0;
		for (i = 0; i < COLUMNLENGTH; i++) {
			j = cidx + i;
			ALocal[i] = A[j];
			RIDX[i] = R_Idx[j];
			residualsAddress[i] = &residuals[RIDX[i]];
			partialDetivative += ALocal[i] * residuals[RIDX[i]];
			//			partialDetivative += A[j] * residuals[R_Idx[j]];
		}

		tmp = LiLocal * (partialDetivative + lambda[0]);
		if (xLocal > tmp) {
			delta = -tmp;
		} else {
			tmp = LiLocal * (partialDetivative - lambda[0]);
			if (xLocal < tmp) {
				delta = -tmp;
			} else {
				delta = -xLocal;
			}
		}
		atomicAdd(&x[idx], delta);
		//		atomicAdd(&x[idx], 1);

		for (i = 0; i < COLUMNLENGTH; i++) {

			atomicAdd(residualsAddress[i], ALocal[i] * delta);
		}

	}
}
__global__ void MersenRandomKernelDoubleGeneratorSingle(float *A, int*R_Idx,
		int*C_Idx, int*C_Count, int* n, int* m, int* nnz, float*residuals,
		float*x, float * lambda, float* Li, int* NMAX) {
	unsigned int j, i;
	float partialDetivative, delta, tmp;
	double const a = 16807;
	double const mersen = 2147483647; //ie 2**31-1
	double const reciprocal_m = 1.0 / mersen;
	//		double const ratioNMersen = n[0]*reciprocal_m;
	long seed = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	double temp;

	int k;
	for (k = 0; k < CUDACount * 100; k++) {
		temp = seed * a;
		seed = (int) (temp - mersen * floor(temp * reciprocal_m));
		//		int idx = ratioNMersen * seed;
		int idx = n[0] * reciprocal_m * seed;

		float ALocal[COLUMNLENGTH];
		int RIDX[COLUMNLENGTH];
		float* residualsAddress[COLUMNLENGTH];
		float xLocal = x[idx];
		float LiLocal = Li[idx];
		int cidx = C_Idx[idx];
		partialDetivative = 0;
		for (i = 0; i < COLUMNLENGTH; i++) {
			j = cidx + i;
			ALocal[i] = A[j];
			RIDX[i] = R_Idx[j];
			residualsAddress[i] = &residuals[RIDX[i]];
			partialDetivative += ALocal[i] * residuals[RIDX[i]];
			//			partialDetivative += A[j] * residuals[R_Idx[j]];
		}

		tmp = LiLocal * (partialDetivative + lambda[0]);
		if (xLocal > tmp) {
			delta = -tmp;
		} else {
			tmp = LiLocal * (partialDetivative - lambda[0]);
			if (xLocal < tmp) {
				delta = -tmp;
			} else {
				delta = -xLocal;
			}
		}
		atomicAdd(&x[idx], delta);
		//				atomicAdd(&x[idx], 1);

		for (i = 0; i < COLUMNLENGTH; i++) {

			atomicAdd(residualsAddress[i], ALocal[i] * delta);
		}
	}
}
__global__ void MersenRandomKernelDoubleGeneratorLessLocalMemmory(float *A,
		int*R_Idx, int*C_Idx, int*C_Count, int* n, int* m, int* nnz,
		float*residuals, float*x, float * lambda, float* Li, int* NMAX) {
	unsigned int j, i, k;
	float partialDetivative, delta, tmp;
	double const a = 16807;
	double const mersen = 2147483647; //ie 2**31-1
	double const reciprocal_m = 1.0 / mersen;
	//		double const ratioNMersen = n[0]*reciprocal_m;
	long seed = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	double temp;

	for (k = 0; k < NMAXITER; k++) {
		temp = seed * a;
		seed = (int) (temp - mersen * floor(temp * reciprocal_m));
		int idx = n[0] * reciprocal_m * seed;

		int cidx = C_Idx[idx];
		partialDetivative = 0;
		for (i = 0; i < 10; i++) {
			j = cidx + i;
			partialDetivative += A[j] * residuals[R_Idx[j]];
		}
		tmp = Li[idx] * (partialDetivative + lambda[0]);
		if (x[idx] > tmp) {
			delta = -tmp;
		} else {
			tmp = Li[idx] * (partialDetivative - lambda[0]);
			if (x[idx] < tmp) {
				delta = -tmp;
			} else {
				delta = -x[idx];
			}
		}
		atomicAdd(&x[idx], delta);
		for (j = C_Idx[idx]; j < C_Idx[idx] + C_Count[idx]; j++) {
			atomicAdd(&residuals[R_Idx[j]], A[j] * delta);
		}

	}
}

__global__ void myFirstKernel(float *x) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//	x[idx] = (float) (-1 * threadIdx.x);
	x[idx] = (float) (1000 * blockIdx.x + threadIdx.x);
}

__global__ void resetXKernel(float *x) {
	int id = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	x[id] = 0;
}

__global__ void myFirstfloatKernel(float *d_a) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	d_a[idx] = (float) (1000 * blockIdx.x + threadIdx.x);
}

__global__ void setup_kernel(curandState *state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	/* Each thread gets different seed, a different sequence number, no offset */
	curand_init(i, i, 0, &state[i]);
}

float NRCDM_SR(float *A, int*R_Idx, int*C_Idx, int*C_Count, int n, int m,
		int nnz, float* b, float*x, float lambda, float* Li, int NMAX,
		float optimalvalue, int logging) {
	float residuals[m];
	float value = 0;
	int i, j, N;

	FILE *fp;
	//	fp = fopen("/exports/home/s1052689/nesterov.txt", "w");
	fp = fopen("/tmp/sparseregression.csv", "w");
	// calculate residuals
	for (j = 0; j < m; j++)
		residuals[j] = -b[j];
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count[i]; j++) {
			residuals[R_Idx[C_Idx[i] + j]] += A[C_Idx[i] + j] * x[i];
		}
	}
	float partialDetivative, delta, tmp;
	// iteration counter
	for (N = 0; N < NMAX * n; N++) {
		int idx = (int) (n * (rand() / (RAND_MAX + 1.0)));
		partialDetivative = 0;
		for (j = C_Idx[idx]; j < C_Idx[idx] + C_Count[idx]; j++) {
			partialDetivative += A[j] * residuals[R_Idx[j]];
		}

		tmp = Li[idx] * (partialDetivative + lambda);
		if (x[idx] > tmp) {
			delta = -tmp;
		} else {
			tmp = Li[idx] * (partialDetivative - lambda);
			if (x[idx] < tmp) {
				delta = -tmp;
			} else {
				delta = -x[idx];
			}
		}
		x[idx] += delta;
		for (j = C_Idx[idx]; j < C_Idx[idx] + C_Count[idx]; j++) {
			residuals[R_Idx[j]] += A[j] * delta;
		}
		if (N % (n / 100) == 0) {
			int nnzcount = 0;
			value = 0;
			for (i = 0; i < n; i++) {
				if (x[i] != 0)
					nnzcount++;
				if (x[i] > 0)
					value += x[i];
				else
					value -= x[i];
			}
			for (j = 0; j < m; j++)
				value += 0.5 * residuals[j] * residuals[j];
			fprintf(fp, "Iteracia:%d, value:%f, nnz:%d, epsilon: %f\n", N,
					value, nnzcount, value - optimalvalue);
		}

	}
	fclose(fp);
	return value;
}

double ComputeObjectiveValue(float *A, int*R_Idx, int*C_Idx, int*C_Count,
		int n, int m, int nnz, float* b, float*x, float lambda) {
	double residuals[m];
	double value = 0;
	int i, j;

	for (j = 0; j < m; j++)
		residuals[j] = -b[j];
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count[i]; j++) {
			residuals[R_Idx[C_Idx[i] + j]] += A[C_Idx[i] + j] * x[i];
		}
	}

	value = 0;
	for (i = 0; i < n; i++) {
		if (x[i] > 0)
			value += x[i];
		else
			value -= x[i];
	}
	value = value * lambda;
	//	printf("value%f\n",value);
	double residualsVal = 0;
	for (j = 0; j < m; j++)
		residualsVal += 0.5 * residuals[j] * residuals[j];
	//	printf("res%f\n",residualsVal);
	value = value + residualsVal;
	//	printf("value%f\n",value);
	return value;
}

float NRCDM_SR_TIMING(float *A, int*R_Idx, int*C_Idx, int*C_Count, int n,
		int m, int nnz, float* b, float*x, float lambda, float* Li, int NMAX,
		float optimalvalue, int logging, float* residuals) {
	float value = 0;
	long j, N, k;
	float partialDetivative, delta, tmp;
	// iteration counter
	//	printf("NMAX:%d, n*NMAX:%d\n", NMAX, NMAX * n);
	for (N = 0; N < NMAX; N++) {
		for (k = 0; k < n; k++) {
			int idx = (int) (n * (rand() / (RAND_MAX + 1.0)));
			partialDetivative = 0;
			for (j = C_Idx[idx]; j < C_Idx[idx] + C_Count[idx]; j++) {
				partialDetivative += A[j] * residuals[R_Idx[j]];
			}

			tmp = Li[idx] * (partialDetivative + lambda);
			if (x[idx] > tmp) {
				delta = -tmp;
			} else {
				tmp = Li[idx] * (partialDetivative - lambda);
				if (x[idx] < tmp) {
					delta = -tmp;
				} else {
					delta = -x[idx];
				}
			}
			x[idx] += delta;
			for (j = C_Idx[idx]; j < C_Idx[idx] + C_Count[idx]; j++) {
				residuals[R_Idx[j]] += A[j] * delta;
			}
		}
		//		printf("k=%d\n", N);
	}
	return value;
}

void computeLipsitzConstantsForSparseRegression(float** L, float ** Li,
		float *A, int*R_Idx, int*C_Idx, int*C_Count, int n, int nnz) {
	int i, j;
	*L = (float *) malloc(n * sizeof(float));
	*Li = (float *) malloc(n * sizeof(float));
	for (i = 0; i < n; i++) {
		(*L)[i] = 0;
		for (j = 0; j < C_Count[i]; j++) {
			(*L)[i] += A[C_Idx[i] + j] * A[C_Idx[i] + j];
		}
		(*Li)[i] = 1 / (*L)[i];
	}
}

void cudaSolver() {
	FILE *fp;
	//		fp = fopen("/exports/home/s1052689/nesterov2.txt", "w");
	fp = fopen("/tmp/taki_cuda.txt", "w");
	clock_t t1, t2;
	int NMax_d_h[1];
	int pMin = COLUMNLENGTH;
	int pMax = COLUMNLENGTH;
	float lambda = 1;
	float diff;
	float* A_h;
	int * C_Idx_h;
	int * R_Idx_h;
	int * C_Count_h;
	int nnz, i, j;

	printf("CudaSolver\n");

	curandState *devStates;
	int totalThreads = 256;
	int dim1 = 1024;
	int dim2 = 2;
	int n = totalThreads * dim1 * dim2;
	int m = n / 4;
	NMax_d_h[0] = 1;

	printf("CudaSolver2\n");

	/* Allocate space for prng states on device */
	printf("idem alokovat random states");
	dim3 dimGrid( dim1, dim2);
	dim3 dimBlock( totalThreads);
	dim3 dimGridRCDM( dim1, dim2);
	t1 = clock();
	cudaMalloc((void **) &devStates, dim1 * dim2 * totalThreads
			* sizeof(curandState));
	checkCUDAError("Alloc error");
	/* Setup prng states */
	setup_kernel<<< dimGrid, dimBlock >>>(devStates);
	cudaThreadSynchronize();
	checkCUDAError("Inicializacia ranom states");
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Inicializacia ranom states: %f\n", diff);
	printf("Idem generaovt random problem");
	t1 = clock();
	generateRandomProblem(&A_h, &R_Idx_h, &C_Idx_h, &C_Count_h, n, m, pMin,
			pMax - pMin, &nnz);
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Random problem generation: %f\n", diff);
	printf("Random problem generated\n");
	float* L;
	float* Li;
	computeLipsitzConstantsForSparseRegression(&L, &Li, A_h, R_Idx_h, C_Idx_h,
			C_Count_h, n, nnz);
	printf("Lipshitz constants computed\n");
	//		printMatrixA(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, nnz);
	//	print_float_array(&L[0], n);
	float b[m];
	float xOpt[n];
	float x[n];
	for (j = 0; j < n; j++) {
		xOpt[j] = 2 * ((float) rand() / RAND_MAX) - 1;
	}
	printf("Optimal solution generated\n");
	// set b = A*x
	matrix_vector_product(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, xOpt, b,
			1);
	printf("matrix vector product multiplied\n");
	//	print_float_array(&xOpt[0], n);
	//	print_float_array(&b[0], m);
	float value = 0;
	for (i = 0; i < n; i++)
		x[i] = 0;
	printf("Idem inicializovat reziduals\n");
	float residuals[m];
	for (j = 0; j < m; j++)
		residuals[j] = -b[j];
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count_h[i]; j++) {
			residuals[R_Idx_h[C_Idx_h[i] + j]] += A_h[C_Idx_h[i] + j] * x[i];
		}
	}

	float* A_d;
	float* Li_d;
	int * C_Idx_d;
	int * R_Idx_d;
	int * C_Count_d;
	int *n_d;
	int *nnz_d;
	int *m_d;
	float* x_d;
	float* b_d;
	float* lambda_d;
	float * residuals_d;
	int * NMax_d;
	//	size_t memSize = ;


	t1 = clock();
	printf("Idem alokovat data na device\n");
	cudaMalloc((void**) &A_d, nnz * sizeof(float));
	cudaMalloc((void**) &R_Idx_d, nnz * sizeof(int));
	cudaMalloc((void**) &C_Idx_d, n * sizeof(int));
	cudaMalloc((void**) &Li_d, n * sizeof(float));
	cudaMalloc((void**) &C_Count_d, n * sizeof(int));
	cudaMalloc((void**) &n_d, 1 * sizeof(int));
	cudaMalloc((void**) &nnz_d, 1 * sizeof(int));
	cudaMalloc((void**) &m_d, 1 * sizeof(int));
	cudaMalloc((void**) &x_d, n * sizeof(float));
	cudaMalloc((void**) &b_d, m * sizeof(float));
	cudaMalloc((void**) &residuals_d, m * sizeof(float));
	cudaMalloc((void**) &lambda_d, 1 * sizeof(float));
	cudaMalloc((void**) &NMax_d, 1 * sizeof(int));

	cudaThreadSynchronize();
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Data alocated %d, %d from host -> device:%f\n", nnz, n, diff);

	// Part 2 of 5: host to device memory copy
	t1 = clock();
	cudaMemcpy(A_d, A_h, nnz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C_Count_d, C_Count_h, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(R_Idx_d, R_Idx_h, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Li_d, Li, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(C_Idx_d, C_Idx_h, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(n_d, &n, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nnz_d, &nnz, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m_d, &m, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(NMax_d, &NMax_d_h, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, m * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(residuals_d, residuals, m * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(lambda_d, &lambda, 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Copy data %d, %d from host -> device:%f\n", nnz, n, diff);

	printf("Idem spustat paralelny RCDM\n");
	cudaThreadSynchronize();
	t1 = clock();

	cudaEvent_t start, stop;
	float time, timeSerial;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	RCDMKernel<<< dimGridRCDM, dimBlock >>>(A_d, R_Idx_d, C_Idx_d, C_Count_d,
			n_d, m_d, nnz_d, b_d, residuals_d,x_d, lambda_d, Li_d, NMax_d,devStates);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	checkCUDAError("kernel execution");
	printf("Paralel NRCDM CUDA TIMMER:  %f ms\n", time);

	printf("Reset X");
	//	cudaFree(x_d);
	//	cudaFree(residuals_d);
	//	cudaMalloc((void**) &x_d, n * sizeof(float));
	//	cudaMalloc((void**) &residuals_d, m * sizeof(float));
	cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(residuals_d, residuals, m * sizeof(float),
			cudaMemcpyHostToDevice);
	//
	checkCUDAError("resetXKernel execution");
	printf("Reset X finished");
	//
	printf("Mersen Kernel");
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//	MersenRandomKernel<<< dimGridRCDM, dimBlock >>>(A_d, R_Idx_d, C_Idx_d, C_Count_d,
	//			n_d, m_d, nnz_d, b_d, residuals_d,x_d, lambda_d, Li_d, NMax_d);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	checkCUDAError("kernel execution");
	printf("Paralel MERSEN NRCDM CUDA TIMMER:  %f ms\n", time);

	//	checkCUDAError("kernel execution");
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	NRCDM_SR(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, b, x, lambda, Li,
			NMAXITER / NMAXITER, value, 1);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeSerial, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	checkCUDAError("kernel execution");
	printf("SERIAL NRCDM CUDA TIMMER:  %f ms, speed up:%f\n", timeSerial,
			timeSerial / time);

	//------------------------- LOAD RESULTS FROM CUDA AND COMPARE
	float* xcuda;
	xcuda = (float*) malloc(n * sizeof(float));
	cudaMemcpy(xcuda, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy");
	cudaThreadSynchronize();
	checkCUDAError("cudaMemcpy");
	for (i = 0; i < 20; i++) {
		printf("x_h[%d]=%f, x_d[%d]=%f\n", i, x[i], i, xcuda[i]);
	}

	float normErrorSquered = 0;
	float normX = 0;
	for (i = 0; i < n; i++) {
		normErrorSquered += (x[i] - xcuda[i]) * (x[i] - xcuda[i]);
		normX += x[i] * x[i];
	}
	for (i = 0; i < n; i++) {
		if (xcuda[i] * xcuda[i] < normX) {
			;
		} else {
			printf("x_h[%d]=%f, x_d[%d]=%f\n", i, x[i], i, xcuda[i]);
		}
	}
	printf("normaChyby=%f;   norm(x)=%f,   n=%d, m=%d\n", normErrorSquered,
			normX, n, m);

	//	int minE=100000000;
	//	int maxE=0;
	//	for (i = 0; i < n; i++) {
	//		if (xcuda[i]<minE) minE=xcuda[i];
	//		if (xcuda[i]>minE) maxE=xcuda[i];
	//	}
	//	printf("minE=%d, maxE=%d\n", minE,maxE);

	//------------------------- CLEAN UP
	printf("Idem uvolnovat\n");
	cudaFree(A_d);
	cudaFree(Li_d);
	cudaFree(R_Idx_d);
	cudaFree(C_Idx_d);
	cudaFree(C_Count_d);

	cudaFree(n_d);
	cudaFree(nnz_d);
	cudaFree(m_d);
	cudaFree(x_d);
	cudaFree(b_d);
	cudaFree(lambda_d);
	cudaFree(residuals_d);
	cudaFree(devStates);
	fclose(fp);
}
__global__ void myDoNothingKernel() {
	//	int idx =   threadIdx.x;

}
void cudaMersenSolver() {
	int totalThreads = 256;
	//	int dim1 = 1024;
	//	int dim2 = 128;
	int dim1 = 1000;
	int dim2 = 160;
	int n = totalThreads * dim1 * dim2;
	//	dim2=dim2/10;
	int m = n / 4;

	FILE *fp;
	//		fp = fopen("/exports/home/s1052689/nesterov2.txt", "w");
	fp = fopen("/tmp/taki_cuda.txt", "w");
	clock_t t1, t2;
	int NMax_d_h[1];
	int pMin = COLUMNLENGTH;
	int pMax = COLUMNLENGTH;
	float lambda = 1;
	float diff;
	float* A_h;
	int * C_Idx_h;
	int * R_Idx_h;
	int * C_Count_h;
	int nnz, i, j;

	printf("CudaSolver\n");

	NMax_d_h[0] = 1;

	/* Allocate space for prng states on device */
	dim3 dimBlock( totalThreads);
	dim3 dimGridRCDM( dim1, dim2);
	t1 = clock();
	/* Setup prng states */
	t1 = clock();
	generateRandomProblem(&A_h, &R_Idx_h, &C_Idx_h, &C_Count_h, n, m, pMin,
			pMax - pMin, &nnz);
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("NNZ: %d\n", nnz);
	printf("Random problem generation: %f\n", diff);
	printf("Random problem generated\n");
	float* L;
	float* Li;
	computeLipsitzConstantsForSparseRegression(&L, &Li, A_h, R_Idx_h, C_Idx_h,
			C_Count_h, n, nnz);
	printf("Lipshitz constants computed\n");
	//		printMatrixA(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, nnz);
	//	print_float_array(&L[0], n);
	float b[m];
	float xOpt[n];
	float x[n];
	for (j = 0; j < n; j++) {
		xOpt[j] = 2 * ((float) rand() / RAND_MAX) - 1;
	}
	printf("Optimal solution generated\n");
	// set b = A*x
	matrix_vector_product(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, xOpt, b,
			1);
	printf("matrix vector product multiplied\n");
	//	print_float_array(&xOpt[0], n);
	//	print_float_array(&b[0], m);
	float value = 0;
	for (i = 0; i < n; i++)
		x[i] = 0;
	printf("Idem inicializovat reziduals\n");
	float residuals[m];
	for (j = 0; j < m; j++)
		residuals[j] = -b[j];
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count_h[i]; j++) {
			residuals[R_Idx_h[C_Idx_h[i] + j]] += A_h[C_Idx_h[i] + j] * x[i];
		}
	}

	float* A_d;
	float* Li_d;
	int * C_Idx_d;
	int * R_Idx_d;
	int * C_Count_d;
	int *n_d;
	int *nnz_d;
	int *m_d;
	float* x_d;
	float* lambda_d;
	float * residuals_d;
	int * NMax_d;
	//	size_t memSize = ;


	t1 = clock();
	printf("Idem alokovat data na device\n");
	cudaMalloc((void**) &A_d, nnz * sizeof(float));
	checkCUDAError("kernel execution Alloc A_d");
	cudaMalloc((void**) &R_Idx_d, nnz * sizeof(int));
	checkCUDAError("kernel execution Alloc R_IDX");
	cudaMalloc((void**) &C_Idx_d, n * sizeof(int));
	checkCUDAError("kernel execution Alloc C_IDX");
	cudaMalloc((void**) &Li_d, n * sizeof(float));
	checkCUDAError("kernel execution Alloc Li");
	cudaMalloc((void**) &C_Count_d, n * sizeof(int));
	checkCUDAError("kernel execution Alloc CCount");
	cudaMalloc((void**) &n_d, 1 * sizeof(int));
	cudaMalloc((void**) &nnz_d, 1 * sizeof(int));
	cudaMalloc((void**) &m_d, 1 * sizeof(int));
	cudaMalloc((void**) &x_d, n * sizeof(float));
	cudaMalloc((void**) &residuals_d, m * sizeof(float));
	checkCUDAError("kernel execution Alloc Residuals");
	cudaMalloc((void**) &lambda_d, 1 * sizeof(float));
	cudaMalloc((void**) &NMax_d, 1 * sizeof(int));
	checkCUDAError("kernel execution Alloc N_d");
	cudaThreadSynchronize();
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Data alocated %d, %d from host -> device:%f\n", nnz, n, diff);

	// Part 2 of 5: host to device memory copy
	t1 = clock();
	cudaMemcpy(A_d, A_h, nnz * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy A_d");
	cudaMemcpy(C_Count_d, C_Count_h, n * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy C_Cound_d");
	cudaMemcpy(R_Idx_d, R_Idx_h, nnz * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy R");
	cudaMemcpy(Li_d, Li, n * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy Li");
	cudaMemcpy(C_Idx_d, C_Idx_h, n * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy C_ID");
	cudaMemcpy(n_d, &n, 1 * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy Nd");
	cudaMemcpy(nnz_d, &nnz, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m_d, &m, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(NMax_d, &NMax_d_h, 1 * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy NMax");
	cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy x_d");
	cudaMemcpy(residuals_d, residuals, m * sizeof(float),
			cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy Resiaduals");
	cudaMemcpy(lambda_d, &lambda, 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Copy data %d, %d from host -> device:%f\n", nnz, n, diff);

	printf("Idem spustat paralelny RCDM\n");
	cudaThreadSynchronize();
	t1 = clock();
	checkCUDAError("kernel execution Paralel MERSEN NRCDM CUDA - BEFORE");
	cudaEvent_t start, stop;
	float time, timeSerial;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	MersenRandomKernelDoubleGenerator<<< dimGridRCDM, dimBlock >>>(A_d, R_Idx_d, C_Idx_d, C_Count_d,
			n_d, m_d, nnz_d, residuals_d,x_d, lambda_d, Li_d, NMax_d);

	//	MersenRandomKernel<<< dimGridRCDM, dimBlock >>>(A_d, R_Idx_d, C_Idx_d, C_Count_d,
	//			n_d, m_d, nnz_d, residuals_d,x_d, lambda_d, Li_d, NMax_d);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	checkCUDAError("kernel execution Paralel MERSEN NRCDM CUDA");
	printf("Paralel MERSEN NRCDM CUDA TIMMER:  %f ms\n", time);

	//	checkCUDAError("kernel execution");
	for (i = 0; i < m; i++)
		residuals[i] = -b[i];

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	NRCDM_SR_TIMING(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, b, x, lambda,
			Li, SerialExecution, value, 1, residuals);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeSerial, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	checkCUDAError("kernel execution");
	printf("SERIAL NRCDM CUDA TIMMER:  %f ms, speed up:%f\n", timeSerial
			* NMAXITER / SerialExecution, timeSerial / time * NMAXITER
			/ SerialExecution);

	//------------------------- LOAD RESULTS FROM CUDA AND COMPARE
	float* xcuda;
	xcuda = (float*) malloc(n * sizeof(float));
	cudaMemcpy(xcuda, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy");
	cudaThreadSynchronize();
	checkCUDAError("cudaMemcpy");
	for (i = 0; i < 20; i++) {
		printf("x_h[%d]=%f, x_d[%d]=%f\n", i, x[i], i, xcuda[i]);
	}

	float normErrorSquered = 0;
	float normX = 0;
	for (i = 0; i < n; i++) {
		normErrorSquered += (x[i] - xcuda[i]) * (x[i] - xcuda[i]);
		normX += x[i] * x[i];
	}
	for (i = 0; i < n; i++) {
		if (xcuda[i] * xcuda[i] < normX) {
			;
		} else {
			//			printf("x_h[%d]=%f, x_d[%d]=%f\n", i, x[i], i, xcuda[i]);
		}
	}
	printf("normaChyby=%f;   norm(x)=%f,   n=%d, m=%d\n", normErrorSquered,
			normX, n, m);

	float minE = 100000000;
	float maxE = -100000000;
	for (i = 0; i < n; i++) {
		if (xcuda[i] < minE)
			minE = xcuda[i];
		if (xcuda[i] > maxE)
			maxE = xcuda[i];
	}
	printf("minE=%f, maxE=%f\n", minE, maxE);

	//------------------------- CLEAN UP
	printf("Idem uvolnovat\n");
	printf("Paralel MERSEN NRCDM CUDA TIMMER:  %f ms\n", time);
	printf("SERIAL NRCDM CUDA TIMMER:  %f ms, speed up:%f\n", timeSerial
			* NMAXITER / SerialExecution, timeSerial / time * NMAXITER
			/ SerialExecution);
	printf("m:%d, n:%d, p:%d\n", m, n, COLUMNLENGTH);

	cudaFree(A_d);
	cudaFree(Li_d);
	cudaFree(R_Idx_d);
	cudaFree(C_Idx_d);
	cudaFree(C_Count_d);

	cudaFree(n_d);
	cudaFree(nnz_d);
	cudaFree(m_d);
	cudaFree(x_d);
	cudaFree(lambda_d);
	cudaFree(residuals_d);
	fclose(fp);
}

void cudaMersenSolverEvolution() {
	int totalThreads = 32;
	//	int dim1 = 1024;
	//	int dim2 = 128;
	int dim1 = 14;
	int dim2 = 1;
	int n = totalThreads * dim1 * dim2  *CUDACount;
	//	dim2=dim2/10;
	int m =2* n ;

	FILE *fp;
	//		fp = fopen("/exports/home/s1052689/nesterov2.txt", "w");
	fp = fopen("/tmp/taki_cuda.txt", "w");
	clock_t t1, t2;
	int NMax_d_h[1];
	int pMin = COLUMNLENGTH;
	int pMax = COLUMNLENGTH;
	float lambda = 1;
	float diff;
	float* A_h;
	int * C_Idx_h;
	int * R_Idx_h;
	int * C_Count_h;
	int nnz, i, j;

	printf("CudaSolver\n");

	NMax_d_h[0] = 1;

	/* Allocate space for prng states on device */
	dim3 dimBlock( totalThreads);
	dim3 dimGridRCDM( dim1, dim2);
	t1 = clock();
	/* Setup prng states */
	t1 = clock();
	float * b;
	float * xOpt;
	generateNesterovProblem(&A_h, &R_Idx_h, &C_Idx_h, &C_Count_h, &b, &xOpt, n,
			m, pMin, pMax - pMin, &nnz);


	double OptimalValueByNesterov = ComputeObjectiveValue(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m,
					nnz, b, xOpt, lambda);
	printf("optimal value 2:%f\n",OptimalValueByNesterov);

	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("NNZ: %d\n", nnz);
	printf("Random problem generation: %f\n", diff);
	printf("Random problem generated\n");
	float* L;
	float* Li;
	computeLipsitzConstantsForSparseRegression(&L, &Li, A_h, R_Idx_h, C_Idx_h,
			C_Count_h, n, nnz);
	printf("Lipshitz constants computed\n");
	//		printMatrixA(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, nnz);
	//	print_float_array(&L[0], n);
	float x[n];
	float value = 0;
	for (i = 0; i < n; i++)
		x[i] = 0;
	printf("Idem inicializovat reziduals\n");
	float residuals[m];
	for (j = 0; j < m; j++)
		residuals[j] = -b[j];
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count_h[i]; j++) {
			residuals[R_Idx_h[C_Idx_h[i] + j]] += A_h[C_Idx_h[i] + j] * x[i];
		}
	}

	float* A_d;
	float* Li_d;
	int * C_Idx_d;
	int * R_Idx_d;
	int * C_Count_d;
	int *n_d;
	int *nnz_d;
	int *m_d;
	float* x_d;
	float* lambda_d;
	float * residuals_d;
	int * NMax_d;
	//	size_t memSize = ;


	t1 = clock();
	printf("Idem alokovat data na device\n");
	cudaMalloc((void**) &A_d, nnz * sizeof(float));
	checkCUDAError("kernel execution Alloc A_d");
	cudaMalloc((void**) &R_Idx_d, nnz * sizeof(int));
	checkCUDAError("kernel execution Alloc R_IDX");
	cudaMalloc((void**) &C_Idx_d, n * sizeof(int));
	checkCUDAError("kernel execution Alloc C_IDX");
	cudaMalloc((void**) &Li_d, n * sizeof(float));
	checkCUDAError("kernel execution Alloc Li");
	cudaMalloc((void**) &C_Count_d, n * sizeof(int));
	checkCUDAError("kernel execution Alloc CCount");
	cudaMalloc((void**) &n_d, 1 * sizeof(int));
	cudaMalloc((void**) &nnz_d, 1 * sizeof(int));
	cudaMalloc((void**) &m_d, 1 * sizeof(int));
	cudaMalloc((void**) &x_d, n * sizeof(float));
	cudaMalloc((void**) &residuals_d, m * sizeof(float));
	checkCUDAError("kernel execution Alloc Residuals");
	cudaMalloc((void**) &lambda_d, 1 * sizeof(float));
	cudaMalloc((void**) &NMax_d, 1 * sizeof(int));
	checkCUDAError("kernel execution Alloc N_d");
	cudaThreadSynchronize();
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Data alocated %d, %d from host -> device:%f\n", nnz, n, diff);

	// Part 2 of 5: host to device memory copy
	t1 = clock();
	cudaMemcpy(A_d, A_h, nnz * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy A_d");
	cudaMemcpy(C_Count_d, C_Count_h, n * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy C_Cound_d");
	cudaMemcpy(R_Idx_d, R_Idx_h, nnz * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy R");
	cudaMemcpy(Li_d, Li, n * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy Li");
	cudaMemcpy(C_Idx_d, C_Idx_h, n * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy C_ID");
	cudaMemcpy(n_d, &n, 1 * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy Nd");
	cudaMemcpy(nnz_d, &nnz, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m_d, &m, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(NMax_d, &NMax_d_h, 1 * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy NMax");
	cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy x_d");
	cudaMemcpy(residuals_d, residuals, m * sizeof(float),
			cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy Resiaduals");
	cudaMemcpy(lambda_d, &lambda, 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Copy data %d, %d from host -> device:%f\n", nnz, n, diff);

	printf("Idem spustat paralelny RCDM\n");
	cudaThreadSynchronize();

	for (i = 0; i < m; i++)
		residuals[i] = -b[i];
	double FunVal = 0;
	FunVal = ComputeObjectiveValue(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz,
			b, x, lambda);
	printf("CUDA:Objective,itr,0,value,%1.16f\n", FunVal);
	printf("CPUD:Objective,itr,0,value,%1.16f\n", FunVal);
	fprintf(fp,"CUDA:Objective,itr,0,value,%1.16f\n", FunVal);
	fprintf(fp,"CPUD:Objective,itr,0,value,%1.16f\n", FunVal);

	float time, timeSerial;
	float* xcuda;
	xcuda = (float*) malloc(n * sizeof(float));

	curandState *devStates;
	cudaMalloc((void **) &devStates, dim1 * dim2 * totalThreads
			* sizeof(curandState));
	setup_kernel<<< dimGridRCDM, dimBlock >>>(devStates);
	checkCUDAError("Inicializacia ranom states");

	for (i = 0; i < 200; i++) {
		t1 = clock();
		checkCUDAError("kernel execution Paralel MERSEN NRCDM CUDA - BEFORE");
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		RCDMKernel<<< dimGridRCDM, dimBlock >>>(A_d, R_Idx_d, C_Idx_d, C_Count_d,
				n_d, m_d, nnz_d, residuals_d,residuals_d,x_d, lambda_d, Li_d,NMax_d, devStates);
		//		MersenRandomKernelDoubleGeneratorSingle<<< dimGridRCDM, dimBlock >>>(A_d, R_Idx_d, C_Idx_d, C_Count_d,
		//				n_d, m_d, nnz_d, residuals_d,x_d, lambda_d, Li_d, NMax_d);

		//	MersenRandomKernel<<< dimGridRCDM, dimBlock >>>(A_d, R_Idx_d, C_Idx_d, C_Count_d,
		//			n_d, m_d, nnz_d, residuals_d,x_d, lambda_d, Li_d, NMax_d);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaMemcpy(xcuda, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy");
		FunVal = ComputeObjectiveValue(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m,
				nnz, b, xcuda, lambda);
		printf("CUDA:Objective,itr,%d,value,%1.16f,value,%1.16f,time,%f\n", i, FunVal,FunVal-OptimalValueByNesterov, time);
		fprintf(fp, "CUDA:Objective,itr,%d,value,%1.16f,value,%1.16f,time,%f\n", i, FunVal,FunVal-OptimalValueByNesterov, time);

		checkCUDAError("kernel execution Paralel MERSEN NRCDM CUDA");

		//	checkCUDAError("kernel execution");


		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		NRCDM_SR_TIMING(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, b, x,
				lambda, Li, SerialExecution, value, 1, residuals);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timeSerial, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		FunVal = ComputeObjectiveValue(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m,
				nnz, b, x, lambda);
		printf("CPUD:Objective,itr,%d,value,%1.16f,value,%1.16f,time,%f\n", i, FunVal,FunVal-OptimalValueByNesterov,timeSerial);
		fprintf(fp, "CPUD:Objective,itr,%d,value,%1.16f,value,%1.16f,time,%f\n", i, FunVal,FunVal-OptimalValueByNesterov, 			timeSerial);

		checkCUDAError("kernel execution");
		printf("SERIAL NRCDM CUDA TIMMER:  %f ms, speed up:%f, paralel: %f\n", 		timeSerial, timeSerial / time, time);
		fprintf(fp,
				"SERIAL NRCDM CUDA TIMMER:  %f ms, speed up:%f, paralel: %f\n",
				timeSerial, timeSerial / time, time);

	}

	//------------------------- LOAD RESULTS FROM CUDA AND COMPARE


	cudaThreadSynchronize();
	checkCUDAError("cudaMemcpy");
	for (i = 0; i < 20; i++) {
		printf("x_h[%d]=%f, x_d[%d]=%f\n", i, x[i], i, xcuda[i]);
	}

	float normErrorSquered = 0;
	float normX = 0;
	for (i = 0; i < n; i++) {
		normErrorSquered += (x[i] - xcuda[i]) * (x[i] - xcuda[i]);
		normX += x[i] * x[i];
	}
	for (i = 0; i < n; i++) {
		if (xcuda[i] * xcuda[i] < normX) {
			;
		} else {
			//			printf("x_h[%d]=%f, x_d[%d]=%f\n", i, x[i], i, xcuda[i]);
		}
	}
	printf("normaChyby=%f;   norm(x)=%f,   n=%d, m=%d\n", normErrorSquered,
			normX, n, m);

	float minE = 100000000;
	float maxE = -100000000;
	int nnzCuda = 0;
	for (i = 0; i < n; i++) {
		if (xcuda[i] == 0)
			nnzCuda++;
		if (xcuda[i] < minE)
			minE = xcuda[i];
		if (xcuda[i] > maxE)
			maxE = xcuda[i];
	}
	printf("minE=%f, maxE=%f\n", minE, maxE);
	printf("Zeros in cuda %d \n", nnzCuda);
	//------------------------- CLEAN UP
	printf("Idem uvolnovat\n");
	printf("Paralel MERSEN NRCDM CUDA TIMMER:  %f ms\n", time);
	printf("SERIAL NRCDM CUDA TIMMER:  %f ms, speed up:%f\n", timeSerial
			* NMAXITER / SerialExecution, timeSerial / time * NMAXITER
			/ SerialExecution);
	printf("m:%d, n:%d, p:%d\n", m, n, COLUMNLENGTH);
	printf("optimal value 2:%1.16f\n",OptimalValueByNesterov);
	fprintf(fp,"optimal value 2:%1.16f\n",OptimalValueByNesterov);
	cudaFree(A_d);
	cudaFree(Li_d);
	cudaFree(R_Idx_d);
	cudaFree(C_Idx_d);
	cudaFree(C_Count_d);

	cudaFree(n_d);
	cudaFree(nnz_d);
	cudaFree(m_d);
	cudaFree(x_d);
	cudaFree(lambda_d);
	cudaFree(residuals_d);
	fclose(fp);
}

void cudaEDDIESolver() {
	FILE *fp;
	fp = fopen("/exports/home/s1052689/cudaLogs.txt", "w");
	//	fp = fopen("/tmp/taki_cuda.txt", "w");
	clock_t t1, t2;
	int NMax_d_h[1];
	int pMin = 10;
	int pMax = 10;
	float lambda = 1;
	float diff;
	float* A_h;
	int * C_Idx_h;
	int * R_Idx_h;
	int * C_Count_h;
	int nnz, i, j;
	cudaEvent_t start, stop;
	float time, timeSerial;
	printf("CudaSolver\n");

	int NMax = 1;
	curandState *devStates;
	int totalThreads = 128;
	int dim1 = 300;
	int dim2 = NMax;
	int n = totalThreads * dim1;
	int m = n * 2;
	NMax_d_h[0] = 1;

	printf("CudaSolver2\n");

	printf("idem alokovat random states");
	dim3 dimGrid( dim1, dim2);
	dim3 dimBlock( totalThreads);

	t1 = clock();
	cudaMalloc((void **) &devStates, dim1 * dim2 * totalThreads
			* sizeof(curandState));
	checkCUDAError("Alloc error");
	/* Setup prng states */
	setup_kernel<<< dimGrid, dimBlock >>>(devStates);
	//	myDoNothingKernel<<< 13, 1 >>>();


	cudaThreadSynchronize();
	checkCUDAError("Inicializacia ranom states");
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Inicializacia ranom states: %f\n", diff);
	printf("Idem generaovt random problem");

	/* Allocate space for prng states on device */

	t1 = clock();
	generateRandomProblem(&A_h, &R_Idx_h, &C_Idx_h, &C_Count_h, n, m, pMin,
			pMax - pMin, &nnz);
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Random problem generation: %f\n", diff);
	printf("Random problem generated\n");
	float* L;
	float* Li;
	computeLipsitzConstantsForSparseRegression(&L, &Li, A_h, R_Idx_h, C_Idx_h,
			C_Count_h, n, nnz);
	printf("Lipshitz constants computed\n");
	//		printMatrixA(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, nnz);
	//	print_float_array(&L[0], n);
	float b[m];
	float xOpt[n];
	float x[n];
	for (j = 0; j < n; j++) {
		xOpt[j] = 2 * ((float) rand() / RAND_MAX) - 1;
	}
	printf("Optimal solution generated\n");
	// set b = A*x
	matrix_vector_product(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, xOpt, b,
			1);
	printf("matrix vector product multiplied\n");
	//	print_float_array(&xOpt[0], n);
	//	print_float_array(&b[0], m);
	float value = 0;
	for (i = 0; i < n; i++)
		x[i] = 0;
	printf("Idem inicializovat reziduals\n");
	float residuals[m];
	for (j = 0; j < m; j++)
		residuals[j] = -b[j];
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count_h[i]; j++) {
			residuals[R_Idx_h[C_Idx_h[i] + j]] += A_h[C_Idx_h[i] + j] * x[i];
		}
	}

	float* A_d;
	float* Li_d;
	int * C_Idx_d;
	int * R_Idx_d;
	int * C_Count_d;
	int *n_d;
	int *nnz_d;
	int *m_d;
	float* x_d;
	float* b_d;
	float* lambda_d;
	float * residuals_d;
	int * NMax_d;
	//	size_t memSize = ;


	t1 = clock();
	printf("Idem alokovat data na device\n");
	cudaMalloc((void**) &A_d, nnz * sizeof(float));
	cudaMalloc((void**) &R_Idx_d, nnz * sizeof(int));
	cudaMalloc((void**) &C_Idx_d, n * sizeof(int));
	cudaMalloc((void**) &Li_d, n * sizeof(float));
	cudaMalloc((void**) &C_Count_d, n * sizeof(int));
	cudaMalloc((void**) &n_d, 1 * sizeof(int));
	cudaMalloc((void**) &nnz_d, 1 * sizeof(int));
	cudaMalloc((void**) &m_d, 1 * sizeof(int));
	cudaMalloc((void**) &x_d, n * sizeof(float));
	cudaMalloc((void**) &b_d, m * sizeof(float));
	cudaMalloc((void**) &residuals_d, m * sizeof(float));
	cudaMalloc((void**) &lambda_d, 1 * sizeof(float));
	cudaMalloc((void**) &NMax_d, 1 * sizeof(int));

	cudaThreadSynchronize();
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Data alocated %d, %d from host -> device:%f\n", nnz, n, diff);

	// Part 2 of 5: host to device memory copy
	t1 = clock();
	cudaMemcpy(A_d, A_h, nnz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C_Count_d, C_Count_h, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(R_Idx_d, R_Idx_h, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Li_d, Li, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(C_Idx_d, C_Idx_h, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(n_d, &n, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nnz_d, &nnz, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m_d, &m, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(NMax_d, &NMax_d_h, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, m * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(residuals_d, residuals, m * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(lambda_d, &lambda, 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Copy data %d, %d from host -> device:%f\n", nnz, n, diff);

	printf("Idem spustat paralelny RCDM\n");
	cudaThreadSynchronize();
	t1 = clock();

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	RCDMKernel<<< dimGrid, dimBlock >>>(A_d, R_Idx_d, C_Idx_d, C_Count_d,
			n_d, m_d, nnz_d, b_d, residuals_d,x_d, lambda_d, Li_d, NMax_d,devStates);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	checkCUDAError("kernel execution");
	printf("Paralel NRCDM CUDA TIMMER:  %f ms\n", time);

	//	checkCUDAError("kernel execution");
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	NRCDM_SR(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, b, x, lambda, Li, 50,
			value, 1);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeSerial, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	checkCUDAError("kernel execution");
	printf("SERIAL NRCDM CUDA TIMMER:  %f ms, speed up:%f\n", timeSerial,
			timeSerial / time);

	//------------------------- LOAD RESULTS FROM CUDA AND COMPARE
	float* xcuda;
	xcuda = (float*) malloc(n * sizeof(float));
	cudaMemcpy(xcuda, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy");
	cudaThreadSynchronize();
	checkCUDAError("cudaMemcpy");
	for (i = 0; i < 2; i++) {
		printf("x_h[%d]=%f, x_d[%d]=%f\n", i, x[i], i, xcuda[i]);
	}

	//------------------------- CLEAN UP
	printf("Idem uvolnovat\n");
	cudaFree(A_d);
	cudaFree(Li_d);
	cudaFree(R_Idx_d);
	cudaFree(C_Idx_d);
	cudaFree(C_Count_d);

	cudaFree(n_d);
	cudaFree(nnz_d);
	cudaFree(m_d);
	cudaFree(x_d);
	cudaFree(b_d);
	cudaFree(lambda_d);
	cudaFree(residuals_d);
	cudaFree(devStates);
	fclose(fp);
}

void generator() {
	//	size_t n = 100;
	//	size_t i;
	//	curandGenerator_t gen;
	//	float *devData, *hostData;
	//	/* Allocate n floats on host */
	//	hostData = (float *) calloc(n, sizeof(float));
	//	/* Allocate n floats on device */
	//	cudaMalloc((void **) &devData, n * sizeof(float));
	//	/* Create pseudo-random number generator */
	//	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	//	/* Set seed */
	//	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	//	/* Generate n floats on device */
	//	curandGenerateUniform(gen, devData, n);
	//	/* Copy device memory to host */
	//	cudaMemcpy(hostData, devData, n * sizeof(float), cudaMemcpyDeviceToHost);
	//	/* Show result */
	//	for (i = 0; i < n; i++) {
	//		printf("%1.4f ", hostData[i]);
	//	}
	//	printf("\n");
}
__global__ void myFirstIntKernel(int *d_a) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	d_a[idx] = 1000 * blockIdx.x + threadIdx.x;
}

void eddietest() {
	// pointer for host memory
	int *h_a;

	// pointer for device memory
	int *d_a;

	// define grid and block size
	int numBlocks = 8;
	int numThreadsPerBlock = 8;

	// Part 1 of 5: allocate host and device memory
	size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int);
	h_a = (int *) malloc(memSize);
	cudaMalloc((void **) &d_a, memSize);

	// Part 2 of 5: launch kernel
	dim3 dimGrid( numBlocks);
	dim3 dimBlock( numThreadsPerBlock);
	myFirstIntKernel<<< dimGrid, dimBlock >>>( d_a );

	// block until the device has completed
	cudaThreadSynchronize();

	// check if kernel execution generated an error
	checkCUDAError("kernel execution");

	// Part 4 of 5: device to host copy
	cudaMemcpy(h_a, d_a, memSize, cudaMemcpyDeviceToHost);

	// Check for any CUDA errors
	checkCUDAError("cudaMemcpy");

	// Part 5 of 5: verify the data returned to the host is correct
	for (int i = 0; i < numBlocks; i++) {
		for (int j = 0; j < numThreadsPerBlock; j++) {

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
	cudaMersenSolverEvolution();

}

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}

void cudaSolverOLD() {
	FILE *fp;
	//		fp = fopen("/exports/home/s1052689/nesterov2.txt", "w");
	fp = fopen("/tmp/taki_cuda.txt", "w");
	clock_t t1, t2;
	int NMax_d_h[1];
	int pMin = 10;
	int pMax = 10;
	float lambda = 1;
	float diff;
	float* A_h;
	int * C_Idx_h;
	int * R_Idx_h;
	int * C_Count_h;
	int nnz, i, j;

	printf("CudaSolver\n");

	int NMax = 1;
	curandState *devStates;
	int totalThreads = 512;
	int dim1 = 1;
	int dim2 = NMax;
	int n = totalThreads * dim1;
	int m = n * 2;
	NMax_d_h[0] = 1;

	printf("CudaSolver2\n");

	/* Allocate space for prng states on device */
	printf("idem alokovat random states");
	dim3 dimGrid( dim1, dim2);
	dim3 dimBlock( totalThreads);

	t1 = clock();
	cudaMalloc((void **) &devStates, dim1 * dim2 * totalThreads
			* sizeof(curandState));
	checkCUDAError("Alloc error");
	/* Setup prng states */
	setup_kernel<<< dimGrid, dimBlock >>>(devStates);
	cudaThreadSynchronize();
	checkCUDAError("Inicializacia ranom states");
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Inicializacia ranom states: %f\n", diff);
	printf("Idem generaovt random problem");
	t1 = clock();
	generateRandomProblem(&A_h, &R_Idx_h, &C_Idx_h, &C_Count_h, n, m, pMin,
			pMax - pMin, &nnz);
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Random problem generation: %f\n", diff);
	printf("Random problem generated\n");
	float* L;
	float* Li;
	computeLipsitzConstantsForSparseRegression(&L, &Li, A_h, R_Idx_h, C_Idx_h,
			C_Count_h, n, nnz);
	printf("Lipshitz constants computed\n");
	//		printMatrixA(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, nnz);
	//	print_float_array(&L[0], n);
	float b[m];
	float xOpt[n];
	float x[n];
	for (j = 0; j < n; j++) {
		xOpt[j] = 2 * ((float) rand() / RAND_MAX) - 1;
	}
	printf("Optimal solution generated\n");
	// set b = A*x
	matrix_vector_product(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, xOpt, b,
			1);
	printf("matrix vector product multiplied\n");
	//	print_float_array(&xOpt[0], n);
	//	print_float_array(&b[0], m);
	float value = 0;
	for (i = 0; i < n; i++)
		x[i] = 0;
	printf("Idem inicializovat reziduals\n");
	float residuals[m];
	for (j = 0; j < m; j++)
		residuals[j] = -b[j];
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count_h[i]; j++) {
			residuals[R_Idx_h[C_Idx_h[i] + j]] += A_h[C_Idx_h[i] + j] * x[i];
		}
	}

	float* A_d;
	float* Li_d;
	int * C_Idx_d;
	int * R_Idx_d;
	int * C_Count_d;
	int *n_d;
	int *nnz_d;
	int *m_d;
	float* x_d;
	float* b_d;
	float* lambda_d;
	float * residuals_d;
	int * NMax_d;
	//	size_t memSize = ;


	t1 = clock();
	printf("Idem alokovat data na device\n");
	cudaMalloc((void**) &A_d, nnz * sizeof(float));
	cudaMalloc((void**) &R_Idx_d, nnz * sizeof(int));
	cudaMalloc((void**) &C_Idx_d, n * sizeof(int));
	cudaMalloc((void**) &Li_d, n * sizeof(float));
	cudaMalloc((void**) &C_Count_d, n * sizeof(int));
	cudaMalloc((void**) &n_d, 1 * sizeof(int));
	cudaMalloc((void**) &nnz_d, 1 * sizeof(int));
	cudaMalloc((void**) &m_d, 1 * sizeof(int));
	cudaMalloc((void**) &x_d, n * sizeof(float));
	cudaMalloc((void**) &b_d, m * sizeof(float));
	cudaMalloc((void**) &residuals_d, m * sizeof(float));
	cudaMalloc((void**) &lambda_d, 1 * sizeof(float));
	cudaMalloc((void**) &NMax_d, 1 * sizeof(int));

	cudaThreadSynchronize();
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Data alocated %d, %d from host -> device:%f\n", nnz, n, diff);

	// Part 2 of 5: host to device memory copy
	t1 = clock();
	cudaMemcpy(A_d, A_h, nnz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C_Count_d, C_Count_h, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(R_Idx_d, R_Idx_h, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Li_d, Li, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(C_Idx_d, C_Idx_h, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(n_d, &n, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nnz_d, &nnz, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m_d, &m, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(NMax_d, &NMax_d_h, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, m * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(residuals_d, residuals, m * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(lambda_d, &lambda, 1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Copy data %d, %d from host -> device:%f\n", nnz, n, diff);

	printf("Idem spustat paralelny RCDM\n");
	cudaThreadSynchronize();
	t1 = clock();

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	RCDMKernel<<< dimGrid, dimBlock >>>(A_d, R_Idx_d, C_Idx_d, C_Count_d,
			n_d, m_d, nnz_d, b_d, residuals_d,x_d, lambda_d, Li_d, NMax_d,devStates);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	checkCUDAError("kernel execution");
	printf("Paralel NRCDM CUDA TIMMER:  %f\n", time);

	//	checkCUDAError("kernel execution");


	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Paralel NRCDM + Synchronize:  %f\n", diff);
	fprintf(fp, "Paralel NRCDM + Synchronize:  %f\n", diff);

	//	myFirstfloatKernel<<< dimGrid, dimBlock >>>( x_d );
	//	myFirstKernel<<< dimGrid, dimBlock >>>(x_d);
	//	cudaThreadSynchronize();


	t1 = clock();
	NRCDM_SR(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, b, x, lambda, Li,
			NMax, value, 1);
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("NRCDM:  %f\n", nnz, n, diff);
	fprintf(fp, "NRCDM:  %f\n", nnz, n, diff);

	//------------------------- LOAD RESULTS FROM CUDA AND COMPARE
	float* xcuda;
	xcuda = (float*) malloc(n * sizeof(float));
	cudaMemcpy(xcuda, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy");
	cudaThreadSynchronize();
	checkCUDAError("cudaMemcpy");
	for (i = 0; i < 2; i++) {
		printf("x_h[%d]=%f, x_d[%d]=%f\n", i, x[i], i, xcuda[i]);
	}

	//------------------------- CLEAN UP
	printf("Idem uvolnovat\n");
	cudaFree(A_d);
	cudaFree(Li_d);
	cudaFree(R_Idx_d);
	cudaFree(C_Idx_d);
	cudaFree(C_Count_d);

	cudaFree(n_d);
	cudaFree(nnz_d);
	cudaFree(m_d);
	cudaFree(x_d);
	cudaFree(b_d);
	cudaFree(lambda_d);
	cudaFree(residuals_d);
	cudaFree(devStates);
	fclose(fp);
}
