#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//ulimit -s unlimited

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
void print_double_array(const double *array, size_t len) {
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

void generateRandomProblem(double ** A, int** R_Idx, int**C_Idx, int **C_Count,
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
	*A = (double *) malloc(nnz * sizeof(double));
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
			(*A)[nnz] = (double) rand() / RAND_MAX;
			//			printf("A[%d][%d] = %f\n", nnz, (*R_Idx)[nnz], (*A)[nnz]);
			nnz++;
		}
	}
}

void fullVector(double** test, int n) {

	*test = (double *) malloc(n * sizeof(double));
	int i;
	for (i = 0; i < n; i++) {
		(*test)[i] = -11;
	}

	//	double* TEST;
	//		fullVector(&TEST, 100);
	//		int asdf;
	//		for (asdf = 0; asdf < 10; asdf++) {
	//			printf("TEST[%d]= %f\n", asdf, TEST[asdf]);
	//		}

}

double NRCDM_SR(double *A, int*R_Idx, int*C_Idx, int*C_Count, int n, int m,
		int nnz, double* b, double*x, double lambda, double* Li, int NMAX, double optimalvalue
, int log
) {
	double residuals[m];double value = 0;
	int i, j, N;
	int sample = n/100;
	if (log = 0) sample = n;
	FILE *fp;
	fp = fopen("/tmp/sparseregression.csv", "w");
	// calculate residuals
	for (j = 0; j < m; j++)
		residuals[j] = -b[j];
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count[i]; j++) {
			residuals[R_Idx[C_Idx[i] + j]] += A[C_Idx[i] + j] * x[i];
		}
	}
	double partialDetivative, delta, tmp;
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
		if (N % (n/100) == 0) {
			int nnzcount = 0;
			  value = 0;
			for (i = 0; i < n; i++) {
				if (x[i]!=0) nnzcount++;
				if (x[i] > 0)
					value += x[i];
				else
					value -= x[i];
			}
			for (j = 0; j < m; j++)
				value += 0.5 * residuals[j] * residuals[j];
			fprintf(fp,"Iteracia:%d, value:%f, nnz:%d, epsilon: %f\n", N, value, nnzcount, value-optimalvalue);
		}

	}
	fclose(fp);
	return value;
}

void printMatrixA(double *A, int*R_Idx, int*C_Idx, int*C_Count, int n, int nnz) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count[i]; j++) {
			printf("A[%d][%d] = %f \n", i, R_Idx[C_Idx[i] + j], A[C_Idx[i] + j]);
		}
	}
}

void matrix_vector_product(double *A, int*R_Idx, int*C_Idx, int*C_Count, int n,
		int m, int nnz, double *x, double* b, double c) {
	int i, j;
	for (j = 0; j < m; j++)
		b[j] = 0;
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count[i]; j++) {
			b[R_Idx[C_Idx[i] + j]] += c * A[C_Idx[i] + j] * x[i];
		}
	}
}

void computeLipsitzConstantsForSparseRegression(double** L, double ** Li,
		double *A, int*R_Idx, int*C_Idx, int*C_Count, int n, int nnz) {
	int i, j;
	*L = (double *) malloc(n * sizeof(double));
	*Li = (double *) malloc(n * sizeof(double));
	for (i = 0; i < n; i++) {
		(*L)[i] = 0;
		for (j = 0; j < C_Count[i]; j++) {
			(*L)[i] += A[C_Idx[i] + j] * A[C_Idx[i] + j];
		}
		(*Li)[i] = 1 / (*L)[i];
	}
}

int main(void) {
	srand(1);
	int n = 100;
	int m =50;
	int pMin = 10;
	int pMax = 40;
	double lambda = 1;
	double C = 1;

	int NMax = 1000;

	double* A_h;
	int * C_Idx_h;
	int * R_Idx_h;
	int * C_Count_h;
	int nnz, i, j, k, l;
	generateRandomProblem(&A_h, &R_Idx_h, &C_Idx_h, &C_Count_h, n, m, pMin,
			pMax - pMin, &nnz);

	double* L;
	double* Li;
	computeLipsitzConstantsForSparseRegression(&L, &Li, A_h, R_Idx_h, C_Idx_h,
			C_Count_h, n, nnz);
	//		printMatrixA(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, nnz);
	//	print_double_array(&L[0], n);
	double b[m];
	double xOpt[n];
	double x[n];
	for (j = 0; j < n; j++) {
		xOpt[j] = 2 * ((double) rand() / RAND_MAX) - 1;
	}
	// set b = A*x
	matrix_vector_product(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, xOpt, b,
			1);
	//	print_double_array(&xOpt[0], n);
	//	print_double_array(&b[0], m);
	double value=0;
	for (i = 0; i < n; i++)
		x[i] = 0;
	  value = NRCDM_SR(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, b, x, lambda, Li,
			NMax*100, value,0);
	  for (i = 0; i < n; i++)
	  		x[i] = 0;
	  value = NRCDM_SR(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, b, x, lambda, Li,
	  			NMax,value,1);




//	print_double_array(&x[0], n);
//	print_double_array(&xOpt[0], n);







	float * A_dev;
	float * L_dev;
	float * x_dev;
	float * b_dev;
	float * lambda_dev;

}
