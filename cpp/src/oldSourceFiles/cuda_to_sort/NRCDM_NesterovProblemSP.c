#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//ulimit -s unlimited
#include <time.h>
//#include <cuda.h>

// Kernel that executes on the CUDA device
//__global__ void square_array(double *a, int N)
//{
//  int idx = blockIdx.x * blockDim.x + threadIdx.x;
//  if (idx<N) a[idx] = a[idx] * a[idx];
//}

// main routine that executes on the host

//void RCDM(double** A, int IDX, double L, double b, double lambda, int n,
//		int N) {
//
//}
/* integer array printing function */
void print_double_array(const double *array, size_t len) {
	size_t i;

	for (i = 0; i < len; i++)
		printf("%f | ", array[i]);

	putchar('\n');
}
void print_int_array(const int *array, size_t len) {
	size_t i;

	for (i = 0; i < len; i++)
		printf("%d | ", array[i]);

	putchar('\n');
}

/* an example of struct */
struct st_sortingByAbsWithIndex {
	double value;
	int idx;
};

struct optimalityAnalysis {
	double accuracy;
	int nnz;
	int correctnnz;
	int iteration;
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

int main(void) {
	srand(2);

	double lambda = 1;
	double rho = 1;
	long int n = 100000;
	int n_nonzero = 1600;
	long int m = n / 10;
	double sqrtofnonzeros = 40;
	int p = 100;

	long int NMAX = 100000;
	long int N;

	int samplingsize = n;

	int i, j, k;

	double A_h[n][p]; // host A matrix pointers
	int IDX_h[n][p]; // host Aindex matrix pointers
	double x_optimal[n];
	double optimalvalue = 0;
	double b[m];
	double x[n];
	double tmp;
	double L[n]; // Lipschitz constants
	double Li[n]; // Lipschitz constants


	FILE *fp;
	fp = fopen("/tmp/nesterov.csv", "w");

	//Generovanie problemu-------------------------------------------------------------------
	for (i = 0; i < n; i++) {
		int idx = 0;
		for (j = 0; j < p; j++) {
			int notfinished = 1;
			double val = (double) rand() / RAND_MAX;
			while (notfinished) {
				notfinished = 0;
				idx = ((int) ((m) * (rand() / (RAND_MAX + 1.0))));
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
	double v[m];
	double y[m];
	tmp = 0;
	for (j = 0; j < m; j++) {
		v[j] = (double) rand() / RAND_MAX;
		tmp += v[j] * v[j];
	}
	for (j = 0; j < m; j++)
		y[j] = v[j] / tmp;
	struct st_sortingByAbsWithIndex dataToSort[n];
	for (i = 0; i < n; i++) {
		dataToSort[i].idx = i;
		dataToSort[i].value = 0;
	}
	for (i = 0; i < n; i++) {
		double tmp = 0;
		for (j = 0; j < p; j++) {
			dataToSort[i].value += y[IDX_h[i][j]] * A_h[i][j];
		}
	}
	//Sorting B
	printf("SORTING START\n");
	size_t structs_len = sizeof(dataToSort)
			/ sizeof(struct st_sortingByAbsWithIndex);
	qsort(dataToSort, structs_len, sizeof(struct st_sortingByAbsWithIndex),
			struct_cmp_by_value);
	printf("SORTING END\n");
	for (i = 0; i < n; i++) { // vytvaranie matice A
		int idx = dataToSort[i].idx;
		double alpha = 1;
		x_optimal[idx] = 0;
		if (i < n_nonzero) {

			alpha = (double) abs(1 / dataToSort[idx].value);
			//			printf("alpha = %f \n", alpha);
			x_optimal[idx] = ((double) rand() / RAND_MAX) * rho
					/ (sqrtofnonzeros);
			if (dataToSort[idx].value < 0) {
				x_optimal[idx] = -x_optimal[idx];
			}
		} else if (dataToSort[idx].value > 0.1 || dataToSort[idx].value < -0.1) {
			alpha = (double) abs(1 / dataToSort[idx].value) * (double) rand()
					/ RAND_MAX;
			//			printf("alpha = %f \n", alpha);
		}
		L[idx] = 0;
		for (j = 0; j < p; j++) {

			A_h[idx][j] = A_h[idx][j] * alpha;
			//			printf("A[%d][%d]=%1.10f \n",idx,j,A_h[idx][j]);
			L[idx] += A_h[idx][j] * A_h[idx][j];
		}
		Li[idx] = 1 / L[idx];
		//		printf("L[%d]=%1.10f  Li[%d]=%1.10f \n",idx,L[idx],idx,Li[idx]);
	}
	//	print_double_array(&L[0],n);
	//	print_double_array(&Li[0],n);

	for (j = 0; j < m; j++) {
		b[j] = y[j];
	}
	for (i = 0; i < n; i++) {
		for (j = 0; j < p; j++) {
			b[IDX_h[i][j]] += x_optimal[i] * A_h[i][j];
		}

	}
	for (i = 0; i < m; i++) {
		optimalvalue += y[i] * y[i];
	}
	optimalvalue = optimalvalue * 0.5;
	for (i = 0; i < n; i++) {
		//		printf("optval %1.16f \n", optimalvalue);
		if (x_optimal[i] > 0)
			optimalvalue += x_optimal[i];
		else
			optimalvalue -= x_optimal[i];
	}
	printf("optval %1.16f \n", optimalvalue);
	//Generovanie problemu----------------------------END----------------------------------

	// Calculate residuals
	double residuals[m];
	for (j = 0; j < m; j++)
		residuals[j] = -b[j];
	//----------------RCDM----------serial===================================---
	for (i = 0; i < n; i++)
		x[i] = 0;
	double tmp1;
	double currentvalue = 0;

	if (0 == 1)// warmstart
	{
		printf("warmstart BEGIN \n");
		for (N = 0; N < (NMAX * n); N++) {
			int idx = (int) (n * (rand() / (RAND_MAX + 1.0)));
			double tmp = 0;
			for (j = 0; j < p; j++) {
				tmp += A_h[idx][j] * residuals[IDX_h[idx][j]];
			}
			tmp = -Li[idx] * (tmp);
			x[idx] += tmp;
			for (j = 0; j < p; j++) {
				residuals[IDX_h[idx][j]] += tmp * A_h[idx][j];
			}
		}
		printf("warmstart END \n");
	}

	int analysisLength = NMAX * n / samplingsize;
	struct optimalityAnalysis analysis[analysisLength];
	int analisisIDX = 0;
	float diff;
	printf("ZACIATOK RIESENIA\n");
	clock_t t1, t2;
	t1 = clock();
	for (N = 0; N < (NMAX * n); N++) {
		//		for (k = 0; k < n; k++) {
		int idx = (int) (n * (rand() / (RAND_MAX + 1.0)));
		double tmp = 0;
		for (j = 0; j < p; j++) {
			//						printf("tmp:%f  A:%f   residual:%f  \n",tmp,A_h[idx][j],residuals[IDX_h[idx][j]]);
			tmp += A_h[idx][j] * residuals[IDX_h[idx][j]];
		}
		//				printf("Li[%d] =  %f; tmp=%f  \n", idx, Li[idx], tmp);
		tmp1 = Li[idx] * (tmp + lambda);
		if (x[idx] > tmp1) {
			tmp = -tmp1;
		} else {
			tmp1 = Li[idx] * (tmp - lambda);
			if (x[idx] < tmp1) {
				tmp = -tmp1;
			} else {
				tmp = -x[idx];
			}
		}
		x[idx] += tmp;
		//update residuals:
		for (j = 0; j < p; j++) {
			residuals[IDX_h[idx][j]] += tmp * A_h[idx][j];
		}
		//				printf("Iteration %d,  x[%d]=%f  \n", N, idx, x[idx]);
		if (N % samplingsize == 0) {

			t2 = clock();
			diff = ((float) t2 - (float) t1) / 1000000.0F;
			printf("%f:%f sec\n", (float) N / n, diff);
			fprintf(fp, "%f:%f sec\n", (float) N / n, diff);
		}
		//		}


	}
	/// SErIAL RCDM =========================================================END
	printf("KONIEC RIESENIA\n");

	//----------------RCDM------- parallel


	fclose(fp);
}
