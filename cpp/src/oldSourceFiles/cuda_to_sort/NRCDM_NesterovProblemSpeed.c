#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

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
void print_float_array(const float *array, size_t len) {
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
	float value;
	int idx;
};

struct optimalityAnalysis {
	float accuracy;
	int nnz;
	int correctnnz;
	int iteration;
};

/* qsort struct comparision function (price float field) */
int struct_cmp_by_value(const void *a, const void *b) {
	struct st_sortingByAbsWithIndex *ia = (struct st_sortingByAbsWithIndex *) a;
	struct st_sortingByAbsWithIndex *ib = (struct st_sortingByAbsWithIndex *) b;
	float aa = ia->value;
	float bb = ib->value;
	if (aa * aa > bb * bb)
		return -1;
	else if (aa * aa < bb * bb)
		return 1;
	else
		return 0;

	/* float comparison: returns negative if b > a
	 and positive if a > b. We multiplied result by 100.0
	 to preserve decimal fraction */

}

solveRCDMSpeedProblem(long int n, long int m, int p, int n_nonzero,
		float sqrtofnonzeros) {
	FILE *fp;
	char buff[200];
	sprintf(buff, "/tmp/ns_m_%d_n_%d_p_%d_nnzofx_%d.csv", m, n, p, n_nonzero);
	fp = fopen(buff, "w");
	srand(1);
	float lambda = 1;
	float rho = 1;

	long int NMAX = 5;
	int N;
	int i, j, k;
	float A_h[n][p]; // host A matrix pointers
	int IDX_h[n][p]; // host Aindex matrix pointers
	float x_optimal[n];
	float optimalvalue = 0;
	float b[m];
	float x[n];
	float tmp;
	float L[n]; // Lipschitz constants
	float Li[n]; // Lipschitz constants
	clock_t t1, t2;
	t1 = clock();

	//Generovanie problemu-------------------------------------------------------------------
	for (i = 0; i < n; i++) {
		int idx = 0;
		for (j = 0; j < p; j++) {
			int notfinished = 1;
			float val = (float) rand() / RAND_MAX;
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
	float v[m];
	float y[m];
	tmp = 0;
	for (j = 0; j < m; j++) {
		v[j] = (float) rand() / RAND_MAX;
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
		float tmp = 0;
		for (j = 0; j < p; j++) {
			dataToSort[i].value += y[IDX_h[i][j]] * A_h[i][j];
		}
	}
	t2 = clock();
	float diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("generovanieMatice:%f\n", diff);
	//Sorting B
	t1 = clock();
	printf("SORTING START\n");
	size_t structs_len = sizeof(dataToSort)
			/ sizeof(struct st_sortingByAbsWithIndex);
	qsort(dataToSort, structs_len, sizeof(struct st_sortingByAbsWithIndex),
			struct_cmp_by_value);
	printf("SORTING END\n");
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("sorting:%f\n", diff);
	t1 = clock();
	for (i = 0; i < n; i++) { // vytvaranie matice A
		int idx = dataToSort[i].idx;
		float alpha = 1;
		x_optimal[idx] = 0;
		if (i < n_nonzero) {

			alpha = (float) abs(1 / dataToSort[idx].value);
			//			printf("alpha = %f \n", alpha);
			x_optimal[idx] = ((float) rand() / RAND_MAX) * rho
					/ (sqrtofnonzeros);
			if (dataToSort[idx].value < 0) {
				x_optimal[idx] = -x_optimal[idx];
			}
		} else if (dataToSort[idx].value > 0.1 || dataToSort[idx].value < -0.1) {
			alpha = (float) abs(1 / dataToSort[idx].value) * (float) rand()
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
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("generovanieproblemu:%f\n", diff);

	//Generovanie problemu----------------------------END----------------------------------

	// Calculate residuals
	float residuals[m];
	for (j = 0; j < m; j++)
		residuals[j] = -b[j];
	//----------------RCDM----------serial===================================---
	for (i = 0; i < n; i++)
		x[i] = 0;
	float tmp1;
	double currentvalue = 0;

	t1 = clock();
	for (N = 0; N < (NMAX * n); N++) {
		int idx = (int) (n * (rand() / (RAND_MAX + 1.0)));
		float tmp = 0;
		for (j = 0; j < p; j++) {
			tmp += A_h[idx][j] * residuals[IDX_h[idx][j]];
		}
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
		for (j = 0; j < p; j++) {
			residuals[IDX_h[idx][j]] += tmp * A_h[idx][j];
		}
	}
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	fprintf(fp, "uplynuty cas:%f\n", diff);
	diff = diff / NMAX;
	fprintf(fp, "uplynuty cas pre iteraciu:%f\n", diff);

	float * A_dev;
	float * L_dev;
	float * x_dev;
	float * b_dev;
	float * lambda_dev;
	fclose(fp);

}

main(int argc, char** argv) {

	int size = 1600000;
	float sqe = 1264.911;
	long int n = 100000000;
	long int m = 100000000;

	printf("Idem riesit prvy priklad");

	int i;

	for (i=0;i<4;i++){


	solveRCDMSpeedProblem(1000000, 10000000, 10, size, sqe);
	solveRCDMSpeedProblem(1000000, 10000000, 100, size, sqe);
	solveRCDMSpeedProblem(1000000, 10000000, 1000, size, sqe);
	solveRCDMSpeedProblem(10000000, 100000000, 10, size, sqe);
	solveRCDMSpeedProblem(10000000, 100000000, 100, size, sqe);
	solveRCDMSpeedProblem(n, n, 10, size, sqe);

	size = size/10;
	sqe = sqe/3.162278;
	}







	//	solveRCDMSpeedProblem(1000000, 10000000, 100, 1600, 40);
	//	solveRCDMSpeedProblem(1000000, 10000000, 100, 160000, 400);
	//	solveRCDMSpeedProblem(1000000, 10000000, 10000, 1600, 40);
	//	solveRCDMSpeedProblem(1000000, 10000000, 10000, 160000, 400);
	//	solveRCDMSpeedProblem(10000000, 100000000, 10, 1600, 40);
	//	solveRCDMSpeedProblem(10000000, 100000000, 100, 1600, 40);
	//	solveRCDMSpeedProblem(100000000, 1000000000, 10, 1600, 40);
}
