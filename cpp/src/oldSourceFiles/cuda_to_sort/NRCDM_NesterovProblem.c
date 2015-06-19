#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

//ulimit -s unlimited

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

long getTotalSystemMemory() {
	long pages = sysconf(_SC_PHYS_PAGES);
	long page_size = sysconf(_SC_PAGE_SIZE);
	return pages * page_size;
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
	long int idx;
};

struct optimalityAnalysis {
	double accuracy;
	int nnz;
	int correctnnz;
	double iteration;
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
	FILE *fp;
//	fp = fopen("/exports/home/s1052689/nesterov.txt", "w");
			fp = fopen("/tmp/nesterov.txt", "w");
	srand(2);
	double lambda = 1;
	double rho = 1;
	long int n = 1000000;
	int n_nonzero = 160000;
	long int m = 10 * n;
	double sqrtofnonzeros = 400;
	int p = 15;

	int NMAX = 60;
	int N;
	int samplingsize = n / 1;
	long int i, j, k;
	printf("texst\n");
	printf("free memory:%d\n", getTotalSystemMemory());
	//	double* AAA;

	printf("Idem alokovat data s obsahom %d\n", n);
	//	AAA = (double*) malloc(n * sizeof(double));

	printf("alokacia poli start\n");
	double A_h[n][p]; // host A matrix pointers
	printf("alokacia A done \n");
	long int IDX_h[n][p]; // host Aindex matrix pointers
	printf("alokacia I done \n");
	printf("alokacia x done \n");
	double optimalvalue = 0;

	int analysisLength = NMAX * n / samplingsize;

	struct optimalityAnalysis* analysis;
	analysis = (struct optimalityAnalysis*) calloc(analysisLength,
			sizeof(struct optimalityAnalysis));

	double tmp;
	printf("alokacia poli END\n");

	printf("free memory:%d\n", getTotalSystemMemory());

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
	printf("free memory:%d\n", getTotalSystemMemory());

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

	//Generovanie problemu----------------------------END----------------------------------
	double * residuals;
	residuals = (double*) calloc(m, sizeof(double));
	for (i = 0; i < m; i++) {
		residuals[i] = -b[i];
	}
	for (i = 0; i < n; i++) {
		for (j = 0; j < p; j++) {
			residuals[IDX_h[i][j]] += x[i] * A_h[i][j];
		}
	}
	double nesterovvalue = 0;
	for (i = 0; i < m; i++) {
		nesterovvalue = residuals[i] * residuals[i];
	}
	nesterovvalue = nesterovvalue / 2;
	for (i = 0; i < n; i++) {
		if (x[i] > 0)
			nesterovvalue += x[i];
		else
			nesterovvalue -= x[i];
	}

	// Calculate residuals
	for (j = 0; j < m; j++)
		residuals[j] = -b[j];
	//----------------RCDM----------serial===================================---
	for (i = 0; i < n; i++)
		x[i] = 0;
	double tmp1;
	double currentvalue = 0;

	int analisisIDX = 0;

	printf("ZACIATOK RIESENIA\n");

	for (N = 0; N < NMAX; N++) {
		for (k = 0; k < n; k++) {
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
			if (k % samplingsize == 0) {
				currentvalue = 0;
				//			print_double_array(&residuals[0],m);
				for (i = 0; i < m; i++) {
					currentvalue += residuals[i] * residuals[i];
				}
				currentvalue = currentvalue * 0.5;
				//			printf("CV:%1.16f\n", currentvalue);

				//			printf(" %1.16f\n",  currentvalue  );
				double normsize = 0;
				for (i = 0; i < n; i++) {
					if (x[i] > 0)
						normsize += lambda * x[i];
					else
						normsize -= lambda * x[i];
				}
				//			print_double_array(&x[0],n);
				currentvalue = currentvalue + normsize;
				//			printf("NZ:%1.16f; :%1.16f\n", currentvalue, normsize);

				//			printf(" %1.16f\n",  currentvalue  );
				analysis[analisisIDX].accuracy = currentvalue;
				analysis[analisisIDX].nnz = 0;
				analysis[analisisIDX].correctnnz = 0;
				analysis[analisisIDX].iteration = N + (double) k / n;
				for (i = 0; i < n; i++) {
					if (x[i] != 0)
						analysis[analisisIDX].nnz++;
//					if (x_optimal[i] != 0 && x[i] != 0)
//						analysis[analisisIDX].correctnnz++;

				}

				printf("%f,%d,%d,%1.16f\n", N + (double) k / n,
						analysis[analisisIDX].nnz,
						analysis[analisisIDX].correctnnz, currentvalue
								- optimalvalue);

				//			printf("%d: nnz %d   correct nnz %d \n", N, analysis[analisisIDX].nnz,analysis[analisisIDX].correctnnz);
				//			printf("%d: f^*=%1.16f,   f(x)=%1.16f \n", N, optimalvalue,
				//					currentvalue);
				//			printf("%d: f(x)-f^*=%1.16f\n", N, currentvalue - optimalvalue);


				analisisIDX++;
			}
		}

	}
	/// SErIAL RCDM =========================================================END
	printf("KONIEC RIESENIA\n");

	currentvalue = 0;
	for (i = 0; i < m; i++) {
		currentvalue = residuals[i] * residuals[i];
	}
	currentvalue = currentvalue / 2;
	for (i = 0; i < n; i++) {
		if (x[i] > 0)
			currentvalue += x[i];
		else
			currentvalue -= x[i];
	}

	printf("Comparison \n");

	//	for (i = 0; i < n; i++) {
	//		if (x[i] > 0 || x[i] < 0 || x_optimal[i] > 0 || x_optimal[i] < 0) {
	//			printf("x[%d] =  %1.10f ;x*[%d]=%1.10f  \n", i, x[i], i,
	//					x_optimal[i]);
	//		}
	//	}

	printf("f^*=%1.16f,   f(x)=%1.16f \n", optimalvalue, currentvalue);
	printf("f(x)-f^*=%1.16f\n", currentvalue - optimalvalue);

	// Skutocna optimalna hodnota dana nesterovym vysledkom


	printf("=====================================\n");
	printf("f^N=%1.16f,   f(x)=%1.16f \n", nesterovvalue, currentvalue);
	printf("f(x)-f^N=%1.16f \n", -nesterovvalue + currentvalue);
	printf("f^N=%1.16f,   f(x)=%1.16f \n", nesterovvalue, currentvalue);
	printf("f(x)-f^N=%1.16f \n", -nesterovvalue + currentvalue);
	//tmp=0;
	//	for (i = 0; i < n; i++) {
	// tmp+=(x[i]-x_optimal[i])*(x[i]-x_optimal[i]);
	//	}
	//	printf("|x-xoptimal|^2 = %1.16f \n",tmp);
	// Allocation arrays on cuda device:

	//VYPISANIE VYSLEDKOV
	double epsilon = 1000000000;
	double minvalue = nesterovvalue;
	for (i = 1; i < analisisIDX; i++) {
		if (analysis[i].accuracy < minvalue) {
			minvalue = analysis[i].accuracy;
		}
	}
	printf("min value: %f\n", minvalue);
	printf("min value: %f\n", minvalue);
	//	i = analisisIDX - 1;
	//	printf("it: %d; eps: %1.16f; nnzofX: %d, basis: %f \n",
	//			analysis[i].iteration, analysis[i].accuracy - minvalue,
	//			analysis[i].nnz, (double) analysis[i].correctnnz / n_nonzero);
	i = 1;

	printf("it: %1.4f; eps: %1.16f; nnzofX: %d, basis: %f \n",
			analysis[i].iteration, analysis[i].accuracy - minvalue,
			analysis[i].nnz, (double) analysis[i].correctnnz / n_nonzero);
	fprintf(fp, "it: %1.4f; eps: %1.16f; nnzofX: %d, basis: %f \n",
			analysis[i].iteration, analysis[i].accuracy - minvalue,
			analysis[i].nnz, (double) analysis[i].correctnnz / n_nonzero);

	printf("analisisIdx:%d", analisisIDX);

	for (i = 1; i < analisisIDX; i++) {
		if (analysis[i].accuracy - minvalue <= epsilon && epsilon >= 0) {
			fprintf(fp, "it: %1.4f; eps: %1.16f; nnzofX: %d, basis: %f \n",
					(double) analysis[i].iteration, analysis[i].accuracy
							- minvalue, analysis[i].nnz,
					(double) analysis[i].correctnnz / n_nonzero);
			printf("it: %1.4f; eps: %1.16f; nnzofX: %d, basis: %f \n",
					(double) analysis[i].iteration, analysis[i].accuracy
							- minvalue, analysis[i].nnz,
					(double) analysis[i].correctnnz / n_nonzero);

			epsilon = epsilon * 0.1;
			printf("epsilon: %f \n", epsilon);
		}
		if (i > 10 && analysis[i].accuracy - minvalue == 0) {
			break;
		}
	}
	//	return 1;
	double * A_dev;
	double * L_dev;
	double * x_dev;
	double * b_dev;
	double * lambda_dev;

	//----------------RCDM------- parallel


	fclose(fp);
}
