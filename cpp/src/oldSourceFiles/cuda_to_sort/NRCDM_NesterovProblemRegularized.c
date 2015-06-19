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

int main(void) {
	srand(1);
	float lambda = 1;
	float rho = 1;
	long int n =    10000;
	int n_nonzero = 1600;
	float sqrtofnonzeros = 40;
float mu=0.0000001;


	long int NMAX = 2000;
	int N;

	int samplingsize = 1;
	long int m = n/10;
	int p = 10;
	int i, j, k;

	float A_h[n][p]; // host A matrix pointers
	int IDX_h[n][p]; // host Aindex matrix pointers
	float x_optimal[n];
	float optimalvalue = 0;
	float b[m];
	float x[n];
	float x_0[n];
	float tmp;
	float L[n]; // Lipschitz constants
	float Li[n]; // Lipschitz constants



	FILE *fp;
	fp = fopen("/tmp/output.csv", "w");




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
	//Sorting B
	printf("SORTING START\n");
	size_t structs_len = sizeof(dataToSort)
			/ sizeof(struct st_sortingByAbsWithIndex);
	qsort(dataToSort, structs_len, sizeof(struct st_sortingByAbsWithIndex),
			struct_cmp_by_value);
	printf("SORTING END\n");
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
	// Skutocna optimalna hodnota dana nesterovym vysledkom
	float residuals[m];
	for (i = 0; i < m; i++) {
			residuals[i] = -b[i];
		}
		for (i = 0; i < n; i++) {
			for (j = 0; j < p; j++) {
				residuals[IDX_h[i][j]] += x_optimal[i] * A_h[i][j];
			}
		}
		float nesterovvalue = 0;
		for (i = 0; i < m; i++) {
			nesterovvalue = residuals[i] * residuals[i];
		}
		nesterovvalue = nesterovvalue / 2;
		for (i = 0; i < n; i++) {
			if (x_optimal[i] > 0)
				nesterovvalue += x_optimal[i];
			else
				nesterovvalue -= x_optimal[i];
		}
		printf("=====================================\n");
optimalvalue=nesterovvalue;
printf("optval %1.16f \n", optimalvalue);


	//Generovanie problemu----------------------------END----------------------------------

	// Calculate residuals

	for (j = 0; j < m; j++)
		residuals[j] = -b[j];
	//----------------RCDM----------serial===================================---
	for (i = 0; i < n; i++)
		x[i] = 0; x_0[i]=0;
	float tmp1;
	float currentvalue = 0;

	if (0==1)// warmstart
	{
		printf("warmstart BEGIN \n");
		for (N = 0; N < (NMAX * n); N++) {
				int idx = (int) (n * (rand() / (RAND_MAX + 1.0)));
				float tmp = 0;
				for (j = 0; j < p; j++) {
					tmp += A_h[idx][j] * residuals[IDX_h[idx][j]];
				}
				tmp = -Li[idx] * (tmp  );
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
	printf("ZACIATOK RIESENIA\n");
	for (N = 0; N < (NMAX * n); N++) {
		//		for (k = 0; k < n; k++) {
		int idx = (int) (n * (rand() / (RAND_MAX + 1.0)));
		float tmp = 0;
		for (j = 0; j < p; j++) {
			//			printf("tmp:%f  A:%f   residual:%f  \n",tmp,A_h[idx][j],residuals[IDX_h[idx][j]]);
			tmp += A_h[idx][j] * residuals[IDX_h[idx][j]];
		}
		//		printf("Li[%d] =  %f; tmp=%f  \n", idx, Li[idx], tmp);
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
		//		printf("Iteration %d,  x[%d]=%f  \n", N, idx, x[idx]);
		if (N % samplingsize == 0) {
			currentvalue = 0;
			for (i = 0; i < m; i++) {
				currentvalue = residuals[i] * residuals[i];
			}
			currentvalue = currentvalue / 2;
			for (i = 0; i < n; i++) {
				if (x[i] > 0)
					currentvalue += lambda*x[i];
				else
					currentvalue -= lambda*x[i];
			}
			analysis[analisisIDX].accuracy = currentvalue;
			analysis[analisisIDX].nnz = 0;
			analysis[analisisIDX].correctnnz = 0;
			analysis[analisisIDX].iteration = N;
			for (i = 0; i < n; i++) {
				if (x[i] != 0)
					analysis[analisisIDX].nnz++;
				if (x_optimal[i] != 0 && x[i] != 0)
					analysis[analisisIDX].correctnnz++;

			}

			fprintf(fp,"%d, %d, %d,%1.16f\n",N, analysis[analisisIDX].nnz,analysis[analisisIDX].correctnnz, currentvalue);

			if (N % n == 0) {
			printf("%d: nnz %d   correct nnz %d \n", N, analysis[analisisIDX].nnz,analysis[analisisIDX].correctnnz);
			printf("%d: f^*=%1.16f,   f(x)=%1.16f \n", N, optimalvalue,
					currentvalue);
			printf("%d: f(x)-f^*=%1.16f\n", N, currentvalue - optimalvalue);
			}

			analisisIDX++;
		}
		//		}


	}
	/// SErIAL RCDM =========================================================END
	printf("KONIEC RIESENIA\n");

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

//		for (i = 0; i < n; i++) {
//			if (x[i] > 0 || x[i] < 0 || x_optimal[i] > 0 || x_optimal[i] < 0) {
//				printf("x[%d] =  %1.10f ;x*[%d]=%1.10f  \n", i, x[i], i,
//						x_optimal[i]);
//			}
//		}

	printf("f^*=%1.16f,   f(x)=%1.16f \n", optimalvalue, currentvalue);
	printf("f(x)-f^*=%1.16f\n", currentvalue - optimalvalue);


	printf("f^N=%1.16f,   f(x)=%1.16f \n", nesterovvalue, currentvalue);
	printf("f(x)-f^N=%1.16f \n", -nesterovvalue + currentvalue);
	//tmp=0;
	//	for (i = 0; i < n; i++) {
	// tmp+=(x[i]-x_optimal[i])*(x[i]-x_optimal[i]);
	//	}
	//	printf("|x-xoptimal|^2 = %1.16f \n",tmp);
	// Allocation arrays on cuda device:

	//VYPISANIE VYSLEDKOV
	float epsilon = 1000;
	float minvalue = nesterovvalue;
	for (i = 1; i < analisisIDX; i++) {
		if (analysis[i].accuracy < minvalue) {
			minvalue = analysis[i].accuracy;
		}
	}
	printf("min value: %f\n", minvalue);
	//	i = analisisIDX - 1;
	//	printf("it: %d; eps: %1.16f; nnzofX: %d, basis: %f \n",
	//			analysis[i].iteration, analysis[i].accuracy - minvalue,
	//			analysis[i].nnz, (float) analysis[i].correctnnz / n_nonzero);

	for (i = 1; i < analisisIDX; i++) {
		if (analysis[i].accuracy - minvalue <= epsilon && epsilon >= 0) {
			printf("it: %1.4f; eps: %1.16f; nnzofX: %d, basis: %f \n",
					(float) analysis[i].iteration / n, analysis[i].accuracy
							- minvalue, analysis[i].nnz,
					(float) analysis[i].correctnnz / n_nonzero);
			epsilon = epsilon * 0.1;
			printf("epsilon: %f \n", epsilon);
		}
		if (i > 10 && analysis[i].accuracy - minvalue == 0) {
			break;
		}
	}

	float * A_dev;
	float * L_dev;
	float * x_dev;
	float * b_dev;
	float * lambda_dev;

	//----------------RCDM------- parallel


	fclose(fp);
}
