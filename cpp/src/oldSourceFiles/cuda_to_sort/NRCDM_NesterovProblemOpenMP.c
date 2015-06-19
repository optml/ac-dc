#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include <time.h>
//ulimit -s unlimited
//gcc -fopenmp -lm  NRCDM_NesterovProblemOpenMP.c -o NRCDM.out && ./NRCDM.out
//
//export OMP_NUM_THREADS=24
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

void print_time_message(clock_t t1, char message[200]) {
	clock_t t2 = clock();
	double diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("%s: %f\n", message, diff);
}

int main(void) {
	clock_t t1, t2;
	t1 = clock();
	FILE *fp;
//				fp = fopen("/exports/home/s1052689/nesterov.txt", "w");
	fp = fopen("/tmp/nesterov.txt", "w");
	srand(2);
	double lambda = 1;
	double diff;
	double rho = 1;
	long int    n = 100000000;
	int n_nonzero = 160000;
	long int m = 5 * n;
	double sqrtofnonzeros = 400;
	int p = 10;
	int NMAX = 90;
	int samplingsize = n/100  ;

	printf("outside thread num is %d\n", omp_get_num_threads());
	int totalThreds=0;
#pragma omp parallel  shared(totalThreds)
	{
		totalThreds = omp_get_num_threads();
//		printf("total threds:%d",totalThreds);
	}
	printf("total threds:%d",totalThreds);
	int N;

	long int i, j, k;
	unsigned int s;
	unsigned int seed[omp_get_num_threads()];
	for (i = 0; i < totalThreds; i++) {
		seed[i] = (int) RAND_MAX*rand();
		if (seed[i]<0)
			seed[i]=-seed[i];
//		printf("seed %d, val %d\n",i, seed[i]);

	}

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

	print_time_message(t1, "alokacia poli");
	//Generovanie problemu-------------------------------------------------------------------
	t1 = clock();
	long int idx;
	int notfinished;
	double val;

#pragma omp parallel private(i,j,idx,notfinished,val,k,s ), shared(IDX_h, A_h,n,m,p)
	{
		s = seed[omp_get_thread_num()];
//		printf("thred %d, val:%f\n",omp_get_thread_num(),(double) rand_r(&s) / RAND_MAX);
//		printf("thred %d, val:%f\n",omp_get_thread_num(),(double) rand_r(&s) / RAND_MAX);
#pragma omp for
		for (i = 0; i < n; i++) {
			idx = 0;

			for (j = 0; j < p; j++) {
			notfinished = 1;
				val = (double) rand_r(&s) / RAND_MAX;
				while (notfinished) {
					notfinished = 0;
					idx = ((long int) ((m) * (rand_r(&s) / (RAND_MAX + 1.0))));
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
	}
//return 1;
	print_time_message(t1, "Matrix B Generated");
	t1 = clock();
	double* y;
	y = (double*) calloc(m, sizeof(double));
	tmp = 0;

#pragma omp parallel private(j,s), shared(y), reduction(+:tmp)
	{
		s = seed[omp_get_thread_num()];
#pragma omp for
		for (j = 0; j < m; j++) {
			y[j] = (double) rand_r(&s) / RAND_MAX;
			tmp += y[j] * y[j];
		}
	}

#pragma omp parallel private(j),shared(y,tmp)
	{
#pragma omp for
		for (j = 0; j < m; j++) {
			y[j] = y[j] / tmp;
		}
	}
	print_time_message(t1, "vector y Generated");
	struct st_sortingByAbsWithIndex* dataToSort;
	dataToSort = (struct st_sortingByAbsWithIndex*) calloc(n,
			sizeof(struct st_sortingByAbsWithIndex));

#pragma omp parallel private(i,j,tmp), shared(dataToSort,A_h,IDX_h,y)
	{
#pragma omp for
		for (i = 0; i < n; i++) {
			dataToSort[i].idx = i;
			tmp = 0;
			for (j = 0; j < p; j++) {
				tmp += y[IDX_h[i][j]] * A_h[i][j];
			}
			dataToSort[i].value = tmp;
		}
	}
	print_time_message(t1, "Struc created");
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

#pragma omp parallel private(i,s), shared(x)
	{
		s = seed[omp_get_thread_num()];
#pragma omp for
		for (i = 0; i < n; i++) {
			x[i] = ((double) rand_r(&s) / RAND_MAX);
		}
	}
	print_time_message(t1, "GENEROVANIE RANDOM X END");
	double alpha = 0;

#pragma omp parallel private(i,alpha,idx,j), shared(x,A_h,dataToSort,sqrtofnonzeros,rho ,n,p)
	{
#pragma omp for
		for (i = 0; i < n; i++) { // vytvaranie matice A
			idx = dataToSort[i].idx;
			alpha = 1;
			if (i < n_nonzero) {
				alpha = (double) abs(1 / dataToSort[idx].value);
				x[idx] = x[idx] * rho / (sqrtofnonzeros);
				if (dataToSort[idx].value < 0) {
					x[idx] = -x[idx];
				}
			} else if (dataToSort[idx].value > 0.1 || dataToSort[idx].value
					< -0.1) {
				alpha = (double) abs(1 / dataToSort[idx].value) * x[idx];
				x[idx] = 0;
			} else {
				x[idx] = 0;
			}
			for (j = 0; j < p; j++) {
				A_h[idx][j] = A_h[idx][j] * alpha;
			}
		}
	}
	print_time_message(t1, "A modified");
	t1 = clock();
	//	print_double_array(&L[0],n);
	//	print_double_array(&Li[0], 10);
	free(dataToSort);
	// Compute Li
	double* Li; // Lipschitz constants
	Li = (double*) calloc(n, sizeof(double));
	print_time_message(t1, "Alokacia Li");
	t1 = clock();
#pragma omp parallel private(i,j), shared(Li,A_h,p,n)
	{
#pragma omp for
		for (i = 0; i < n; i++) {
			Li[i] = 0;
			for (j = 0; j < p; j++) {
				Li[i] += A_h[i][j] * A_h[i][j];
			}
			Li[i] = 1 / Li[i];
		}
	}
	// END compute Li
	print_time_message(t1, "Compute Li");
	t1 = clock();
#pragma omp parallel private(i), shared(y,m), reduction(+:optimalvalue)
	{
#pragma omp for
		for (i = 0; i < m; i++) {
			optimalvalue += y[i] * y[i];
		}
	}
	print_time_message(t1, "OptVal1");
	t1 = clock();
	optimalvalue = optimalvalue * 0.5;
	double* b;
	b = y;

	for (j = 0; j < p; j++) {
		for (i = 0; i < n; i++) {
			b[IDX_h[i][j]] += x[i] * A_h[i][j];
		}
	}
	print_time_message(t1, "OptVal2 serial");
	t1 = clock();

#pragma omp parallel private(i), shared(n,x), reduction(+:optimalvalue)
	{
#pragma omp for
		for (i = 0; i < n; i++) {
			if (x[i] > 0)
				optimalvalue += (x[i]);
			else
				optimalvalue -= x[i];
		}
	}
	print_time_message(t1, "OptVal3");
	t1 = clock();

	printf("optval %1.16f \n", optimalvalue);
	t2 = clock();
	diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("Generating END:%f\n", diff);
	fprintf(fp, "Generating END:%f\n", diff);
	//Generovanie problemu----------------------------END----------------------------------
	double * residuals;
	residuals = (double*) calloc(m, sizeof(double));
	printf("Residuals alocated");
#pragma omp parallel private(i), shared(m,b,residuals)
	{
#pragma omp for
		for (i = 0; i < m; i++) {
			residuals[i] = -b[i];
		}
	}
	printf("Residuals = -b");
	for (i = 0; i < n; i++) {
		for (j = 0; j < p; j++) {
			residuals[IDX_h[i][j]] += x[i] * A_h[i][j];
		}
	}
	printf("Residuals =updated");
	double nesterovvalue = 0;

#pragma omp parallel private(i), shared(m, residuals), reduction(+:nesterovvalue)
	{
#pragma omp for
		for (i = 0; i < m; i++) {
			nesterovvalue += residuals[i] * residuals[i];
		}
	}
	nesterovvalue = nesterovvalue / 2;
#pragma omp parallel private(i), shared(n, x), reduction(+:nesterovvalue)
	{
#pragma omp for
		for (i = 0; i < n; i++) {
			if (x[i] > 0)
				nesterovvalue += x[i];
			else
				nesterovvalue -= x[i];
		}
	}
	// Calculate residuals
#pragma omp parallel private(j,i), shared(m,b,x,n,residuals)
	{
#pragma omp for
		for (j = 0; j < m; j++)
			residuals[j] = -b[j];
#pragma omp for
		for (i = 0; i < n; i++)
			x[i] = 0;
	}
	//----------------RCDM----------serial===================================---
	double tmp1;
	double currentvalue = 0;
printf("RCDM serial");
	int analisisIDX = 0;
	double epsilon = 0;
	currentvalue = 0;
	//			print_double_array(&residuals[0],m);

#pragma omp parallel private(i), shared(residuals,m), reduction(+:currentvalue)
	{
#pragma omp for
		for (i = 0; i < m; i++) {
			currentvalue += residuals[i] * residuals[i];
		}
	}
	currentvalue = currentvalue * 0.5;
	//			printf("CV:%1.16f\n", currentvalue);

	//			printf(" %1.16f\n",  currentvalue  );
	double normsize = 0;
#pragma omp parallel private(i), shared(lambda,n,x), reduction(+:normsize)
	{
#pragma omp for
		for (i = 0; i < n; i++) {
			if (x[i] > 0)
				normsize += lambda * x[i];
			else
				normsize -= lambda * x[i];
		}
	}

	//			print_double_array(&x[0],n);
	epsilon = currentvalue + normsize;


	srand(2);
	printf("ZACIATOK RIESENIA\n");
	t1 = clock();
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

#pragma omp parallel private(i), shared(residuals,m), reduction(+:currentvalue)
				{
#pragma omp for
					for (i = 0; i < m; i++) {
						currentvalue += residuals[i] * residuals[i];
					}
				}
				currentvalue = currentvalue * 0.5;
				//			printf("CV:%1.16f\n", currentvalue);

				//			printf(" %1.16f\n",  currentvalue  );
				double normsize = 0;
#pragma omp parallel private(i), shared(lambda,n,x), reduction(+:normsize)
				{
#pragma omp for
					for (i = 0; i < n; i++) {
						if (x[i] > 0)
							normsize += lambda * x[i];
						else
							normsize -= lambda * x[i];
					}
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
				t2 = clock();
				diff = ((float) t2 - (float) t1) / 1000000.0F;

				printf("%f,%d,%d,%1.16f,TIME:%f\n",
						analysis[analisisIDX].iteration,
						analysis[analisisIDX].nnz,
						analysis[analisisIDX].correctnnz, currentvalue
								- optimalvalue, diff);

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
#pragma omp parallel private(i), shared(residuals,m), reduction(+:currentvalue)
	{
#pragma omp for
		for (i = 0; i < m; i++) {
			currentvalue = residuals[i] * residuals[i];
		}
	}
	currentvalue = currentvalue / 2;
#pragma omp parallel private(i), shared(x,n), reduction(+:currentvalue)
	{
#pragma omp for
		for (i = 0; i < n; i++) {
			if (x[i] > 0)
				currentvalue += x[i];
			else
				currentvalue -= x[i];
		}
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

	double minvalue = nesterovvalue;
	for (i = 1; i < analisisIDX; i++) {
		if (analysis[i].accuracy < minvalue) {
			minvalue = analysis[i].accuracy;
		}
	}
	printf("min value: %f\n", minvalue);
	fprintf(fp,"min value: %f\n", minvalue);
	//	i = analisisIDX - 1;
	//	printf("it: %d; eps: %1.16f; nnzofX: %d, basis: %f \n",
	//			analysis[i].iteration, analysis[i].accuracy - minvalue,
	//			analysis[i].nnz, (double) analysis[i].correctnnz / n_nonzero);
	printf("F(x_0): %f\n", epsilon);
	fprintf(fp,"F(x_0): %f\n", epsilon);

	epsilon=epsilon-minvalue;
	fprintf(fp,"F(x_0)-F^*: %f\n", epsilon);
	epsilon=epsilon*0.1;
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

	fclose(fp);
		fp = fopen("/tmp/nesterov_time.txt", "w");
//	fp = fopen("/exports/home/s1052689/nesterov_time.txt", "w");

	// Calculate residuals
#pragma omp parallel private(j,i), shared(m,b,x,n,residuals)
	{
#pragma omp for
		for (j = 0; j < m; j++)
			residuals[j] = -b[j];
#pragma omp for
		for (i = 0; i < n; i++)
			x[i] = 0;
	}
	//----------------RCDM----------serial===================================---
	srand(2);
	printf("ZACIATOK RIESENIA\n");
	t1 = clock();
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
				t2 = clock();
				diff = ((float) t2 - (float) t1) / 1000000.0F;

				printf("%f,TIME:%f\n", N + (double) k / n, diff);
				fprintf(fp, "%f,TIME:%f\n", N + (double) k / n, diff);

			}
		}

	}
	/// SErIAL RCDM =========================================================END
	printf("KONIEC RIESENIA\n");

	//	return 1;
	double * A_dev;
	double * L_dev;
	double * x_dev;
	double * b_dev;
	double * lambda_dev;

	//----------------RCDM------- parallel


	fclose(fp);
}
