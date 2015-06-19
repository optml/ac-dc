//ulimit -s unlimited
//gcc -lm -std=c99 NRCDML1RegLog.c && ./a.out

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#define MAXFEATURES  30000000
#define MAXINSTANCES 50000000

double CS[] = { 0.031250, 0.044194, 0.062500, 0.088388, 0.125000, 0.176777,
		0.250000, 0.353553, 0.500000, 0.707107, 1.000000, 1.414214, 2.000000,
		2.828427, 4.000000, 5.656854, 8.000000, 11.313708, 16.000000,
		22.627417, 32.000000 };
int cscount = 21;

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

double NRCDM_L1L2SVM(double *A, int*R_Idx, int*C_Idx, int*C_Count, int n,
		int m, int nnz, double* y, double*w, double lambda, double C,
		double* Li, int NMAX, double optimalvalue, int loging) {
	FILE *fp;
	fp = fopen("/tmp/L1regularizeL2.csv", "w");

	double xi[m];
	int i, j;
	long int N;
	for (j = 0; j < m; j++) {
		xi[j] = 0;
	}
	printf("NRCDM1\n");

	printf("n=%d, m=%d \n", n, m);

	printf("NRCDM1-b\n");

	for (i = 0; i < n; i++) {
		//		printf("%d  , %d \n",i, C_Count[i]);
		for (j = 0; j < C_Count[i]; j++) {
			//			printf("X");
			//			printf("DATA: %d\n", R_Idx[C_Idx[i] + j]);
			//			printf("DATA: %f\n", y[R_Idx[C_Idx[i] + j]]);
			//			printf("DATA: %f \n", A[C_Idx[i] + j]);
			//			printf("DATA: %f \n",w[i]);
			//			printf("DATA: %d\n", C_Idx[i]);

			xi[R_Idx[C_Idx[i] + j]] -= y[R_Idx[C_Idx[i] + j]] * A[C_Idx[i] + j]
					* w[i];
		}
	}
	printf("NRCDM2\n");
	double partialDetivative, delta, tmp, value;
	for (N = 0; N < n * NMAX; N++) { //
		if (N < 10) {
			//			printf("NRCDM-KOLECKO\n");
		}
		int idx = (int) (n * (rand() / (RAND_MAX + 1.0)));
		//	    %minimize  alfa*d + 0.5 * L*d^2 + lambda |x+d|
		partialDetivative = 0;
		for (j = C_Idx[idx]; j < C_Idx[idx] + C_Count[idx]; j++) {
			i = R_Idx[j];
			if (xi[i] > -1) {
				partialDetivative += (1 + xi[i]) * (y[i] * A[j]);
			}
		}
		partialDetivative = -2 * C * partialDetivative;
		tmp = Li[idx] * (partialDetivative + lambda);
		if (w[idx] > tmp) {
			delta = -tmp;
		} else {
			tmp = Li[idx] * (partialDetivative - lambda);
			if (w[idx] < tmp) {
				delta = -tmp;
			} else {
				delta = -w[idx];
			}
		}
		w[idx] += delta;
		for (j = C_Idx[idx]; j < C_Idx[idx] + C_Count[idx]; j++) {
			xi[R_Idx[j]] -= y[R_Idx[j]] * A[j] * delta;
		}
		if (N % (n) == 0) {
			int nnzcount = 0;
			value = 0;
			for (j = 0; j < m; j++) {
				if (xi[j] > -1)
					value += (1 + xi[j]) * (1 + xi[j]);
			}
			value = value * C;
			for (i = 0; i < n; i++) {
				if (w[i] != 0)
					nnzcount++;
				if (w[i] > 0)
					value += w[i];
				else
					value -= w[i];
			}
			printf("Iteracia:%d, value:%1.16f, nnz:%d, epsilon: %1.16f\n", N,
					value, nnzcount, value - optimalvalue);
		}
	}
	printf("NRCDM3\n");
	fclose(fp);
}

double NRCDM_L1Log(double *A, int*R_Idx, int*C_Idx, int*C_Count, int n, int m,
		int nnz, double* y, double*w, double lambda, double C, double* Li,
		int NMAX, double optimalvalue, int loging) {
	FILE *fp;
	fp = fopen("/tmp/L1regularizeLogistics.csv", "w");
	double xi[m];
	int i, j;
	long int N;
	for (j = 0; j < m; j++) {
		xi[j] = 0;
	}
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count[i]; j++) {
			xi[R_Idx[C_Idx[i] + j]] -= y[R_Idx[C_Idx[i] + j]] * A[C_Idx[i] + j]
					* w[i];
		}
	}
	//	print_double_array(&xi[0], m);
	double partialDetivative, delta, tmp, value;
	printf("Zacinam kolecko %d  %d \n", NMAX, n);
	for (N = 0; N < NMAX * n; N++) {
		int idx = (int) (n * (rand() / (RAND_MAX + 1.0)));
		//	    %minimize  alfa*d + 0.5 * L*d^2 + lambda |x+d|
		partialDetivative = 0;
		for (j = C_Idx[idx]; j < C_Idx[idx] + C_Count[idx]; j++) {
			i = R_Idx[j];
			tmp = exp(xi[i]);
			partialDetivative -= tmp / (1 + tmp) * (y[i] * A[j]);
			partialDetivative = C * partialDetivative;
		}

		tmp = Li[idx] * (partialDetivative + lambda);
		if (w[idx] > tmp) {
			delta = -tmp;
		} else {
			tmp = Li[idx] * (partialDetivative - lambda);
			if (w[idx] < tmp) {
				delta = -tmp;
			} else {
				delta = -w[idx];
			}
		}
		w[idx] += delta;
		for (j = C_Idx[idx]; j < C_Idx[idx] + C_Count[idx]; j++) {
			xi[R_Idx[j]] -= y[R_Idx[j]] * A[j] * delta;
		}
		if (N % (n) == 0) {
			int nnzcount = 0;
			value = 0;
			for (i = 0; i < n; i++) {
				if (w[i] != 0)
					nnzcount++;
				if (w[i] > 0)
					value += w[i];
				else
					value -= w[i];
			}
			for (j = 0; j < m; j++)
				value += log(1 + exp(xi[j]));
			value = value * C;
			printf("Iteracia:%d, value:%f, nnz:%d, epsilon: %f\n", N, value,
					nnzcount, value - optimalvalue);
		}
	}
	fclose(fp);
}

double NRCDM_SR(double *A, int*R_Idx, int*C_Idx, int*C_Count, int n, int m,
		int nnz, double* b, double*x, double lambda, double* Li, int NMAX,
		double optimalvalue, int log) {
	double residuals[m];
	double value = 0;
	int i, j, N;
	int sample = n / 100;
	if (log = 0)
		sample = n;
	FILE *fp;
	fp = fopen("/exports/home/s1052689/nesterov.txt", "w");
	//	fp = fopen("/tmp/sparseregression.csv", "w");
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
void computeLipsitzConstantsForL1RegLog(double** L, double ** Li, double *A,
		int*R_Idx, int*C_Idx, int*C_Count, int n, int nnz, double C) {
	int i, j;
	*L = (double *) malloc(n * sizeof(double));
	*Li = (double *) malloc(n * sizeof(double));
	for (i = 0; i < n; i++) {
		(*L)[i] = 0;
		for (j = 0; j < C_Count[i]; j++) {
			(*L)[i] += A[C_Idx[i] + j] * A[C_Idx[i] + j];
		}
		(*L)[i] = C * (*L)[i] * 0.25;
		if ((*L)[i] > 0) {
			(*Li)[i] = 1 / (*L)[i];
		} else {
			(*Li)[i] = 0;
		}
	}
}
void computeLipsitzConstantsForL1RegL2(double** L, double ** Li, double *A,
		int*R_Idx, int*C_Idx, int*C_Count, int n, int nnz, double C) {
	int i, j;
	*L = (double *) malloc(n * sizeof(double));
	*Li = (double *) malloc(n * sizeof(double));
	for (i = 0; i < n; i++) {
		(*L)[i] = 0;
		for (j = 0; j < C_Count[i]; j++) {
			(*L)[i] += A[C_Idx[i] + j] * A[C_Idx[i] + j];
		}
		(*L)[i] = C * (*L)[i] * 2;
		if ((*L)[i] > 0) {
			(*Li)[i] = 1 / (*L)[i];
		} else {
			(*Li)[i] = 0;
		}
	}

}

void loadDataFromFile(char filepath[200], double ** A, int** R_Idx, int**C_Idx,
		int **C_Count, double** y, int* n_out, int* m_out, int * nnzout,
		int maxpocetzaznamov, int biasTerm) {
	FILE *fp;
	fp = fopen(filepath, "r");
	int pocetZazanmov = 0;
	int i, j, n, m;
	int *features;
	printf("Idem alokovat fatures block\n");
	features = (int*) malloc(MAXFEATURES * sizeof(int));
	printf("Fatures block allocated\n");
	for (i = 0; i < MAXFEATURES; i++) {
		features[i] = 0;
	}
	printf("Fatures block allocation tested \n");
	char strin[80];
	int nnz = 0;
	while (!feof(fp) && pocetZazanmov < maxpocetzaznamov) {
		fscanf(fp, "%s", &strin);
		//		printf("Load : %s\n", strin);
		if (strlen(strin) < 2) {
			pocetZazanmov++;

			//			printf("Pocet nacianych zaznamov : %d\n", pocetZazanmov);
		} else {
			int index;
			float value;
			sscanf(strin, "%d:%f", &index, &value);
			//			printf("index %d, value %f\n", index - 1, value);
			features[index - 1] = features[index - 1] + 1;
			nnz++;
		}
	}
	if (biasTerm == 1) {
		printf("Pocet zanzamov:%d, nnz:%d\n", pocetZazanmov, nnz);
		nnz += pocetZazanmov;
		printf("Pocet zanzamov:%d, nnz:%d\n", pocetZazanmov, nnz);
	}
	printf("zisteny pocet blokov - done\n");
	*nnzout = nnz;
	n = 0;
	j = 0;
	for (i = 0; i < MAXFEATURES; i++) {
		//		if (features[i] > n) {
		//			n = features[i];
		////			printf("biggest count %d, feature %d\n", n, i);
		//		}

		if (features[i] > 0) {
			j = i;
		}
	}

	n = j;
	if (biasTerm == 1) {
		n++;
	}
	printf("LOAD DATA: STEP 1\n");
	m = pocetZazanmov;
	*C_Idx = (int *) malloc(n * sizeof(int));
	*C_Count = (int *) malloc(n * sizeof(int));
	*A = (double *) malloc(nnz * sizeof(double));
	*y = (double *) malloc(m * sizeof(double));
	*R_Idx = (int *) malloc(nnz * sizeof(int));
	j = 0;
	for (i = 0; i < n; i++) { //prepocet zaznamov
		(*C_Idx)[i] = j;
		j += features[i];
		(*C_Count)[i] = 0;
	}
	rewind(fp);
	j = -1;
	float value;
	nnz = 0;
	pocetZazanmov = 0;
	while (!feof(fp) && pocetZazanmov < maxpocetzaznamov) {
		fscanf(fp, "%s", &strin);
		if (strlen(strin) < 2) {
			sscanf(strin, "%f", &value);
			j++;
			(*y)[j] = value;
			pocetZazanmov++;

		} else {
			int index;
			sscanf(strin, "%d:%f", &index, &value);
			index = index - 1;
			i = (*C_Idx)[index] + (*C_Count)[index];
			(*A)[i] = value;
			(*R_Idx)[i] = j;
			(*C_Count)[index]++;
			nnz++;
		}
	}
	printf("Pocet zanzamov:%d, nnz:%d\n", pocetZazanmov, nnz);
	printf("LOAD DATA: STEP 2\n");
	if (biasTerm == 1) {
		for (j = 1; j < pocetZazanmov; j++) {
			//			printf("j:%d, nnz:%d\n", nnz - pocetZazanmov + j, nnz);
			(*A)[nnz - pocetZazanmov + j] = 1;
			(*R_Idx)[nnz - pocetZazanmov + j] = j;
		}
		printf("LOAD DATA: STEP 3\n");
		(*C_Count)[n - 1] = pocetZazanmov;
	}
	*n_out = n;
	*m_out = m;
	fclose(fp);
}

void RandomSparseRegression() {
	int n = 100000000;
	int m = 10 * n;
	int pMin = 10;
	int pMax = 10;
	double lambda = 1;
	double C = 1;

	int NMax = 100;

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
	double value = 0;
	//	for (i = 0; i < n; i++)
	//		x[i] = 0;
	//	value = NRCDM_SR(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, b, x, lambda,
	//			Li, NMax * 100, value, 0);
	for (i = 0; i < n; i++)
		x[i] = 0;
	value = NRCDM_SR(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, b, x, lambda,
			Li, NMax, value, 1);

}

void exportDataForSVM(char filenameA[200], char filenamey[200], double *A,
		int* R_Idx, int* C_Idx, int* C_Count, double * y, int n, int m) {
	int i, j;
	FILE *fp;
	fp = fopen(filenameA, "w");
	printf("start print A\n");
	for (i = 0; i < n; i++) {
		for (j = 0; j < C_Count[i]; j++) {
			fprintf(fp, "%d,%d,%1.16f\n", R_Idx[C_Idx[i] + j], i, A[C_Idx[i]
					+ j]);
		}
	}
	fclose(fp);

	fp = fopen(filenamey, "w");
	printf("start print y\n");
	for (i = 0; i < m; i++) {
		fprintf(fp, "%d,%1.16f\n", i, y[i]);
	}
	fclose(fp);

}
double crossvalidationForL1L2SVM(double *A, int*R_Idx, int*C_Idx, int*C_Count,
		int n, int m, double* y, double*w, int instanceFrom, int instanceTo) {
	double fitted_y[m];
	int i, j;
	for (i = 0; i < m; i++) {
		fitted_y[i] = 0;
	}
	printf("validation - matrix product TODO:\n");
	for (i = 0; i < n; i++) {
		//		printf("%d  , %d \n", i, C_Count[i]);
		for (j = 0; j < C_Count[i]; j++) {
			//						printf("X");
			//						printf("DATA: %f \n",w[i]);
			//						printf("DATA: %d\n", C_Idx[i]);
			//						printf("DATA:R %d\n", R_Idx[C_Idx[i] + j]);
			//						printf("DATA: %f\n", fitted_y[R_Idx[C_Idx[i] + j]]);
			//						printf("DATA: %f \n", A[C_Idx[i] + j]);
			//						printf("VAL LOOP \n");
			//						printf("%f %f %d  %d\n",A[C_Idx[i] + j],w[i],C_Idx[i],R_Idx[C_Idx[i] + j]);
			fitted_y[R_Idx[C_Idx[i] + j]] += A[C_Idx[i] + j] * w[i];
		}
	}
	printf("validation - matrix product DONE:\n");
	int together = 0;
	int correct = 0;
	for (i = instanceFrom; i < instanceTo; i++) {
		//		printf("is:%f,shouldBe:%f\n",y[i],fitted_y[i]);
		together++;
		if (fitted_y[i] > 0 && y[i] > 0) {
			correct++;
		}
		if (fitted_y[i] < 0 && y[i] < 0) {
			correct++;
		}
	}
	return (double) correct / together;
}

void L1regularizedLogisticRegression() {
	int n;
	int m;
	double lambda = 1;
	double C = 1;
	int NMax = 100;
	double* A_h;
	int * C_Idx_h;
	int * R_Idx_h;
	int * C_Count_h;
	int nnz, i, j, k, l;
	double* y;
	double *L;
	double *Li;
	double* A_test;
	int * C_Idx_test;
	int * R_Idx_test;
	int * C_Count_test;
	double* y_test;
	int n_test;
	int m_test;
	int nnz_test;

	loadDataFromFile("/document/rcv1_train.binary", &A_test, &R_Idx_test,
			&C_Idx_test, &C_Count_test, &y_test, &n_test, &m_test, &nnz_test,
			MAXINSTANCES, 1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test, m_test, nnz_test);

	int trainingset = m_test * 9 / 10;
	loadDataFromFile("/document/rcv1_train.binary", &A_h, &R_Idx_h, &C_Idx_h,
			&C_Count_h, &y, &n, &m, &nnz, trainingset, 1);
	printf("all features:%d , all data:%d, nnz:%d\n", n, m, nnz);

	double* A_test_final;
	int * C_Idx_test_final;
	int * R_Idx_test_final;
	int * C_Count_test_final;
	double* y_test_final;
	int n_test_final;
	int m_test_final;
	int nnz_test_final;

	loadDataFromFile("/document/rcv1_test.binary", &A_test_final,
			&R_Idx_test_final, &C_Idx_test_final, &C_Count_test_final,
			&y_test_final, &n_test_final, &m_test_final, &nnz_test_final,
			MAXINSTANCES, 1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test_final,
			m_test_final, nnz_test_final);

	double w[n];
	double value = 0;
	for (i = 0; i < n; i++)
		w[i] = 0;

	double crossvalidationValue;
	double finalTestValue;
	double finalvalidation[cscount];
	double validation[cscount];

	for (i = 0; i < cscount; i++) {
		if (i > 0) {
			free(L);
			free(Li);
		}
		C = CS[i];
		computeLipsitzConstantsForL1RegLog(&L, &Li, A_h, R_Idx_h, C_Idx_h,
				C_Count_h, n, nnz, C);
		value = NRCDM_L1Log(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, y, w,
				lambda, C, Li, NMax, value, 0);
		crossvalidationValue = crossvalidationForL1L2SVM(A_test, R_Idx_test,
				C_Idx_test, C_Count_test, n_test, m_test, y_test, w,
				trainingset, m_test);
		finalTestValue = crossvalidationForL1L2SVM(A_test_final,
				R_Idx_test_final, C_Idx_test_final, C_Count_test_final,
				n_test_final, m_test_final, y_test_final, w, 0, m_test_final);
		printf("C:%f, crossvalidation: %f, final: %f \n", C,
				crossvalidationValue, finalTestValue);
		validation[i] = crossvalidationValue;
		finalvalidation[i] = finalTestValue;
		printf("C:%f, crossvalidation: %f \n", C, crossvalidationValue);
		validation[i] = crossvalidationValue;
	}

	for (i = 0; i < cscount; i++) {
		printf("C:%f, crossvalidation: %f; final: %f \n", CS[i], validation[i],
				finalvalidation[i]);
	}

	//	exportDataForSVM("/document/rcv1_matrixA.csv", "/document/rcv1_vectorY.csv", A_h,
	//			 R_Idx_h, C_Idx_h, C_Count_h, y, n,  m);
}

void L1regularizedL2() {
	int n;
	int m;
	double lambda = 1;
	double C = 1;
	int NMax = 10000;
	double* A_h;
	int * C_Idx_h;
	int * R_Idx_h;
	int * C_Count_h;
	int nnz, i, j, k, l;
	double* y;
	double *L;
	double *Li;

	double* A_test;
	int * C_Idx_test;
	int * R_Idx_test;
	int * C_Count_test;
	double* y_test;
	int n_test;
	int m_test;
	int nnz_test;

	loadDataFromFile("/document/rcv1_train.binary", &A_test, &R_Idx_test,
			&C_Idx_test, &C_Count_test, &y_test, &n_test, &m_test, &nnz_test,
			MAXINSTANCES, 1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test, m_test, nnz_test);

	int trainingset = m_test * 9 / 10;
	loadDataFromFile("/document/rcv1_train.binary", &A_h, &R_Idx_h, &C_Idx_h,
			&C_Count_h, &y, &n, &m, &nnz, trainingset, 1);
	printf("all features:%d , all data:%d, nnz:%d\n", n, m, nnz);

	double* A_test_final;
	int * C_Idx_test_final;
	int * R_Idx_test_final;
	int * C_Count_test_final;
	double* y_test_final;
	int n_test_final;
	int m_test_final;
	int nnz_test_final;

	loadDataFromFile("/document/rcv1_test.binary", &A_test_final,
			&R_Idx_test_final, &C_Idx_test_final, &C_Count_test_final,
			&y_test_final, &n_test_final, &m_test_final, &nnz_test_final,
			MAXINSTANCES, 1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test_final,
			m_test_final, nnz_test_final);

	double w[n];
	for (i = 0; i < n; i++)
		w[i] = 0;

	double value = 0;
	double crossvalidationValue;
	double finalTestValue;
	double finalvalidation[cscount];
	double validation[cscount];
	for (i = 0; i < n; i++)
		w[i] = 0;
	for (i = 0; i < cscount; i++) {
		if (i > 0) {
			free(L);
			free(Li);
		}
		C = CS[i];
		computeLipsitzConstantsForL1RegL2(&L, &Li, A_h, R_Idx_h, C_Idx_h,
				C_Count_h, n, nnz, C);
		value = NRCDM_L1L2SVM(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, y,
				w, lambda, C, Li, NMax, value, 0);
		crossvalidationValue = crossvalidationForL1L2SVM(A_test, R_Idx_test,
				C_Idx_test, C_Count_test, n_test, m_test, y_test, w,
				trainingset, m_test);
		finalTestValue = crossvalidationForL1L2SVM(A_test_final,
				R_Idx_test_final, C_Idx_test_final, C_Count_test_final,
				n_test_final, m_test_final, y_test_final, w, 0, m_test_final);

		printf("C:%f, crossvalidation: %f, final: %f \n", C,
				crossvalidationValue, finalTestValue);
		validation[i] = crossvalidationValue;
		finalvalidation[i] = finalTestValue;
	}

	for (i = 0; i < cscount; i++) {
		printf("C:%f, crossvalidation: %f; final: %f \n", CS[i], validation[i],
				finalvalidation[i]);
	}

}

void L1regularizedLogisticRegression_KDDB() {
	int n;
	int m;
	double lambda = 1;
	double C = 1;
	int NMax = 10;
	double* A_h;
	int * C_Idx_h;
	int * R_Idx_h;
	int * C_Count_h;
	int nnz, i, j, k, l;
	double* y;
	double *L;
	double *Li;
	double* A_test;
	int * C_Idx_test;
	int * R_Idx_test;
	int * C_Count_test;
	double* y_test;
	int n_test;
	int m_test;
	int nnz_test;

	loadDataFromFile("/home/s1052689/kddb", &A_test, &R_Idx_test, &C_Idx_test,
			&C_Count_test, &y_test, &n_test, &m_test, &nnz_test, MAXINSTANCES,
			1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test, m_test, nnz_test);

	int trainingset = 0;
	loadDataFromFile("/home/s1052689/kddb", &A_h, &R_Idx_h, &C_Idx_h,
			&C_Count_h, &y, &n, &m, &nnz, MAXINSTANCES, 1);
	printf("all features:%d , all data:%d, nnz:%d\n", n, m, nnz);

	double* A_test_final;
	int * C_Idx_test_final;
	int * R_Idx_test_final;
	int * C_Count_test_final;
	double* y_test_final;
	int n_test_final;
	int m_test_final;
	int nnz_test_final;

	loadDataFromFile("/home/s1052689/kddb.t", &A_test_final, &R_Idx_test_final,
			&C_Idx_test_final, &C_Count_test_final, &y_test_final,
			&n_test_final, &m_test_final, &nnz_test_final, MAXINSTANCES, 1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test_final,
			m_test_final, nnz_test_final);

	double w[n];
	double value = 0;
	for (i = 0; i < n; i++)
		w[i] = 0;

	double crossvalidationValue;
	double finalTestValue;
	double finalvalidation[cscount];
	double validation[cscount];

	for (i = 0; i < cscount; i++) {
		if (i > 0) {
			free(L);
			free(Li);
		}
		C = CS[i];
		computeLipsitzConstantsForL1RegLog(&L, &Li, A_h, R_Idx_h, C_Idx_h,
				C_Count_h, n, nnz, C);
		printf("Idem ucit \n");
		value = NRCDM_L1Log(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, y, w,
				lambda, C, Li, NMax, value, 0);
		printf("Naucena siet \n");
		crossvalidationValue = crossvalidationForL1L2SVM(A_test, R_Idx_test,
				C_Idx_test, C_Count_test, n_test, m_test, y_test, w,
				trainingset, m_test);
		finalTestValue = crossvalidationForL1L2SVM(A_test_final,
				R_Idx_test_final, C_Idx_test_final, C_Count_test_final,
				n_test_final, m_test_final, y_test_final, w, 0, m_test_final);
		printf("C:%f, crossvalidation: %f, final: %f \n", C,
				crossvalidationValue, finalTestValue);
		validation[i] = crossvalidationValue;
		finalvalidation[i] = finalTestValue;
		printf("C:%f, crossvalidation: %f \n", C, crossvalidationValue);
		validation[i] = crossvalidationValue;
	}

	for (i = 0; i < cscount; i++) {
		printf("C:%f, crossvalidation: %f; final: %f \n", CS[i], validation[i],
				finalvalidation[i]);
	}

	//	exportDataForSVM("/document/rcv1_matrixA.csv", "/document/rcv1_vectorY.csv", A_h,
	//			 R_Idx_h, C_Idx_h, C_Count_h, y, n,  m);
}

void L1regularizedL2_KDDB() {
	int n;
	int m;
	double lambda = 1;
	double C = 1;
	int NMax = 50;
	double* A_h;
	int * C_Idx_h;
	int * R_Idx_h;
	int * C_Count_h;
	int nnz, i, j, k, l;
	double* y;
	double *L;
	double *Li;

	double* A_test;
	int * C_Idx_test;
	int * R_Idx_test;
	int * C_Count_test;
	double* y_test;
	int n_test;
	int m_test;
	int nnz_test;

	printf("Idem nacitavat data!\n");

	loadDataFromFile("/home/s1052689/kddb", &A_test, &R_Idx_test, &C_Idx_test,
			&C_Count_test, &y_test, &n_test, &m_test, &nnz_test, MAXINSTANCES,
			1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test, m_test, nnz_test);

	int trainingset = m_test;
	loadDataFromFile("/home/s1052689/kddb", &A_h, &R_Idx_h, &C_Idx_h,
			&C_Count_h, &y, &n, &m, &nnz, MAXINSTANCES, 1);
	printf("all features:%d , all data:%d, nnz:%d\n", n, m, nnz);

	printf("Training data loades!\n");

	double* A_test_final;
	int * C_Idx_test_final;
	int * R_Idx_test_final;
	int * C_Count_test_final;
	double* y_test_final;
	int n_test_final;
	int m_test_final;
	int nnz_test_final;

	loadDataFromFile("/home/s1052689/kddb.t", &A_test_final, &R_Idx_test_final,
			&C_Idx_test_final, &C_Count_test_final, &y_test_final,
			&n_test_final, &m_test_final, &nnz_test_final, MAXINSTANCES, 1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test_final,
			m_test_final, nnz_test_final);

	double w[n];
	for (i = 0; i < n; i++)
		w[i] = 0;

	double value = 0;
	double crossvalidationValue;
	double finalTestValue;
	double finalvalidation[cscount];
	double validation[cscount];
	for (i = 0; i < n; i++)
		w[i] = 0;
	for (i = 0; i < cscount; i++) {
		if (i > 0) {
			free(L);
			free(Li);
		}
		C = CS[i];
		printf("idem pocitat Li\n");
		computeLipsitzConstantsForL1RegL2(&L, &Li, A_h, R_Idx_h, C_Idx_h,
				C_Count_h, n, nnz, C);
		printf("idem ucit model \n");
		value = NRCDM_L1L2SVM(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, y,
				w, lambda, C, Li, NMax, value, 0);
		printf("idem ctosvalidation \n");
		crossvalidationValue = crossvalidationForL1L2SVM(A_test, R_Idx_test,
				C_Idx_test, C_Count_test, n_test, m_test, y_test, w, 1, m_test);
		printf("idem fiinal test robit \n");
		finalTestValue = crossvalidationForL1L2SVM(A_test_final,
				R_Idx_test_final, C_Idx_test_final, C_Count_test_final,
				n_test_final, m_test_final, y_test_final, w, 0, m_test_final);
		//		crossvalidationValue = finalTestValue;

		printf("C:%f, crossvalidation: %f, final: %f \n", C,
				crossvalidationValue, finalTestValue);
		validation[i] = crossvalidationValue;
		finalvalidation[i] = finalTestValue;
		print_double_array(&w[0], 100);
	}

	for (i = 0; i < cscount; i++) {
		printf("C:%f, crossvalidation: %f; final: %f \n", CS[i], validation[i],
				finalvalidation[i]);
	}

}

void porovnaniePodlaIteracii() {
	int n;
	int m;
	double lambda = 1;
	double C = 1;
	int NMax = 100;
	double* A_h;
	int * C_Idx_h;
	int * R_Idx_h;
	int * C_Count_h;
	int nnz, i, j, k, l;
	double* y;
	double *L;
	double *Li;

	double* A_test;
	int * C_Idx_test;
	int * R_Idx_test;
	int * C_Count_test;
	double* y_test;
	int n_test;
	int m_test;
	int nnz_test;

	loadDataFromFile("/document/rcv1_train.binary", &A_test, &R_Idx_test,
			&C_Idx_test, &C_Count_test, &y_test, &n_test, &m_test, &nnz_test,
			MAXINSTANCES, 1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test, m_test, nnz_test);

	int trainingset = m_test * 9 / 10;
	loadDataFromFile("/document/rcv1_train.binary", &A_h, &R_Idx_h, &C_Idx_h,
			&C_Count_h, &y, &n, &m, &nnz, trainingset, 1);
	printf("all features:%d , all data:%d, nnz:%d\n", n, m, nnz);

	double* A_test_final;
	int * C_Idx_test_final;
	int * R_Idx_test_final;
	int * C_Count_test_final;
	double* y_test_final;
	int n_test_final;
	int m_test_final;
	int nnz_test_final;

	loadDataFromFile("/document/rcv1_test.binary", &A_test_final,
			&R_Idx_test_final, &C_Idx_test_final, &C_Count_test_final,
			&y_test_final, &n_test_final, &m_test_final, &nnz_test_final,
			MAXINSTANCES, 1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test_final,
			m_test_final, nnz_test_final);

	double w[n];
	for (i = 0; i < n; i++)
		w[i] = 0;

	double value = 0;
	double crossvalidationValue;
	double finalTestValue;
	double finalvalidation[NMax];
	double validation[NMax];
	double finalvalidationLog[NMax];
	double validationLog[NMax];
	int nnzofWLog[NMax];
	int nnzofW[NMax];
	computeLipsitzConstantsForL1RegL2(&L, &Li, A_h, R_Idx_h, C_Idx_h,
			C_Count_h, n, nnz, C);
	for (i = 0; i < n; i++)
		w[i] = 0;
	for (i = 0; i < NMax; i++) {
		value = NRCDM_L1L2SVM(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, y,
				w, lambda, C, Li, 1, value, 0);
		crossvalidationValue = crossvalidationForL1L2SVM(A_test, R_Idx_test,
				C_Idx_test, C_Count_test, n_test, m_test, y_test, w,
				trainingset, m_test);
		finalTestValue = crossvalidationForL1L2SVM(A_test_final,
				R_Idx_test_final, C_Idx_test_final, C_Count_test_final,
				n_test_final, m_test_final, y_test_final, w, 0, m_test_final);
		printf("N,%d,crossvalidation,%f,final,%f,nnz,%d\n", i + 1,
				crossvalidationValue, finalTestValue, nnzofW[i]);

		validation[i] = crossvalidationValue;
		finalvalidation[i] = finalTestValue;
		nnzofW[i] = 0;
		for (j = 0; j < n; j++) {
			if (w[j] != 0)
				nnzofW[i]++;
		}
	}
	free(L);
	free(Li);
	computeLipsitzConstantsForL1RegLog(&L, &Li, A_h, R_Idx_h, C_Idx_h,
			C_Count_h, n, nnz, C);
	for (i = 0; i < n; i++)
		w[i] = 0;
	for (i = 0; i < NMax; i++) {
		value = NRCDM_L1Log(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, y, w,
				lambda, C, Li, 1, value, 0);
		crossvalidationValue = crossvalidationForL1L2SVM(A_test, R_Idx_test,
				C_Idx_test, C_Count_test, n_test, m_test, y_test, w,
				trainingset, m_test);
		finalTestValue = crossvalidationForL1L2SVM(A_test_final,
				R_Idx_test_final, C_Idx_test_final, C_Count_test_final,
				n_test_final, m_test_final, y_test_final, w, 0, m_test_final);
		validationLog[i] = crossvalidationValue;
		finalvalidationLog[i] = finalTestValue;
		nnzofWLog[i] = 0;
		for (j = 0; j < n; j++) {
			if (w[j] != 0)
				nnzofWLog[i]++;
		}
		printf("N,%d,crossvalidation,%f,final,%f,nnz,%d\n", i + 1,
				crossvalidationValue, finalTestValue, nnzofWLog[i]);
	}

	FILE *fp;
	fp = fopen("/tmp/PorovnaniePodlaIteracii.csv", "w");
	printf("N,CVLog,CVL2,FVLog,FVL2,nnzLog,nnzL2\n");
	for (i = 0; i < NMax; i++) {
		printf("%d,%f,%f,%f,%f,%d,%d\n", i + 1, validationLog[i],
				validation[i], finalvalidationLog[i], finalvalidation[i],
				nnzofWLog[i], nnzofW[i]);
		fprintf(fp, "%d,%f,%f,%f,%f,%d,%d\n", i + 1, validationLog[i],
				validation[i], finalvalidationLog[i], finalvalidation[i],
				nnzofWLog[i], nnzofW[i]);
	}
	fclose(fp);
}

void porovnaniePodlaIteraciiKDDB() {
	int n;
	int m;
	FILE *fp;
	fp = fopen("/tmp/Vyvoj.csv", "w");

	double lambda = 1;
	double C = 1;
	int NMax = 100;
	double* A_h;
	int * C_Idx_h;
	int * R_Idx_h;
	int * C_Count_h;
	int nnz, i, j, k, l;
	double* y;
	double *L;
	double *Li;

	double* A_test;
	int * C_Idx_test;
	int * R_Idx_test;
	int * C_Count_test;
	double* y_test;
	int n_test;
	int m_test;
	int nnz_test;

	loadDataFromFile("/home/s1052689/kddb", &A_test, &R_Idx_test, &C_Idx_test,
			&C_Count_test, &y_test, &n_test, &m_test, &nnz_test, MAXINSTANCES,
			1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test, m_test, nnz_test);

	int trainingset = m_test;
	loadDataFromFile("/home/s1052689/kddb", &A_h, &R_Idx_h, &C_Idx_h,
			&C_Count_h, &y, &n, &m, &nnz, MAXINSTANCES, 1);
	printf("all features:%d , all data:%d, nnz:%d\n", n, m, nnz);

	double* A_test_final;
	int * C_Idx_test_final;
	int * R_Idx_test_final;
	int * C_Count_test_final;
	double* y_test_final;
	int n_test_final;
	int m_test_final;
	int nnz_test_final;

	loadDataFromFile("/home/s1052689/kddb.t", &A_test_final, &R_Idx_test_final,
			&C_Idx_test_final, &C_Count_test_final, &y_test_final,
			&n_test_final, &m_test_final, &nnz_test_final, MAXINSTANCES, 1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test_final,
			m_test_final, nnz_test_final);

	double w[n];
	for (i = 0; i < n; i++)
		w[i] = 0;

	double value = 0;
	double crossvalidationValue;
	double finalTestValue;
	double finalvalidation[NMax];
	double validation[NMax];
	double finalvalidationLog[NMax];
	double validationLog[NMax];
	int nnzofWLog[NMax];
	int nnzofW[NMax];
	computeLipsitzConstantsForL1RegL2(&L, &Li, A_h, R_Idx_h, C_Idx_h,
			C_Count_h, n, nnz, C);
	for (i = 0; i < n; i++)
		w[i] = 0;
	for (i = 0; i < NMax; i++) {
		crossvalidationValue = crossvalidationForL1L2SVM(A_test, R_Idx_test,
				C_Idx_test, C_Count_test, n_test, m_test, y_test, w, 0, m_test);
		finalTestValue = crossvalidationForL1L2SVM(A_test_final,
				R_Idx_test_final, C_Idx_test_final, C_Count_test_final,
				n_test_final, m_test_final, y_test_final, w, 0, m_test_final);
		nnzofW[i] = 0;
		for (j = 0; j < n; j++) {
			if (w[j] != 0)
				nnzofW[i]++;
		}

		clock_t t1, t2;
		t1 = clock();
		value = NRCDM_L1L2SVM(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, y,
				w, lambda, C, Li, 1, value, 0);
		t2 = clock();
		float diff = ((float) t2 - (float) t1) / 1000000.0F;
		printf("calculation:%f\n", diff);

		fprintf(
				fp,
				"N,%d,crossvalidation,%f,final,%f,nnz,%d, calculaltionTime:%f\n",
				i, crossvalidationValue, finalTestValue, nnzofW[i], diff);
		printf("N,%d,crossvalidation,%f,final,%f,nnz,%d\n", i,
				crossvalidationValue, finalTestValue, nnzofW[i]);
		print_double_array(&w[0], 100);
		validation[i] = crossvalidationValue;
		finalvalidation[i] = finalTestValue;
	}

	free(L);
	free(Li);
	computeLipsitzConstantsForL1RegLog(&L, &Li, A_h, R_Idx_h, C_Idx_h,
			C_Count_h, n, nnz, C);
	for (i = 0; i < n; i++)
		w[i] = 0;
	for (i = 0; i < NMax; i++) {

		crossvalidationValue = crossvalidationForL1L2SVM(A_test, R_Idx_test,
				C_Idx_test, C_Count_test, n_test, m_test, y_test, w, 0, m_test);
		finalTestValue = crossvalidationForL1L2SVM(A_test_final,
				R_Idx_test_final, C_Idx_test_final, C_Count_test_final,
				n_test_final, m_test_final, y_test_final, w, 0, m_test_final);
		clock_t t1, t2;
		t1 = clock();
		value = NRCDM_L1Log(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, y, w,
				lambda, C, Li, 1, value, 0);

		t2 = clock();
		float diff = ((float) t2 - (float) t1) / 1000000.0F;
		printf("calculation:%f\n", diff);

		validationLog[i] = crossvalidationValue;
		finalvalidationLog[i] = finalTestValue;
		nnzofWLog[i] = 0;
		for (j = 0; j < n; j++) {
			if (w[j] != 0)
				nnzofWLog[i]++;
		}
		fprintf(fp, "N,%d,crossvalidation,%f,final,%f,nnz,%d,calculationTime:%f\n", i,
				crossvalidationValue, finalTestValue, nnzofWLog[i],diff);
		printf("N,%d,crossvalidation,%f,final,%f,nnz,%d\n", i,
				crossvalidationValue, finalTestValue, nnzofWLog[i]);
	}

	fclose(fp);
	fp = fopen("/tmp/PorovnaniePodlaIteracii.csv", "w");
	printf("N,CVLog,CVL2,FVLog,FVL2,nnzLog,nnzL2\n");
	for (i = 0; i < NMax; i++) {
		printf("%d,%f,%f,%f,%f,%d,%d\n", i, validationLog[i], validation[i],
				finalvalidationLog[i], finalvalidation[i], nnzofWLog[i],
				nnzofW[i]);
		fprintf(fp, "%d,%f,%f,%f,%f,%d,%d\n", i, validationLog[i],
				validation[i], finalvalidationLog[i], finalvalidation[i],
				nnzofWLog[i], nnzofW[i]);
	}
	fclose(fp);
}

void cas() {
	int n;
	int m;
	double lambda = 1;
	double C = 1;
	int NMax = 100;
	double* A_h;
	int * C_Idx_h;
	int * R_Idx_h;
	int * C_Count_h;
	int nnz, i, j, k, l;
	double* y;
	double *L;
	double *Li;

	double* A_test;
	int * C_Idx_test;
	int * R_Idx_test;
	int * C_Count_test;
	double* y_test;
	int n_test;
	int m_test;
	int nnz_test;

	loadDataFromFile("/document/rcv1_train.binary", &A_test, &R_Idx_test,
			&C_Idx_test, &C_Count_test, &y_test, &n_test, &m_test, &nnz_test,
			MAXINSTANCES, 1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test, m_test, nnz_test);

	int trainingset = m_test * 9 / 10;
	loadDataFromFile("/document/rcv1_train.binary", &A_h, &R_Idx_h, &C_Idx_h,
			&C_Count_h, &y, &n, &m, &nnz, trainingset, 1);
	printf("all features:%d , all data:%d, nnz:%d\n", n, m, nnz);

	double* A_test_final;
	int * C_Idx_test_final;
	int * R_Idx_test_final;
	int * C_Count_test_final;
	double* y_test_final;
	int n_test_final;
	int m_test_final;
	int nnz_test_final;

	loadDataFromFile("/document/rcv1_test.binary", &A_test_final,
			&R_Idx_test_final, &C_Idx_test_final, &C_Count_test_final,
			&y_test_final, &n_test_final, &m_test_final, &nnz_test_final,
			MAXINSTANCES, 1);
	printf("TEST features:%d , all data:%d, nnz:%d\n", n_test_final,
			m_test_final, nnz_test_final);

	double w[n];
	for (i = 0; i < n; i++)
		w[i] = 0;

	double value = 0;
	double crossvalidationValue;
	double finalTestValue;
	double finalvalidation[NMax];
	double validation[NMax];
	double finalvalidationLog[NMax];
	double validationLog[NMax];
	int nnzofWLog[NMax];
	int nnzofW[NMax];
	computeLipsitzConstantsForL1RegL2(&L, &Li, A_h, R_Idx_h, C_Idx_h,
			C_Count_h, n, nnz, C);
	for (i = 0; i < n; i++)
		w[i] = 0;

	clock_t t1, t2;
	t1 = clock();

	value = NRCDM_L1L2SVM(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, y, w,
			lambda, C, Li, 10, value, 0);
	t2 = clock();
	float diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("calculation:%f\n", diff);
}

void cudaSolver() {

	int n = 100;
	int m = 50;
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
	double value = 0;
	for (i = 0; i < n; i++)
		x[i] = 0;
	value = NRCDM_SR(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, b, x, lambda,
			Li, NMax * 100, value, 0);
	for (i = 0; i < n; i++)
		x[i] = 0;
	value = NRCDM_SR(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, b, x, lambda,
			Li, NMax, value, 1);

}

int main(void) {
	srand(1);
	//	cudaSolver();
	//	int i;
	//	long n = 10000000;
	//	int features[n];
	//	printf("Idem alokovat fatures block\n");
	//	printf("Fatures block allocated\n");
	//	for (i = 0; i < n; i++) {
	//		features[i] = 0;
	//	}
	//	printf("Fatures block checked\n");
	//
	//	int* d;
	//	d = (int*) malloc(n * sizeof(int));
	//	printf("Idem alokovat fatures block\n");
	//	printf("Fatures block allocated\n");
	//	for (i = 0; i < n; i++) {
	//		d[i] = 0;
	//	}
	//	printf("Fatures block checked\n");

	porovnaniePodlaIteraciiKDDB();
	//				L1regularizedLogisticRegression_KDDB();
	//	L1regularizedL2_KDDB();
	//	L1regularizedL2();
	//		RandomSparseRegression();
	//		porovnaniePodlaIteracii();


}

