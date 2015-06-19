#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#define M 200
#define N 200

void print_time_message(clock_t t1, char message[200]) {
	clock_t t2 = clock();
	double diff = ((float) t2 - (float) t1) / 1000000.0F;
	printf("%s: %f\n", message, diff);
}


int main() {
	int i, j, ii, jj, kk;
	double B[N];
	double A[M][N];
	double C[N][N], D[N][N], E[N][N];
	double sum = 0;

	puts("PARALEL REGIUON START");
	clock_t t1, t2;
		t1 = clock();
#pragma omp parallel private(i,j), shared(B,C,D,E)
	{
#pragma omp for

		for (j = 0; j < N; j++) {
			B[j] = (double) j;
		}

		for (j = 1; j < N; j++) {
			for (i = 0; i < M; i++) {
				C[i][j] = 1.0;
				D[i][j] = 2.0;
				E[i][j] = 3.0;
			}
		}

	}
	puts("PARALEL REGIUON FINISHED");
	print_time_message(t1, "DONE");
#pragma omp parallel private(i,j,kk,ii), shared(A,B,C,D,E)
	{
#pragma omp for ordered , schedule(static,1)
		for (j = 1; j < N; j++) {
			/* Simulate a lot of work to do */
			for (i = 0; i < M; i++) {
				C[i][j] = C[i][j] * B[j];
				for (jj = 0; jj < N; jj++) {
					E[i][j] = C[i][jj] * D[jj][j];
				}
				for (kk = 0; kk < N; kk++) {
					for (ii = 0; ii < N; ii++) {
						A[i][j] = sqrt(A[i][j] + E[ii][kk] * C[i][j]) * sqrt(
								A[i][j] + E[ii][kk] / C[i][j]);
					}
				}
			}

#pragma omp ordered
			B[j] = B[j - 1] + A[j][j];

		}
	}

	/* Crude check for correct answer*/

	for (i = 0; i < N; i++) {
		sum = sum + B[i];
	}

	printf("%f\n", sum);
	print_time_message(t1, "ALL");
	return 1;

}
