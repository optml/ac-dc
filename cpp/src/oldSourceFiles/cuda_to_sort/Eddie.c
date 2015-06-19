//ulimit -s unlimited
//gcc -lm -std=c99 NRCDML1RegLog.c && ./a.out

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#define MAXFEATURES  30000000
#define MAXINSTANCES 30000000

double CS[] = { 0.062500, 0.125000, 0.250000, 0.500000, 1.000000, 2.000000,
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

int main(void) {
	srand(1);
	int i;
	long n = 1000;
	int features[n];
	printf("Idem alokovat fatures block\n");
	printf("Fatures block allocated\n");
	for (i = 0; i < n; i++) {
		features[i] = 0;
	}
	printf("Fatures block checked\n");

	int* d;
	d = (int*) malloc(n * sizeof(int));
	printf("Idem alokovat fatures block\n");
	printf("Fatures block allocated\n");
	for (i = 0; i < n; i++) {
		d[i] = 0;
	}
	printf("Fatures block checked\n");

	double pole[n];
	for (i = 0; i < n; i++) {
		pole[i] = (double)i;
	}
	printf("pole created\n");
}

