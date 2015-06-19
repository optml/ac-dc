#include <stdio.h>
#include <stdlib.h>


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

int main(void) {

	float lambda = 20;
	int n = 10;
	int p = 3;
	int m = 100;
	int i, j;

	float A_h[n][p]; // host A matrix pointers
	int IDX_h[n][p]; // host Aindex matrix pointers
	float x_optimal[n];
	float b[m];
	float x[n];

	float L[n]; // Lipschitz constants


	for (i = 0; i < n; i++) {
		float tmp = 0;
		for (j = 0; j < p; j++) {
			int idx = (int) (m * (rand() / (RAND_MAX + 1.0)));
			float val = (float) rand() / RAND_MAX;
			A_h[i][j] = 2*val-1;
			IDX_h[i][j] = idx;
			tmp += val * val;
		}
		L[i] = tmp;
		x[i] = 0;
		x_optimal[i] = (float) rand() / RAND_MAX;
	}
	for (j = 0; j < m; j++) b[j]=0;
	for (i = 0; i < n; i++) {
		float tmp = 0;
		for (j = 0; j < p; j++) {
			b[IDX_h[i][j]]+=x_optimal[i] * A_h[i][j];
		}


	}
//	for (j=0;j<m; j++) printf("b[%d] =  %f  \n", j, b[j]);

	// RCDM
	//	RCDM(A_h,IDX_h, L,b, lambda,n,1);
	int N;
	// Calculate residuals
	float residuals[m];
	for (j = 0; j < m; j++)
		residuals[j] = 0;
	for (i = 0; i < n; i++) {
		for (j = 0; j < p; j++) {
			residuals[IDX_h[i][j]] += A_h[i][j] * x[i];
		}
	}
	for (j = 0; j < m; j++)
		residuals[j] += -b[j];
	//----------------RCDM----------serial
	for (N = 0; N < (10*n); N++) {
		int idx = (int) (n * (rand() / (RAND_MAX + 1.0)));

		float tmp = 0;
		for (j = 0; j < p; j++) {
			//			printf("tmp:%f  A:%f   residual:%f  \n",tmp,A_h[idx][j],residuals[IDX_h[idx][j]]);
			tmp += A_h[idx][j] * residuals[IDX_h[idx][j]];
		}
		tmp = -tmp / L[idx];
		//		printf("delta[%d] =  %f  \n",idx, tmp);
		x[idx] += tmp;

		//update residuals:
		for (j = 0; j < p; j++) {
			residuals[IDX_h[idx][j]] += tmp * A_h[idx][j];
		}

//		printf("Iteration %d,  x[%d]=%f  \n", N, idx, x[idx]);

	}
	printf("Comparison \n");
	for (i = 0; i < n; i++) {
		printf("x[%d] =  %f ;x*[%d]=%f  \n", i, x[i], i, x_optimal[i]);
	}

	// Allocation arrays on cuda device:

	float * A_dev;
	float * L_dev;
	float * x_dev;
	float * b_dev;
	float * lambda_dev;


	//----------------RCDM------- parallel
}
