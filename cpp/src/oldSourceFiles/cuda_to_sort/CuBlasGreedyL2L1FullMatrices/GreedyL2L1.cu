//ulimit -s unlimited
//nvcc -lcublas GreedyL2L1.cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <cuda.h>
#include "device_functions.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "cublas.h"
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
#define lambda 1
#define NMAX 1000
#define toll 0.00000001

void modify(float *m, int ldm, int n, int p, int q, float alpha, float beta) {
	cublasSscal(n - p + 1, alpha, &m[IDX2F(p,q,ldm)], ldm);
	cublasSscal(ldm - p + 1, beta, &m[IDX2F(p,q,ldm)], 1);
}
#define M 6
#define N 5

void computeLipsitzConstantsForL2ObjetciveAndColumnMatrix(double** L,
		double ** Li, double *A, int*R_Idx, int*C_Idx, int*C_Count, int n,
		int nnz);
void checkCUDAError(const char *msg);

void generateRandomProblem(float ** A, float** b, int n, int m);

int parallelGreedyL2L1(double *A, int n, int m, int nnz, double* b, double*x,
		double lambdaVal, double* Li, int NMAXIMUM, float tollerance) {
	int i, j;
	cublasStatus stat;
	float* devPtrA;
	float* a = 0;
	a = (float *) malloc(M * N * sizeof(*a));
	if (!a) {
		printf("host memory allocation failed");
		return EXIT_FAILURE;
	}
	for (j = 1; j <= N; j++) {
		for (i = 1; i <= M; i++) {
			a[IDX2F(i,j,M)] = (i - 1) * M + j;
		}
	}
	cublasInit();
	stat = cublasAlloc(M * N, sizeof(*a), (void**) &devPtrA);

	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("device memory allocation failed");
		return EXIT_FAILURE;
	}
	cublasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);

	for (j = 1; j <= N; j++) {
		for (i = 1; i <= M; i++) {
			printf("%7.0f", a[IDX2F(i,j,M)]);
		}
		printf("\n");
	}

	//	modify(devPtrA, M, N, 2, 3, 16.0f, 12.0f);

	cublasSscal(M, 10, &devPtrA[IDX2F(2,2,M)], 2);

	cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
	cublasFree(devPtrA);
	cublasShutdown();
	for (j = 1; j <= N; j++) {
		for (i = 1; i <= M; i++) {
			printf("%7.0f - %d ", a[IDX2F(i,j,M)], IDX2F(i,j,M));
		}
		printf("\n");
	}
	return EXIT_SUCCESS;

}

void serialGreedyL2L1(double *A, int*R_Idx, int*C_Idx, int*C_Count, int n,
		int m, int nnz, double* b, double*x, double lambdaVal, double* Li,
		int NMAXIMUM, float tollerance) {
	//	double g[m];
	//	int i, j;
	//	for (j = 0; j < m; j++)
	//		g[j] = -b[j];
	//	for (i = 0; i < n; i++) {
	//		x[i] = 0;
	//	}

	//	G=A'*A;
	//	if (sparseData)
	//	    IdxGroup=cell(n,1);
	//	    for i=1:n
	//	        IdxGroup{i}=find(G(:,i));
	//	    end
	//	end
	//	Derivatives = A'*g;
	//	T=CDshrink(Derivatives , x, lambda,Li);
	//
	//
	//
	//
	//
	//	VALS=T.^2.*L;
	//	for i=1:NMax
	//	   [val,j]=max(VALS);
	//	   x(j)=x(j)+T(j);
	//
	//	        Derivatives=Derivatives+G(:,j)*T(j);
	//	        T=CDshrink(Derivatives , x, lambda,Li);
	//	        VALS=T.^2.*L;


	//	double partialDetivative, delta, tmp;
	//	// iteration counter
	//	for (N = 0; N < NMAX * n; N++) {
	//		int idx = (int) (n * (rand() / (RAND_MAX + 1.0)));
	//		partialDetivative = 0;
	//		for (j = C_Idx[idx]; j < C_Idx[idx] + C_Count[idx]; j++) {
	//			partialDetivative += A[j] * residuals[R_Idx[j]];
	//		}
	//
	//		tmp = Li[idx] * (partialDetivative + lambda);
	//		if (x[idx] > tmp) {
	//			delta = -tmp;
	//		} else {
	//			tmp = Li[idx] * (partialDetivative - lambda);
	//			if (x[idx] < tmp) {
	//				delta = -tmp;
	//			} else {
	//				delta = -x[idx];
	//			}
	//		}
	//		x[idx] += delta;
	//		for (j = C_Idx[idx]; j < C_Idx[idx] + C_Count[idx]; j++) {
	//			residuals[R_Idx[j]] += A[j] * delta;
	//		}
	//		if (N % (n / 100) == 0) {
	//			int nnzcount = 0;
	//			value = 0;
	//			for (i = 0; i < n; i++) {
	//				if (x[i] != 0)
	//					nnzcount++;
	//				if (x[i] > 0)
	//					value += x[i];
	//				else
	//					value -= x[i];
	//			}
	//			for (j = 0; j < m; j++)
	//				value += 0.5 * residuals[j] * residuals[j];
	//			fprintf(fp, "Iteracia:%d, value:%f, nnz:%d, epsilon: %f\n", N,
	//					value, nnzcount, value - optimalvalue);
	//		}
	//
	//	}
	//	fclose(fp);
	//	return value;
}

__global__ void getDiagonalElementsIntoL_Kernel(float *G, float * L, int n) {
	int id = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	if (id < n) {
		L[id] = G[IDX2F(id+1,id+1,n)];

	}

}

__global__ void IncreaseElement(float* x, int element, float* T) {
	x[element] = x[element] + T[element];

}

__global__ void ShrinkKernel(float *T, float * E, float* derivatives, float* L,
		float* x, float lambdaParameter, int n) {
	int id = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	if (id < n) {
		float LLocal = L[id];
		float Li = 1 / LLocal;
		float xLocal = x[id];
		float lambdaOverL = lambdaParameter * Li;
		float alpha = derivatives[id] * Li;
		float deltaL = xLocal - alpha - lambdaOverL;
		float deltaR = xLocal - alpha + lambdaOverL;
		float t;
				if (deltaL > 0)
					t = deltaL - xLocal;
				else if (deltaR < 0)
					t = deltaR - xLocal;
				else
					t = -xLocal;

//		t = deltaL;
//		if (t < 0) {
//			t = deltaR;
//			if (t > 0)
//				t = 0;
//		}
//		t = t - xLocal;
		//		 alpha = A(:,j)'*g;
		//		    delta  = - (alpha+lambda)*Li(j);
		//		    if (x(j)+delta  < 0)
		//		        deltaN = - (alpha-lambda)*Li(j);
		//		         if (x(j)+deltaN > 0 )
		//		             delta = -x(j);
		//		         else
		//		             delta=deltaN;
		//		         end
		//		    end
		//		    x(j) = x(j) + delta;

		T[id] = t;
		E[id] = t * t * LLocal;
	}

}

int RandomFullRegression() {
	int row = 2000;
	int col = 10000;
	dim3 dimBlock( 512);
	dim3 dimGridRCDM( 1, 1+col/512);
	float* A_h;
	float *b_h;
	int i ;
	generateRandomProblem(&A_h, &b_h, col, row);
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float* A_dev;
	float* G_dev;

	cublasStatus stat;
	cublasInit();
	stat = cublasAlloc(row * col, sizeof(*A_h), (void**) &A_dev);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("device memory allocation failed\n");
		return EXIT_FAILURE;
	}
	cublasSetMatrix(row, col, sizeof(*A_h), A_h, row, A_dev, row);
	stat = cublasAlloc(col * col, sizeof(float), (void**) &G_dev);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("device memory allocation failed 2\n");
		return EXIT_FAILURE;
	}

	cudaEventRecord(start, 0);
	//	cublasSgemm('T', 'N', col, col, row, 1, A_dev, col, A_dev, col, 0, G_dev, col);
	cublasSgemm('T', 'N', col, col, row, 1, A_dev, row, A_dev, row, 0, G_dev,
			col);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("A'*A: %f ms\n", time);
	float* L_dev;


	cudaMalloc((void**) &L_dev, col * sizeof(float));
	//	stat = cublasAlloc(n, sizeof(float), (void**) &L_dev);
	//	if (stat != CUBLAS_STATUS_SUCCESS) {
	//		printf("device memory allocation failed 2\n");
	//		return EXIT_FAILURE;
	//	}


	getDiagonalElementsIntoL_Kernel<<< dimGridRCDM ,dimBlock >>>(G_dev, L_dev, col);
	checkCUDAError("kernel execution");

	float* derivatives_dev;
	stat = cublasAlloc(col, sizeof(float), (void**) &derivatives_dev);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("device memory allocation failed 2\n");
		return EXIT_FAILURE;
	}

	float* b_dev;

	cublasAlloc(row, sizeof(float), (void**) &b_dev);
	checkCUDAError("malloc  b");
	cublasSetVector(row, sizeof(float), b_h, 1, b_dev, 1);
	cudaThreadSynchronize();
	checkCUDAError("kernel execution Set Vector b");
	cublasSgemv('T', row, col, -1, A_dev, row, b_dev, 1, 0, derivatives_dev, 1);
	// derivatives_dev = A'*(-b) = A'*g
	checkCUDAError("kernel execution Derivatives compute");
	float* x_dev;
	cublasAlloc(col, sizeof(float), (void**) &x_dev);
	checkCUDAError("Alokacia x_dev");
	cublasSscal(col, 0, x_dev, 1);
	checkCUDAError("kernel execution setnutie X na 0");

	float* T_dev;
	float* E_dev;
	cublasAlloc(col, sizeof(float), (void**) &T_dev);
	checkCUDAError("Alokacia T_dev");
	cublasAlloc(col, sizeof(float), (void**) &E_dev);
	checkCUDAError("Alokacia E_dev");

	cudaEventRecord(start, 0);
	ShrinkKernel<<< dimGridRCDM ,dimBlock >>>(T_dev, E_dev, derivatives_dev, L_dev,x_dev,lambda, col);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("shrink: %f ms\n", time);


	cudaEventRecord(start, 0);
	for (i=0;i<NMAX;i++)
	{
	int maxIndex = cublasIsamax(col, E_dev, 1);
	maxIndex = maxIndex - 1;
	IncreaseElement<<< 1 ,1 >>>(x_dev, maxIndex, T_dev);
	checkCUDAError("IncreaseElement");
	float tPoint[1];
	cublasGetVector(1, sizeof(float), &T_dev[maxIndex], 1, tPoint, 1);
	cublasSaxpy(col, tPoint[0], &G_dev[IDX2F(1,maxIndex+1,col)], 1,
			derivatives_dev, 1);
	cudaThreadSynchronize();
	ShrinkKernel<<< dimGridRCDM ,dimBlock >>>(T_dev, E_dev, derivatives_dev, L_dev,x_dev,lambda, col);


	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Totalny cas na %d iteracii: %f ms =  %f ms / iteraciu (dimenzia %dx%d) \n", NMAX, time,(float)time/NMAX,row,col);








//int j;
//float* L_h;
//L_h = (float*) malloc(col * sizeof(float));
//	float T_h[col];
//	cublasGetVector(col, sizeof(float), T_dev, 1, T_h, 1);
//	checkCUDAError("kernel execution Get T");
//	for (i = 0; i < col; i++) {
//		printf("T[%d]=%f\n", i, T_h[i]);
//	}
//	printf("\n");
//
//	float E_h[col];
//	cublasGetVector(col, sizeof(float), E_dev, 1, E_h, 1);
//	checkCUDAError("kernel execution Get E");
//	for (i = 0; i < col; i++) {
//		printf("E[%d]=%f\n", i, E_h[i]);
//	}
//	printf("\n");
//
//	float x_h[col];
//	cublasGetVector(col, sizeof(float), x_dev, 1, x_h, 1);
//	checkCUDAError("kernel execution Get X");
//	for (i = 0; i < col; i++) {
//		printf("x[%d]=%f\n", i, x_h[i]);
//	}
//	printf("\n");
//
//	//	Derivatives = A'*g;
//	//	T=CDshrink(Derivatives , x, lambda,Li);
//
//	//	cudaThreadSynchronize();
//	//	checkCUDAError("kernel Synchronize");
//
//
//	float derivatives_h[col];
//
//	cudaMemcpy(L_h, L_dev, col * sizeof(float), cudaMemcpyDeviceToHost);
//	checkCUDAError("get L");
//
//	cublasGetVector(col, sizeof(float), derivatives_dev, 1, derivatives_h, 1);
//	checkCUDAError("kernel execution det Derivatives");
//
//	for (i = 0; i < col; i++) {
//		printf("L[%d]=%f; D[%d]=%f\n", i, L_h[i], i, derivatives_h[i]);
//	}
//	printf("\n");
//	for (i = 0; i < row * col; i++) {
//		printf("%7.0f", A_h[i]);
//	}
//	printf("\n");
//	printf("\n");
//	for (i = 1; i <= row; i++) {
//		for (j = 1; j <= col; j++) {
//			printf("%7.0f", A_h[IDX2F(i,j,row)]);
//		}
//		printf("| b = %f", b_h[i - 1]);
//		printf("\n");
//	}
//	printf("\n");
//	float * G_h;
//	G_h = (float*) malloc(col * col * sizeof(float));
//	cublasGetMatrix(col, col, sizeof(float), G_dev, col, G_h, col);
//	checkCUDAError("kernel execution get G");
//	for (j = 1; j <= col; j++) {
//		for (i = 1; i <= col; i++) {
//			printf("%7.0f", G_h[IDX2F(i,j,col)]);
//		}
//		printf("\n");
//	}
	printf("Koniec\n");
	cublasFree(A_dev);
	cublasFree(G_dev);
	cublasFree(derivatives_dev);
	cublasFree(b_dev);
	cudaFree(L_dev);
	cublasShutdown();
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//	computeLipsitzConstantsForL2ObjetciveAndColumnMatrix(&L, &Li, A_h, R_Idx_h,
	//			C_Idx_h, C_Count_h, n, nnz);
	//	double* x_Serial;
	//	x_Serial = (double *) malloc(n * sizeof(double));
	//	serialGreedyL2L1(A_h, R_Idx_h, C_Idx_h, C_Count_h, n, m, nnz, b_h,
	//			x_Serial, lambda, Li, NMAX, toll);
	//	for (i = 0; i < n; i++) {
	//		printf("x[%d]=%f \n", i, x_Serial[i]);
	//	}
	return EXIT_SUCCESS;
}

int main() {
	srand(2);
	RandomFullRegression();

	return EXIT_SUCCESS;
}

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}

void computeLipsitzConstantsForL2ObjetciveAndColumnMatrix(double** L,
		double ** Li, double *A, int*R_Idx, int*C_Idx, int*C_Count, int n,
		int nnz) {
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
void generateRandomProblem(float ** A, float** b, int n, int m) {
	int i;
	*A = (float *) malloc(n * m * sizeof(float));
	for (i = 0; i < n * m; i++) {
				(*A)[i] = ((float) rand() / RAND_MAX);
//		(*A)[i] = rand() % 10;

	}
	*b = (float *) malloc(m * sizeof(float));
	for (i = 0; i < m; i++)
				(*b)[i] = (float) rand() / RAND_MAX;
//		(*b)[i] = rand() % 8;

}
