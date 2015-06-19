//ulimit -s unlimited
//nvcc -lcusparse test.cu
//nvcc -lcublas  -lcusparse -arch sm_20  test.cu && ./a.out
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include "device_functions.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "cublas.h"
#include "cusparse.h"

#define MaxTTD_NNZ_ELEMENTS  300000000
#define TOTALTHREDSPERBLOCK 256
#define COLUMNLENGTH 4
#define NMAXITERKernel 100000
#define ROWSCALE 2

using namespace std;
void generateTTDProblem(int row, int col, int exampleType, int* mOut,
		int* nOut, float** A, int** ColIDX, int ** RowIDX, int * nnzElements,
		int* boundary, int boundarySize, int** nodeDescription, int rowScale);
void getBoundaryVector(int row, int col, int exampleType, int** boundary,
		int* boudarySize);
void getForceVector(float ** d, int m, int r, int c, int exampleType);
int GCD(int a, int b);
__global__ void setup_kernel(curandState *state);
void checkCUDAError(const char *msg);
void scaleBetta(int* betta, float* beta, float scale);
int minimabs(int * betta);
int maximabs(int * betta);
void print_time_message(clock_t* t1, char message[200]) {
	clock_t t2 = clock();
	double diff = ((float) t2 - (float) (*t1)) / 1000000.0F;
	printf("%s: %f sec.\n", message, diff);
	*t1 = clock();
}
double getElapsetTime(clock_t* t1) {
	clock_t t2 = clock();
	double diff = ((float) t2 - (float) (*t1)) / 1000000.0F;
	*t1 = clock();
	return diff;
}

__global__ void RCDMKernel(float *A, int*R_Idx, int* n, float*residuals,
		float*x, float * lambda, curandState* cstate) {
	int j, i, k;
	float delta, tmp; //partialDetivative
	int id = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	curandState localState = cstate[id];

	__shared__ float partialDetivative[TOTALTHREDSPERBLOCK];
	float xLocal;
	float LiLocal;
	float ALocal[COLUMNLENGTH];
	int cidx;
	int RIDX[COLUMNLENGTH];
	for (k = 0; k < NMAXITERKernel; k++) {
		double d = curand_uniform_double(&localState);
		int idx = (int) (d * n[0]);
		// LOAD A, R, residuals
		//		float* residualsAddress[COLUMNLENGTH];
		xLocal = x[idx];
		//		LiLocal = Li[idx];
		cidx = idx * COLUMNLENGTH;
		partialDetivative[threadIdx.x] = 0;
		//		        #pragma unroll COLUMNLENGTH
		for (i = 0; i < COLUMNLENGTH; i++) {
			j = cidx + i;
			ALocal[i] = A[j];
			LiLocal += ALocal[i] * ALocal[i];
			RIDX[i] = R_Idx[j] - 1;
			//			residualsAddress[i] = &residuals[RIDX[i]];
			partialDetivative[threadIdx.x] += ALocal[i] * residuals[RIDX[i]];
		}
		LiLocal = 1 / LiLocal;
		tmp = LiLocal * (partialDetivative[threadIdx.x] + lambda[0]);
		if (xLocal > tmp) {
			delta = -tmp;
		} else {
			tmp = LiLocal * (partialDetivative[threadIdx.x] - lambda[0]);
			if (xLocal < tmp) {
				delta = -tmp;
			} else {
				delta = -xLocal;
			}
		}
		atomicAdd(&x[idx], delta);
		//		x[idx]+=delta;
		//		atomicAdd(&x[idx], 1);
		for (i = 0; i < COLUMNLENGTH; i++) {
			atomicAdd(&residuals[RIDX[i]], ALocal[i] * delta);
			//			residuals[RIDX[i]]+= ALocal[i] * delta;
			//			atomicAdd(residualsAddress[i], ALocal[i] * delta);
		}

	}
	//	cstate[id] = localState;
}

void calculateTTDProblem() {
	int dim1 = 14 * 2;
	int dim2 = 1;
	int totalThreads = TOTALTHREDSPERBLOCK;

	int col, row, exampleType;
	//	cout << "Enter number of columns: ";
	col = 115;
	//		cin >> col;
	//	cout << "Enter number of rows: ";
	row = 115;
	//		cin >> row;
	//	cout << "Enter example type: ";
	//	cin >> exampleType;
	exampleType = 2;
	int m, n;
	clock_t t1;//, t2;
	float * A_TTD;
	int * Row_IDX;
	int * Col_IDX;
	int nnzElements;
	int* nodeDescription;
	//-------------------------------GET BOUNDARY POINTS
	int * boundary;
	int boundarySize = 0;
	getBoundaryVector(row, col, exampleType, &boundary, &boundarySize);
	cout << "Boundary size is " << boundarySize << "\n";
	//-------------------------------GENERATE PROBLEMS
	t1 = clock();
	generateTTDProblem(row, col, exampleType, &m, &n, &A_TTD, &Col_IDX,
			&Row_IDX, &nnzElements, boundary, boundarySize, &nodeDescription,
			ROWSCALE);
	print_time_message(&t1, "Getting problem dimension");
	cout << "Dimension of your problem is " << m << " x  " << n << "\n";
	printf("Number of NNZ: %d\n", nnzElements);
	//-------------------------------GET FORCE VECTORS
	float* g;
	getForceVector(&g, m, row, col, exampleType);
	for (int i = 0; i < m; i++) {
		g[i] = -g[i];
	}
	print_time_message(&t1, "Force vector obtained");
	float x[n];
	float Li[n];
	for (int i = 0; i < n; i++) {
		x[i] = 0;
		Li[i] = 0;
		for (int k = 4 * i; k < 4 * i + 4; k++) {
			Li[i] += A_TTD[k] * A_TTD[k];
		}
		if (Li[i] > 0) {
			Li[i] = 1 / Li[i];
		}
	}
	//-------------------------------Inicializacia CuSpare Library
	/* allocate GPU memory and copy the matrix and vectors into it */

	dim3 dimBlock( totalThreads);
	dim3 dimGridRCDM( dim1, dim2);
	int totalthreadsOnBlocks = dim1 * dim2 * totalThreads;
	curandState *devStates;
	cudaMalloc((void **) &devStates, totalthreadsOnBlocks * sizeof(curandState));
	setup_kernel<<< dimGridRCDM, dimBlock >>>(devStates);
	checkCUDAError("Inicializacia ranom states");

	float* A_d;
	int* R_Idx_d;
	cudaMalloc((void**) &A_d, nnzElements * sizeof(float));
	checkCUDAError("kernel execution Alloc A_d");
	cudaMalloc((void**) &R_Idx_d, nnzElements * sizeof(int));
	checkCUDAError("kernel execution Alloc R_IDX");
	int * n_d;
	cudaMalloc((void**) &n_d, 1 * sizeof(int));
	checkCUDAError("kernel execution Alloc ...");
	float* g_d;
	cudaMalloc((void**) &g_d, m * sizeof(float));
	checkCUDAError("kernel execution Alloc g_d");
	float* x_d;
	cudaMalloc((void**) &x_d, n * sizeof(float));
	checkCUDAError("kernel execution Alloc x_d");
	float* lambda_d;
	cudaMalloc((void**) &lambda_d, 1 * sizeof(float));
	checkCUDAError("kernel execution Alloc lambda_d");
	//	float* Li_d;
	//	cudaMalloc((void**) &Li_d, n * sizeof(float));
	//	checkCUDAError("kernel execution Alloc Li_d");
	print_time_message(&t1, "Device memory allocated");

	//------------------------------- COPY DATA
	cudaMemcpy(A_d, A_TTD, nnzElements * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy A_d");
	cudaMemcpy(R_Idx_d, Row_IDX, nnzElements * sizeof(int),
			cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy R");
	cudaMemcpy(g_d, g, m * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution compy ....");
	cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution compy ....");
	cudaMemcpy(n_d, &n, 1 * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution compy ....");
	//	cudaMemcpy(Li_d, Li, n * sizeof(float), cudaMemcpyHostToDevice);
	//	checkCUDAError("kernel execution compy ....");
	print_time_message(&t1, "Data coppied");
	//-------------------------------serial code
	float time;
	double serialPower;
	float lambda = 0.0001;

	double serialObjective = 0;
	for (int i = 0; i < m; i++)
		serialObjective += g[i] * g[i];
	serialObjective = serialObjective * 0.5;
	printf("Serial code execution. Objective at beggining: %f\n",
			serialObjective);
	int SerialContractor = 1;
	getElapsetTime(&t1);
	for (int i = 0; i < n / SerialContractor; i++) {
		int idx = (int) (n * (rand() / (RAND_MAX + 1.0)));
		float tmp = 0;
		for (int j = idx * 4; j < idx * 4 + 4; j++) {
			tmp += A_TTD[j] * g[Row_IDX[j]];
		}
		float tmp1 = Li[idx] * (tmp + lambda);
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
		for (int j = 4 * idx; j < 4 * idx + 4; j++) {
			g[Row_IDX[j]] += tmp * A_TTD[j];
		}
	}
	time = getElapsetTime(&t1);
	serialPower = n / time;
	serialPower = (double) serialPower / SerialContractor;
	printf("Serial code duration: %f,  %f iter/sec\n", time, n / time);
	serialObjective = 0;
	for (int i = 0; i < m; i++)
		serialObjective += g[i] * g[i];
	double serialL1Norm = 0;
	for (int j = 0; j < n; j++)
		serialL1Norm += abs(x[j]);
	serialObjective = 0.5 * serialObjective + lambda * serialL1Norm;
	printf("Serial code execution. Objective after n iterations: %f\n",
			serialObjective);
	//-------------------------------Computation

	int doworking = 1;
	int iterations = 1;
	while (doworking == 1) {
		cout << "Current lambda is " << lambda
				<< ". Do you want to change it? (y/n): ";
		string doContinue;
		cin >> doContinue;
		if (doContinue == "y") {
			cout << "Enter lambda: ";
			cin >> lambda;
		}
		cout << "Enter number of iterations: ";
		cin >> iterations;
		float L1Norm = cublasSasum(n, x_d, 1);
		float residualsSum = cublasSdot(m, g_d, 1, g_d, 1);
		double objectiveValue = 0.5 * residualsSum;
		objectiveValue += lambda * L1Norm;
		printf("L1 norm = %f, redisuals=%f,  objective=%1.16f\n", L1Norm,
				residualsSum, objectiveValue);

		cudaMemcpy(lambda_d, &lambda, 1 * sizeof(float), cudaMemcpyHostToDevice);
		for (int i = 0; i < iterations; i++) {
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);

			RCDMKernel<<< dimGridRCDM, dimBlock >>>(A_d, R_Idx_d, n_d,
					g_d, x_d, lambda_d, devStates);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			double parallelPower = totalthreadsOnBlocks;
			parallelPower = parallelPower / time;
			parallelPower = parallelPower * 1000 * NMAXITERKernel;
			printf(
					"Elapset time: %f ms; %1.2f iterations/sec; speedup: %1.4f\n",
					time, parallelPower, parallelPower / serialPower);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			L1Norm = cublasSasum(n, x_d, 1);
			residualsSum = cublasSnrm2(m, g_d, 1);
			objectiveValue = 0.5 * residualsSum * residualsSum;
			objectiveValue += lambda * L1Norm;
			printf("%d:L1 norm = %f, redisuals=%f,  objective=%1.16f\n", i,
					L1Norm, residualsSum, objectiveValue);
		}

		cout << "Save particular solution to file? (y/n): ";
		cin >> doContinue;
		if (doContinue == "y") {
			float treshHold;
			cout << "Enter treshhold for x: ";
			cin >> treshHold;
			int writtenBars = 0;
			FILE *fp;
			fp = fopen("/tmp/ttd.txt", "w");
			cudaMemcpy(x, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
			checkCUDAError("kernel execution compy ....");
			for (int i = 0; i < n; i++) {
				if (abs(x[i]) > treshHold) {
					writtenBars++;
					fprintf(fp, "%d,%d,%d,%d,%f\n", nodeDescription[i * 4],
							nodeDescription[i * 4 + 1], nodeDescription[i * 4
									+ 2], nodeDescription[i * 4 + 3], x[i]);
				}
			}
			fclose(fp);
			printf("Number of written bars:%d\n", writtenBars);
		}
		cout << "Continue? (y/n): ";
		cin >> doContinue;
		if (doContinue == "n") {
			doworking = 0;
		}
	}

	print_time_message(&t1, "Computation");

	//-------------------------------DeAllocation
	cudaFree(devStates);
	cudaFree(A_d);
	cudaFree(R_Idx_d);
	cudaFree(n_d);
	cudaFree(g_d);
	cudaFree(x_d);
	cudaFree(lambda_d);
	//	cudaFree(Li_d);
	print_time_message(&t1, "Device memory DeAllocated");

	//	   	   cout << "Your name score is " << myName << "\n";
	//	   	   cout << "Your weight in pounds is " << myWeight << "\n";
	//	   	   cout << "Your height in inches is " << myHeight << "\n";
	//	   cout << "Enter your height in inches: ";
	//	   cin >> myHeight;
	//	   string myName;
	//	   cout << "Your name score is " << myName << "\n";
	//	   cout << "Your weight in pounds is " << myWeight << "\n";
	//	   cout << "Your height in inches is " << myHeight << "\n";

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
		T[id] = t;
		E[id] = t * t * LLocal;
	}
}

__global__ void ShrinkKernelSubset(float *T, float * E, float* derivatives,
		float* L, float* x, float lambdaParameter, int* ACounts, int* Indexes) {
	int id = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	if (id < ACounts[0]) {
		id = Indexes[id];
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
		T[id] = t;
		E[id] = t * t * LLocal;
	}
}

void greedyTTD() {
	int col, row, exampleType;
	cout << "Enter number of columns: ";
	col = 80;
	cin >> col;
	cout << "Enter number of rows: ";
	row = 80;
	cin >> row;
	//	cout << "Enter example type: ";
	//	cin >> exampleType;
	exampleType = 7;
	int m, n;
	clock_t t1;//, t2;
	float * A_TTD;
	int * Row_IDX;
	int * Col_IDX;
	int nnzElements;
	int* nodeDescription;
	//-------------------------------GET BOUNDARY POINTS
	int * boundary;
	int boundarySize = 0;
	getBoundaryVector(row, col, exampleType, &boundary, &boundarySize);
	cout << "Boundary size is " << boundarySize << "\n";

	//-------------------------------GENERATE PROBLEMS
	t1 = clock();
	generateTTDProblem(row, col, exampleType, &m, &n, &A_TTD, &Col_IDX,
			&Row_IDX, &nnzElements, boundary, boundarySize, &nodeDescription,
			ROWSCALE);
	print_time_message(&t1, "Getting problem dimension");
	cout << "Dimension of your problem is " << m << " x  " << n << "\n";
	printf("Number of NNZ: %d\n", nnzElements);
	//-------------------------------GET FORCE VECTORS
	float* g;
	getForceVector(&g, m, row, col, exampleType);
	for (int i = 0; i < m; i++) {
		g[i] = -g[i];
	}
	print_time_message(&t1, "Force vector obtained");
	float x[n];
	float L[n];
	for (int i = 0; i < n; i++) {
		x[i] = 0;
		L[i] = 0;
		for (int k = 4 * i; k < 4 * i + 4; k++) {
			L[i] += A_TTD[k] * A_TTD[k];
		}
	}
	//-------------------------------Inicializacia CuSpare Library
	/* allocate GPU memory and copy the matrix and vectors into it */
	cusparseStatus_t status;
	cusparseHandle_t handle = 0;
	/* initialize cusparse library */
	status = cusparseCreate(&handle);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("CUSPARSE Library initialization failed");
	}
	//-------------------------------Preparing A' in colum orientated
	int ATcount[m];
	int actualCount[m];
	int rowCounts[m];
	for (int i = 0; i < m; i++) {
		ATcount[i] = 0;
		actualCount[i] = 0;
	}
	for (int i = 0; i < nnzElements; i++) {
		ATcount[Row_IDX[i] - 1]++; // Shift from 1-based to 0 based
	}
	rowCounts[0] = ATcount[0];
	for (int i = 1; i < m; i++) {
		int tmpCount = ATcount[i];
		rowCounts[i] = ATcount[i];
		ATcount[i] += ATcount[i - 1];
		actualCount[i] = ATcount[i] - tmpCount;
	}
	for (int i = 0; i < m; i++) {
		ATcount[i] = actualCount[i];
	}
	float ATCB[nnzElements];
	int ColAT[nnzElements];
	for (int i = 0; i < n; i++) {
		for (int j = 4 * i; j < 4 * i + 4; j++) {
			int tmprow = Row_IDX[j] - 1;
			ColAT[actualCount[tmprow]] = i;
			ATCB[actualCount[tmprow]] = A_TTD[j];
			actualCount[tmprow]++;
		}
	}
	//-------------------------------Alokacia device memroies
	float* A_d;
	int* C_idx_d;
	cudaMalloc((void**) &A_d, nnzElements * sizeof(float));
	checkCUDAError("kernel execution Alloc A_d");
	cudaMalloc((void**) &C_idx_d, nnzElements * sizeof(int));
	checkCUDAError("kernel execution Alloc R_IDX");
	int * n_d;
	cudaMalloc((void**) &n_d, 1 * sizeof(int));
	checkCUDAError("kernel execution Alloc ...");
	float* derivatives_d;
	cudaMalloc((void**) &derivatives_d, n * sizeof(float));
	checkCUDAError("kernel execution Alloc g_d");
	float* L_d;
	cudaMalloc((void**) &L_d, n * sizeof(float));
	checkCUDAError("kernel execution Alloc Li_d");

	int* ATcount_d;
	cudaMalloc((void**) &ATcount_d, m * sizeof(int));
	checkCUDAError("kernel execution Alloc AtCount_d");

	float* x_d;
	cudaMalloc((void**) &x_d, n * sizeof(float));
	checkCUDAError("kernel execution Alloc x_d");
	float* lambda_d;
	cudaMalloc((void**) &lambda_d, 1 * sizeof(float));
	checkCUDAError("kernel execution Alloc lambda_d");
	//	float* Li_d;
	//	cudaMalloc((void**) &Li_d, n * sizeof(float));
	//	checkCUDAError("kernel execution Alloc Li_d");
	print_time_message(&t1, "Device memory allocated");
	//-------------------------------Copy data
	cudaMemcpy(A_d, &ATCB[0], nnzElements * sizeof(float),
			cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy A_d");
	cudaMemcpy(C_idx_d, &ColAT[0], nnzElements * sizeof(int),
			cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy R");
	cudaMemcpy(derivatives_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution compy ....");
	cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution compy ....");
	cudaMemcpy(n_d, &n, 1 * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution compy ....");
	cudaMemcpy(L_d, L, n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution compy ....");
	for (int i = m - 1; i > 0; i--) {
		actualCount[i] -= actualCount[i - 1];
	}

	//	for (int i=0;i<m;i++)
	//{
	//	printf("AT count[%d]=%d\n",i,actualCount[i]);
	//}
	cudaMemcpy(ATcount_d, actualCount, m * sizeof(int), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution compy ....");

	print_time_message(&t1, "Data coppied");
	//-------------------------------Prepare of Derivatives vector on device
	for (int i = 0; i < m; i++) {
		if (g[i] != 0) {
			cusparseStatus_t copystatus = cusparseSaxpyi(handle, rowCounts[i],
					g[i], &A_d[ATcount[i]], &C_idx_d[ATcount[i]],
					derivatives_d, CUSPARSE_INDEX_BASE_ZERO);
			if (copystatus != CUSPARSE_STATUS_SUCCESS) {
				printf("Nastala chyba!");
			}
		}
	}
	print_time_message(&t1, "Derivatives vector on device");
	//---------------------------------------------------------------------
	float lambda = 0.0001;
	//------------------------------Serial Benchmark-----------------------

	float derivatives[n];
	cudaMemcpy(&derivatives[0], derivatives_d, n * sizeof(float),
			cudaMemcpyDeviceToHost);
	print_time_message(&t1, "Initial Shrink Start");
	int max_IDX = 0;
	float max_VAL = 0;
	float optimalStep = 0;
	float TVALUES[n];
	float EVALUES[n];
	for (int j = 0; j < n; j++) {
		float LLocal = L[j];
		float Li = 1 / LLocal;
		float xLocal = x[j];
		float lambdaOverL = lambda * Li;
		float alpha = derivatives[j] * Li;
		float deltaL = xLocal - alpha - lambdaOverL;
		float deltaR = xLocal - alpha + lambdaOverL;
		float t;
		if (deltaL > 0)
			t = deltaL - xLocal;
		else if (deltaR < 0)
			t = deltaR - xLocal;
		else
			t = -xLocal;
		float tmpEnergy = t * t * LLocal;
		TVALUES[j] = t;
		EVALUES[j] = tmpEnergy;
		if (tmpEnergy > max_VAL) {
			optimalStep = t;
			max_VAL = tmpEnergy;
			max_IDX = j;
		}
	}
	print_time_message(&t1, "Initial Shrink End");
	print_time_message(&t1, "Start serial Code");
	float setialExecutionTime = 0;
	int setialItetaions = 10;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int i = 0; i < setialItetaions; i++) {
		//		printf("optimal t_idx=%d, tval=%f\n", max_IDX, TVALUES[max_IDX]);
		// Update
		x[max_IDX] += TVALUES[max_IDX];
		optimalStep = TVALUES[max_IDX];
		for (int ki = max_IDX * 4; ki < max_IDX * 4 + 4; ki++) {
			for (int jj = ATcount[Row_IDX[ki] - 1]; jj < ATcount[Row_IDX[ki]
					- 1] + rowCounts[Row_IDX[ki] - 1]; jj++) {
				int j = ColAT[jj];
				derivatives[j] += optimalStep * A_TTD[ki] * ATCB[jj];
				float LLocal = L[j];
				float Li = 1 / LLocal;
				float xLocal = x[j];
				float lambdaOverL = lambda * Li;
				float alpha = derivatives[j] * Li;
				float deltaL = xLocal - alpha - lambdaOverL;
				float deltaR = xLocal - alpha + lambdaOverL;
				float t;
				if (deltaL > 0)
					t = deltaL - xLocal;
				else if (deltaR < 0)
					t = deltaR - xLocal;
				else
					t = -xLocal;
				TVALUES[j] = t;
				EVALUES[j] = t * t * LLocal;
			}
		}
		max_VAL = 0;
		max_IDX = 0;
		for (int j = 0; j < n; j++) {
			if (EVALUES[j] > max_VAL) {
				max_VAL = EVALUES[j];
				max_IDX = j;
			}
		}

	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&setialExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	float setialIterationPerSec = setialItetaions / setialExecutionTime * 1000;
	printf("Serial execution time:%f ms = %f it/sec \n", setialExecutionTime,
			setialIterationPerSec);
	print_time_message(&t1, "Serial time for 10 iterations");
	//-------------------------------Computation
	float* T_dev;
	float* E_dev;
	cublasAlloc(n, sizeof(float), (void**) &T_dev);
	checkCUDAError("Alokacia T_dev");
	cublasAlloc(n, sizeof(float), (void**) &E_dev);
	checkCUDAError("Alokacia E_dev");

	int doworking = 1;
	int iterations = 1;
	float time;
	cusparseStatus_t copystatus;
	while (doworking == 1) {
		cout << "Current lambda is " << lambda
				<< ". Do you want to change it? (y/n): ";
		string doContinue;
		cin >> doContinue;
		if (doContinue == "y") {
			cout << "Enter lambda: ";
			cin >> lambda;
		}
		cout << "Enter number of iterations: ";
		cin >> iterations;
		cudaMemcpy(lambda_d, &lambda, 1 * sizeof(float), cudaMemcpyHostToDevice);

		dim3 dimBlock( TOTALTHREDSPERBLOCK);
		dim3 dimGridRCDM( 1, 1+n/(TOTALTHREDSPERBLOCK));
		float tPoint[1];
		int maxIndex = 10;
		ShrinkKernel<<< dimGridRCDM ,dimBlock >>>(T_dev, E_dev, derivatives_d, L_d,x_d,
				lambda, n);

		int maxShrinkSubset = 0;
		for (int i = 0; i < m; i++) {
			if (rowCounts[i] > maxShrinkSubset)
				maxShrinkSubset = rowCounts[i];
		}
		printf("Max shrink subset %d\n", maxShrinkSubset);
		dim3 dimGridShrinkSubset( 1, 1+maxShrinkSubset/(TOTALTHREDSPERBLOCK));
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		for (int i = 0; i < iterations; i++) {
//									ShrinkKernel<<< dimGridRCDM ,dimBlock >>>(T_dev, E_dev, derivatives_d, L_d,x_d,
//											lambda, n);


			//maxIndex=10;
			maxIndex = cublasIsamax(n, E_dev, 1);
			maxIndex = maxIndex - 1;

//									printf("%d;Selected index:%d\n", i, maxIndex);
			IncreaseElement<<< 1 ,1 >>>(x_d, maxIndex, T_dev);
			checkCUDAError("IncreaseElement");
			cublasGetVector(1, sizeof(float), &T_dev[maxIndex], 1, tPoint, 1);
			checkCUDAError("get T");
//			printf("optimal t_idx=%d and value = %f\n", maxIndex, tPoint[0]);
			for (int ki = maxIndex * 4; ki < maxIndex * 4 + 4; ki++) {
				//								printf("pridat %f   %f   %d\n", tPoint[0], A_TTD[ki],Row_IDX[ki]-1);
				copystatus = cusparseSaxpyi(handle, rowCounts[Row_IDX[ki] - 1],
						tPoint[0] * A_TTD[ki], &A_d[ATcount[Row_IDX[ki] - 1]],
						&C_idx_d[ATcount[Row_IDX[ki] - 1]], derivatives_d,
						CUSPARSE_INDEX_BASE_ZERO);
				if (copystatus != CUSPARSE_STATUS_SUCCESS)
					printf("Nastala chyba pri CuSparse!\n");

//				cudaThreadSynchronize();
				ShrinkKernelSubset<<< dimGridShrinkSubset ,dimBlock >>>(T_dev, E_dev, derivatives_d,
						L_d, x_d, lambda, &ATcount_d[Row_IDX[ki] - 1], &C_idx_d[ATcount[Row_IDX[ki] - 1]]);
				//
				checkCUDAError("Shrink subset");
//				cudaThreadSynchronize();
				//				ShrinkKernelSubset<<< dimGridRCDM ,dimBlock >>>(T_dev, E_dev, derivatives_d,
				//										L_d, x_d, lambda, n, &C_idx_d[0]);


			}
			//			cudaThreadSynchronize();
			//			cudaMemcpy(x, derivatives_d, n * sizeof(float),
			//					cudaMemcpyDeviceToHost);
			//			checkCUDAError("kernel execution  : copy data Derivatives");
			//			for (int j = 0; j < n; j++) {
			//				printf("der[%d]=%f\n", j, x[j]);
			//			}
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("pridane  %f   \n", tPoint[0]);
		printf("trvanie %f ms ,  %f iter/sec speedUp:%f \n", time, iterations
				/ time * 1000, iterations / time * 1000 / setialIterationPerSec);
		//			double parallelPower = totalthreadsOnBlocks;
		//			parallelPower = parallelPower / time;
		//			parallelPower = parallelPower * 1000 * NMAXITERKernel;
		//			printf(
		//					"Elapset time: %f ms; %1.2f iterations/sec; speedup: %1.4f\n",
		//					time, parallelPower, parallelPower / serialPower);
		//
		//			L1Norm = cublasSasum(n, x_d, 1);
		//			residualsSum = cublasSnrm2(m, g_d, 1);
		//			objectiveValue = 0.5 * residualsSum * residualsSum;
		//			objectiveValue += lambda * L1Norm;
		//			printf("%d:L1 norm = %f, redisuals=%f,  objective=%1.16f\n", i,
		//					L1Norm, residualsSum, objectiveValue);
		cout << "Save particular solution to file? (y/n): ";
		cin >> doContinue;
		if (doContinue == "y") {
			float treshHold;
			cout << "Enter treshhold for x: ";
			cin >> treshHold;
			int writtenBars = 0;
			FILE *fp;
			fp = fopen("/tmp/ttd.txt", "w");
			cudaMemcpy(x, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
			checkCUDAError("kernel execution compy ....");
			for (int i = 0; i < n; i++) {
				if (abs(x[i]) > treshHold) {
					writtenBars++;
					fprintf(fp, "%d,%d,%d,%d,%f\n", nodeDescription[i * 4],
							nodeDescription[i * 4 + 1], nodeDescription[i * 4
									+ 2], nodeDescription[i * 4 + 3], x[i]);
				}
			}
			fclose(fp);
			printf("Number of written bars:%d\n", writtenBars);
		}
		cout << "Continue? (y/n): ";
		cin >> doContinue;
		if (doContinue == "n") {
			doworking = 0;
		}
	}
	print_time_message(&t1, "Computation");

	//-------------------------------DeAllocation
	cudaFree(lambda_d);
	cudaFree(A_d);
	cudaFree(C_idx_d);
	cudaFree(n_d);
	cudaFree(x_d);
	cudaFree(derivatives_d);
	cudaFree(L_d);
	cudaFree(ATcount_d);
	print_time_message(&t1, "Device memory DeAllocated");
}

int main(void) {

	greedyTTD();
	//	calculateTTDProblem();
	return 1;
}

void generateTTDProblem(int row, int col, int exampleType, int* mOut,
		int* nOut, float** A, int** ColIDX, int ** RowIDX, int * nnzElements,
		int* boundary, int boundarySize, int** nodeDescription, int rowScale) {
	float kappa = 1;
	float scale = sqrt(kappa);
	int m = row * col;
	float A_values[MaxTTD_NNZ_ELEMENTS];
	int Rows_A[MaxTTD_NNZ_ELEMENTS];
	int Cols_A[MaxTTD_NNZ_ELEMENTS];
	*nodeDescription = new int[MaxTTD_NNZ_ELEMENTS*4];
	int nodeDescriptionId = 0;
	int nnz = 0;
	int node = 0;
	int tt = col;
	if (row > tt)
		tt = row;
	int GSDS[tt + 1][tt + 1];
	for (int i = 1; i <= tt; i++) {
		for (int j = i; j <= tt; j++) {
			GSDS[i][j] = GCD(i, j);
			GSDS[j][i] = GCD(i, j);
		}
	}
	int betta[2];
	float beta[2];
	for (int j = 1; j <= col; j++) {
		for (int i = 1; i <= col; i++) {
			for (int k = 1; k <= row; k++) {
				for (int l = 1; l <= col - j; l++) {
					betta[0] = rowScale * l;// for node (i,j) we add bars to all (k,j+i)
					betta[1] = -k + i;
					scaleBetta(betta, beta, scale);
					betta[0] = l;
					if (col == j)
						continue;

					if (l > 1) {
						int skip = 0;
						int ta = minimabs(betta);
						int tb = maximabs(betta);
						if (ta == 0) {
							skip = 1;
						} else {
							if (GSDS[ta][tb] > 1) {
								skip = 1;
							}
						}
						if (skip)
							continue;
					}
					int totalMatchA = 0;
					int totalMatchB = 0;
					for (int bi = 0; bi < boundarySize; bi++) {
						if (boundary[bi] == row * (j - 1) + i)
							totalMatchA++;
						if (boundary[bi] == row * (j) + k + (l - 1) * row)
							totalMatchB++;
					}
					if (totalMatchA + totalMatchB < 2) {
						node++;
						int tmp = row * (j - 1) + i;
						A_values[nnz] = -(1 - totalMatchA) * beta[0];
						Rows_A[nnz] = tmp;
						Cols_A[nnz] = node;
						nnz++;
						A_values[nnz] = -(1 - totalMatchA) * beta[1];
						Rows_A[nnz] = tmp + m;
						Cols_A[nnz] = node;
						nnz++;
						tmp = row * (j) + k + (l - 1) * row;
						A_values[nnz] = (1 - totalMatchB) * beta[0];
						Rows_A[nnz] = tmp;
						Cols_A[nnz] = node;
						nnz++;
						A_values[nnz] = (1 - totalMatchB) * beta[1];
						Rows_A[nnz] = tmp + m;
						Cols_A[nnz] = node;
						nnz++;

						(*nodeDescription)[nodeDescriptionId] = i;
						(*nodeDescription)[nodeDescriptionId + 1] = j;
						(*nodeDescription)[nodeDescriptionId + 2] = k;
						(*nodeDescription)[nodeDescriptionId + 3] = l + j;
						nodeDescriptionId += 4;
					}
				}
			}
			if (i < row) {
				int tmp = i + (j - 1) * row;
				int totalMatchA = 0;
				int totalMatchB = 0;
				for (int bi = 0; bi < boundarySize; bi++) {
					if (boundary[bi] == tmp)
						totalMatchA++;
					if (boundary[bi] == tmp + 1)
						totalMatchB++;
				}
				if (totalMatchA + totalMatchB < 2) {
					node = node + 1;
					A_values[nnz] = -(1 - totalMatchA);
					Rows_A[nnz] = tmp + m;
					Cols_A[nnz] = node;
					nnz++;
					A_values[nnz] = 0; //fake node
					Rows_A[nnz] = tmp + 1;
					Cols_A[nnz] = node;
					nnz++;
					A_values[nnz] = 0; //fake node
					Rows_A[nnz] = tmp + 2;
					Cols_A[nnz] = node;
					nnz++;

					A_values[nnz] = (1 - totalMatchB);
					Rows_A[nnz] = tmp + m + 1;
					Cols_A[nnz] = node;
					nnz++;
					(*nodeDescription)[nodeDescriptionId] = i;
					(*nodeDescription)[nodeDescriptionId + 1] = j;
					(*nodeDescription)[nodeDescriptionId + 2] = i + 1;
					(*nodeDescription)[nodeDescriptionId + 3] = j;
					nodeDescriptionId += 4;
				}
			}
		}
	}*A = new float[nnz];
	*ColIDX = new int[nnz];
	*RowIDX = new int[nnz];
	for (int i = 0; i < nnz; i++) {
		(*A)[i] = A_values[i];
		(*ColIDX)[i] = Cols_A[i];
		(*RowIDX)[i] = Rows_A[i];
	}
	*nOut = node;
	*mOut = row * col * 2;
	*nnzElements = nnz;
}

void getBoundaryVector(int row, int col, int exampleType, int** boundary,
		int* boudarySize) {

	switch (exampleType) {
	case 1:
		*boundary = new int[row];
		for (int i = 0; i < row; i++) {
			(*boundary)[i] = i + 1;
		}
		//		boundaryIDX(1, i) = 1;
		//		boundaryIDX(2, i) = i;
		*boudarySize = row;
		break;
	case 2:
		*boundary = new int[4];
		(*boundary)[0] = row;
		(*boundary)[1] = row * col;
		(*boundary)[2] = row * col - row * (col / 3);
		(*boundary)[3] = row + row * (col / 3);
		*boudarySize = 4;
		break;
	case 3:
		*boundary = new int[4];
		(*boundary)[0] = row - 5;
		(*boundary)[1] = row * col - 5;
		(*boundary)[2] = row * col - row * (col / 3);
		(*boundary)[3] = row + row * (col / 3);
		*boudarySize = 4;
		break;

	case 5:
		*boundary = new int[0];
		*boudarySize = 0;
		break;
	case 7:
		*boundary = new int[2];
		(*boundary)[0] = row;
		(*boundary)[1] = row * col;
		*boudarySize = 2;
		break;

	default:
		break;
	}
}
//		if (boundarytype==3) %bridge
//
//		boundary(1,1)=r;
//		boundary(1, 4) = r + r * (floor(c / 3));
//		boundary(1, 3) = r * c - r * floor(c / 3);
//		boundary(1, 2) = r * c;
//
//		boundaryIDX(1, 1) = 1;
//		boundaryIDX(2, 1) = r;
//		boundaryIDX(1, 2) = c;
//		boundaryIDX(2, 2) = r;
//
//		boundaryIDX(1, 3) = floor(c / 3) + 1;
//		boundaryIDX(2, 3) = r;
//		boundaryIDX(1, 4) = c - floor(c / 3);
//		boundaryIDX(2, 4) = r;
//
//		end
//		if (boundarytype==4) %
//
//		boundary(1,1)=r;
//		% boundary(1,4)=r+r*(floor(c/5));
//		% boundary(1,3)=r*c-r*floor(c/5);
//		boundary(1, 2) = r * c;
//
//		boundaryIDX(1, 1) = 1;
//		boundaryIDX(2, 1) = r;
//		boundaryIDX(1, 2) = c;
//		boundaryIDX(2, 2) = r;
//
//		% boundaryIDX(1,3)=floor(c/5)+1;
//		% boundaryIDX(2,3)=r;
//		% boundaryIDX(1,4)=c-floor(c/5);
//		% boundaryIDX(2,4)=r;
//
//		end
//
//		if (boundarytype==5) %
//
//		end
//
//		if (boundarytype==6) %
//
//		boundary(1,1)=r;
//
//		boundary(1, 4) = r + r * (floor(c / 5));
//		boundary(1, 3) = r * c - r * floor(c / 5);
//		boundary(1, 5) = r + r * 2 * (floor(c / 5));
//		boundary(1, 6) = r * c - r * 2 * floor(c / 5);
//
//		boundary(1, 2) = r * c;
//
//		boundaryIDX(1, 1) = 1;
//		boundaryIDX(2, 1) = r;
//		boundaryIDX(1, 2) = c;
//		boundaryIDX(2, 2) = r;
//
//		boundaryIDX(1, 3) = floor(c / 5) + 1;
//		boundaryIDX(2, 3) = r;
//		boundaryIDX(1, 4) = c - floor(c / 5);
//		boundaryIDX(2, 4) = r;
//
//		boundaryIDX(1, 5) = 2 * floor(c / 5) + 1;
//		boundaryIDX(2, 5) = r;
//		boundaryIDX(1, 6) = c - 2 * floor(c / 5);
//		boundaryIDX(2, 6) = r;
//
//	end

void scaleBetta(int* betta, float* beta, float scale) {
	float tmp = scale / (betta[0] * betta[0] + betta[1] * betta[1]);
	beta[0] = betta[0] * tmp;
	beta[1] = betta[1] * tmp;
}

int GCD(int a, int b) {
	while (1) {
		a = a % b;
		if (a == 0)
			return b;
		b = b % a;

		if (b == 0)
			return a;
	}
}

void getForceVector(float ** d, int m, int r, int c, int exampleType) {
	//	int mid = r / 2;
	//	int midr =(r/2)+1;
	//	int midc = (c/2)+1;
	*d=new float[m];
	m = m / 2;
	int tmp;

	switch (exampleType) {
	case 1:
		//		tmp = r * (c - 1) + mid + 1;
		tmp = r * c;
		(*d)[tmp + m - 1] = -1;
		(*d)[tmp - 1] = 2;
		break;
	case 2:
		for (int cc = 2; cc < c-1; cc++) {
			(*d)[-2 + r * cc + m - 1] = -1;
		}
		break;
	case 3:
		for (int cc = 2; cc < c; cc++) {
			(*d)[-2 + r * cc + m - 1] = -1;
		}
		break;

	case 5:

		break;
	case 7:
			for (int cc = 3; cc < c-3+1; cc++) {
				(*d)[-5 + r * cc + m - 1] = -1;
			}
			break;
	default:
		break;
	}

	//	if (boundarytype==3)
	//	    midr = floor(r/2)+1
	//	    midc = floor(c/2)+1
	//
	//	    for cc=2:c-1
	//	        d(-2+r*cc+m) = -1;
	//	    end
	//	end
	//
	//	if (boundarytype==6)
	//	    midr = floor(r/2)+1
	//	    midc = floor(c/2)+1
	//
	//	    for cc=2:c-1
	//	        d(-1+r*cc+m) = -1;
	//	    end
	//	end
	//
	//	if (boundarytype==4)
	//	    midr = floor(r/2)+1
	//	    midc = floor(c/2)+1
	//
	//
	//
	//	    for cc=3:c-2
	//	        d(-12+r*cc+m) = -1;
	//	    end
	//
	//	   for asdf=1:13
	//	    for cc=6:c-5
	//	         %d(-8-asdf+r*cc+m) = -1;
	//	    end
	//	   end
	//
	//
	//	    for asdf=1:17
	//	        for cc=6:6
	//	         d(-12-asdf+r*cc+m) = -1;
	//	        if (asdf<17)
	//	         d(-12-asdf+r*cc) = (-1)^asdf;
	//	        end
	//	        end
	//	    end
	//
	//
	//	end


}

int maximabs(int * betta) {
	if (abs(betta[0]) >= abs(betta[1]))
		return abs(betta[0]);
	else
		return abs(betta[1]);
}

int minimabs(int * betta) {
	if (abs(betta[0]) <= abs(betta[1]))
		return abs(betta[0]);
	else
		return abs(betta[1]);
}
void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}
__global__ void setup_kernel(curandState *state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	/* Each thread gets different seed, a different sequence number, no offset */
	curand_init(i, i, 0, &state[i]);
}
