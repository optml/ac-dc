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
#include <sstream>

#include "ttd.h"
#include "kernels.h"
#include "helpers.h"

#include "SPBLASTK/include/spblas.h"

#define TOTALTHREDSPERBLOCK 256
#define COLUMNLENGTH 4
#define NMAXITERKernel 100
#define ROWSCALE 2

using namespace std;

__global__ void RCDMKernel(float *A, int*R_Idx, int* n, float*residuals,
		float*x, float * lambda, float* Li, curandState* cstate);

void timeComparison(int row, int col, int exampleType, float executionTime,
		float lambda) {
	double objVal = 0;
	int dim1 = 14;
	int dim2 = 1;
	float PRCDMExecutionTime = executionTime;
	float SRCDMExecutionTime = executionTime;

	int m, n;
	clock_t t1;
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
	float residuals[m];
	float b[m];
	float* redisuals_dev;
	getForceVector(&g, m, row, col, exampleType);
	for (int i = 0; i < m; i++) {
		g[i] = -g[i];
		residuals[i] = g[i];
		b[i] = -g[i];
	}


	print_time_message(&t1, "Force vector obtained");
	//-----------------------------------------
	FILE *fp;
	//		fp = fopen("/tmp/asdf", "w");
	//
	//		for (int i = 0; i < nnzElements; i++) {
	//			fprintf(fp,"%d,%d,%f\n", Row_IDX[i], Col_IDX[i], A_TTD[i]);
	//		}
	//		fclose(fp);
	//		fp = fopen("/tmp/ttd_vectorB.csv", "w");
	//
	//		for (int i = 0; i < m; i++) {
	//			fprintf(fp,"%d,%f\n", i,-g[i]);
	//		}
	//		fclose(fp);
	//	print_time_message(&t1, "Data saved");
	//-----------------------------------------
	float x[n];
	float L[n];
	float Li[n];
	for (int i = 0; i < n; i++) {
		x[i] = 0;
		L[i] = 0;
		for (int k = 4 * i; k < 4 * i + 4; k++) {
			L[i] += A_TTD[k] * A_TTD[k];
		}
		Li[i] = 1 / L[i];
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
	fp = fopen("/tmp/ttd_log.txt", "w");
	//-------------------------------Inicializacia CuSpare Library
	/* allocate GPU memory and copy the matrix and vectors into it */
	cusparseStatus_t status;
	cusparseHandle_t handle = 0;
	/* initialize cusparse library */
	status = cusparseCreate(&handle);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("CUSPARSE Library initialization failed");
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
		float mlt2 = 1;
		if (deltaL > 0)
			t = deltaL - xLocal;
		else if (deltaR < 0)
			t = deltaR - xLocal;
		else {
			t = -xLocal;
			mlt2 = 0;
		}
		float tmpEnergy = t * t * LLocal;
		TVALUES[j] = t;
		//		EVALUES[j] = tmpEnergy / 2;//+ lambda*x[j]*( signbit(x[j]) -signbit(x[j]+TVALUES[j])    );

		EVALUES[j] = mlt2 * (tmpEnergy / 2 + lambda * xLocal * (signbit(xLocal)
				- signbit(xLocal + t))) + (1 - mlt2) * (lambda * xLocal
				* signbit(xLocal) - derivatives[j] * t - LLocal * t * t / 2);

		if (tmpEnergy > max_VAL) {
			optimalStep = t;
			max_VAL = tmpEnergy;
			max_IDX = j;
		}
	}
	print_time_message(&t1, "Initial Shrink End");
	print_time_message(&t1, "Start serial Code");
	float setialExecutionTime = 0;
	long long setialItetaions = 0;

	clock_t t_start;
	t_start = clock();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	while (getTotalElapsetTime(&t_start) < executionTime) {
		// 			printf("optimal t_idx=%d, tval=%f, elapsedTime:%f\n", max_IDX, TVALUES[max_IDX],getTotalElapsetTime(&t_start));
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
				float mlt2 = 1;
				if (deltaL > 0)
					t = deltaL - xLocal;
				else if (deltaR < 0)
					t = deltaR - xLocal;
				else {
					t = -xLocal;
					mlt2 = 0;
				}
				float tmpEnergy = t * t * LLocal;
				TVALUES[j] = t;
				//				EVALUES[j] = tmpEnergy / 2;//+ lambda*x[j]*( signbit(x[j]) -signbit(x[j]+TVALUES[j])    );

				EVALUES[j] = mlt2 * (tmpEnergy / 2 + lambda * xLocal
						* (signbit(xLocal) - signbit(xLocal + t))) + (1 - mlt2)
						* (lambda * xLocal * signbit(xLocal) - derivatives[j]
								* t - LLocal * t * t / 2);
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
		setialItetaions++;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&setialExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	float setialIterationPerSec = setialItetaions / setialExecutionTime * 1000;
	printf("Serial execution time:%f ms = %f it/sec \n", setialExecutionTime,
			setialIterationPerSec);
	fprintf(fp, "Serial execution time:%f ms = %f it/sec \n",
			setialExecutionTime, setialIterationPerSec);

	printf("serial greedy iterations:%d\n", setialItetaions);
	fprintf(fp, "serial greedy iterations:%d\n", setialItetaions);
	saveSolutionIntoFile("/tmp/ttd_g_serial.txt", x, n, nodeDescription, 0);
	objVal = computeTTDObjectiveValue(A_TTD, Row_IDX, Col_IDX, x, n, m, b,
			nnzElements, lambda);
	printf("Obj Val: %1.16f\n", objVal);
	fprintf(fp, "Obj Val: %1.16f\n", objVal);

	//-------------------------------Computation
	float* T_dev;
	float* E_dev;
	cublasAlloc(n, sizeof(float), (void**) &T_dev);
	checkCUDAError("Alokacia T_dev");
	cublasAlloc(n, sizeof(float), (void**) &E_dev);
	checkCUDAError("Alokacia E_dev");

	float time;
	cusparseStatus_t copystatus;
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
	long long parallel_iterations = 0;
	t_start = clock();
	while (getTotalElapsetTime(&t_start) < executionTime) {
		for (int asdf = 0; asdf < 100; asdf++) {
			maxIndex = cublasIsamax(n, E_dev, 1);
			maxIndex = maxIndex - 1;
			//									printf("%d;Selected index:%d\n", i, maxIndex);
			IncreaseElementAndShrink<<< 1 ,1 >>>( maxIndex,T_dev, E_dev, derivatives_d, L_d,x_d,
					lambda, n );

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
			}
			parallel_iterations++;
		}
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("pridane  %f   \n", tPoint[0]);
	printf("trvanie %f ms ,  %f iter/sec speedUp:%f \n", time,
			parallel_iterations / time * 1000, parallel_iterations / time
					* 1000 / setialIterationPerSec);
	printf("parallel greedy iterations:%d\n", parallel_iterations);

	fprintf(fp, "trvanie %f ms ,  %f iter/sec speedUp:%f \n", time,
			parallel_iterations / time * 1000, parallel_iterations / time
					* 1000 / setialIterationPerSec);
	fprintf(fp, "parallel greedy iterations:%d\n", parallel_iterations);

	float treshHold = 0;
	cudaMemcpy(x, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("kernel execution compy ....");
	saveSolutionIntoFile("/tmp/ttd_g_parallel.txt", x, n, nodeDescription,
			treshHold);
	objVal = computeTTDObjectiveValue(A_TTD, Row_IDX, Col_IDX, x, n, m, b,
			nnzElements, lambda);
	printf("Obj Val: %1.16f\n", objVal);
	fprintf(fp, "Obj Val: %1.16f\n", objVal);

	print_time_message(&t1, "Computation");

	//-------------------------------DeAllocation
	cudaFree(ATcount_d);
	print_time_message(&t1, "Device memory DeAllocated");

	for (int i = 0; i < n; i++)
		x[i] = 0;

	cudaMalloc((void**) &redisuals_dev, m * sizeof(float));
	checkCUDAError("kernel execution Alloc residuals_d");
	cudaMemcpy(redisuals_dev, &residuals[0], m * sizeof(float),
			cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy residuals_d");
	cudaMemcpy(x_d, &x[0], n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy x");

	setialItetaions = 0;
	t_start = clock();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	while (getTotalElapsetTime(&t_start) < SRCDMExecutionTime) {
		float tmp = (float) rand() / RAND_MAX;
		int idx = (int) (n * tmp);
		tmp = 0;
		for (int j = idx * 4; j < idx * 4 + 4; j++) {
			//printf("A=%f, res=%f,colid=%d\n",A_TTD[j],residuals[Col_IDX[j]],Col_IDX[j]);
			tmp += A_TTD[j] * residuals[Row_IDX[j] - 1];
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
		//		printf("ID:%d, value%f\n",idx, x[idx]);
		for (int j = idx * 4; j < idx * 4 + 4; j++) {
			residuals[Row_IDX[j] - 1] += tmp * A_TTD[j];
		}
		setialItetaions++;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&setialExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	float serialItPsec = (float) setialItetaions / setialExecutionTime;
	printf("trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			setialItetaions / setialExecutionTime * 1000);
	printf("serial random iterations:%d\n", setialItetaions);
	fprintf(fp, "trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			setialItetaions / setialExecutionTime * 1000);
	fprintf(fp, "serial random iterations:%d\n", setialItetaions);

	saveSolutionIntoFile("/tmp/ttd_r_serial.txt", x, n, nodeDescription, 0);
	objVal = computeTTDObjectiveValue(A_TTD, Row_IDX, Col_IDX, x, n, m, b,
			nnzElements, lambda);
	printf("Obj Val: %1.16f\n", objVal);
	fprintf(fp, "Obj Val: %1.16f\n", objVal);

	//--------------------------------------------------
	// ============  Parallel Random


	dim3 dimBlockRCD2( TOTALTHREDSPERBLOCK);
	dim3 dimGridRCDM2( dim1, dim2);

	curandState *devStates;
	cudaMalloc((void **) &devStates, dim1 * dim2 * TOTALTHREDSPERBLOCK
			* sizeof(curandState));
	setup_kernel<<< dimGridRCDM2, dimBlockRCD2 >>>(devStates);
	checkCUDAError("Inicializacia ranom states");
	cudaFree(L_d);
	checkCUDAError("Dealocation");
	float* Li_dev;
	cudaMalloc((void**) &Li_dev, n * sizeof(float));
	checkCUDAError("kernel execution Alloc Li_d");
	cudaMemcpy(Li_dev, &Li[0], n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Coppy Li_d");

	cudaMemcpy(A_d, A_TTD, nnzElements * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy A_d");

	cudaFree(C_idx_d);
	checkCUDAError("Dealocation");

	int* R_idx_d;
	cudaMalloc((void**) &R_idx_d, nnzElements * sizeof(int));
	checkCUDAError("kernel execution Alloc R_idx_d");

	cudaMemcpy(R_idx_d, Row_IDX, nnzElements * sizeof(int),
			cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy Ridx_d");
	print_time_message(&t1, "Idem spustat paralelny RCDM");
	parallel_iterations = 0;
	float parallelExecutionTime;
	t_start = clock();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	float rcdmBufferedTime = 0;
	while (rcdmBufferedTime < PRCDMExecutionTime) {
		for (int koko = 0; koko < 4000; koko++) {
			RCDMKernel<<< dimGridRCDM2, dimBlockRCD2 >>>(A_d, R_idx_d, n_d, redisuals_dev, x_d, lambda_d, Li_dev,
					devStates);

			parallel_iterations += dim1 * dim2;
		}
		rcdmBufferedTime += getTotalElapsetTime(&t_start);
		float norm_g = cublasSnrm2(m, redisuals_dev, 1);
		float norm_x = cublasSasum(n, x_d, 1);
		double rcdmObjectiveValue = norm_x * lambda + 0.5 * norm_g * norm_g;
		//			printf("Current value = %1.16f\n", rcdmObjectiveValue);
		cudaMemcpy(x, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("kernel execution compy ....");
		double DoubleObjectiveValue = computeTTDObjectiveValue(A_TTD, Row_IDX,
				Col_IDX, x, n, m, b, nnzElements, lambda);
		printf("Double objective value: %1.16f\n", DoubleObjectiveValue);
		printf("Error: %1.16f\n",
				abs(DoubleObjectiveValue - rcdmObjectiveValue));

		updateG(A_TTD, Row_IDX, Col_IDX, x, n, m, b, nnzElements, residuals);

		cudaMemcpy(redisuals_dev, &residuals[0], m * sizeof(float),
				cudaMemcpyHostToDevice);
		checkCUDAError("kernel execution Copy residuals_d");

		printf("RESTART\n");
		t_start = clock();
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&parallelExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("parallel random iterations:%d toital %d\n", parallel_iterations,
			parallel_iterations * NMAXITERKernel * TOTALTHREDSPERBLOCK);
	printf("trvanie %f ms ,  %f iter/sec , speedup:%f \n",
			parallelExecutionTime, parallel_iterations / parallelExecutionTime
					* 1000 * NMAXITERKernel * TOTALTHREDSPERBLOCK,
			TOTALTHREDSPERBLOCK * NMAXITERKernel * parallel_iterations
					/ parallelExecutionTime / serialItPsec);
	fprintf(fp, "parallel random iterations:%d toital %d\n",
			parallel_iterations, parallel_iterations * NMAXITERKernel
					* TOTALTHREDSPERBLOCK);
	fprintf(fp, "trvanie %f ms ,  %f iter/sec , speedup:%f \n",
			parallelExecutionTime, parallel_iterations / parallelExecutionTime
					* 1000 * NMAXITERKernel * TOTALTHREDSPERBLOCK,
			TOTALTHREDSPERBLOCK * NMAXITERKernel * parallel_iterations
					/ parallelExecutionTime / serialItPsec);

	print_time_message(&t1, "END spustat paralelny RCDM");
	cudaMemcpy(x, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("kernel execution compy ....");
	saveSolutionIntoFile("/tmp/ttd_r_parallel.txt", x, n, nodeDescription,
			treshHold);
	objVal = computeTTDObjectiveValue(A_TTD, Row_IDX, Col_IDX, x, n, m, b,
			nnzElements, lambda);
	printf("Obj Val: %1.16f\n", objVal);
	fprintf(fp, "Obj Val: %1.16f\n", objVal);

	//------------------------------------------------------
	//===================== Nesterov Composite Accelerated
	float A_nest = 0;
	float L_nest = L[0];
	float mu = 0;
	float v[n];
	float y[n];
	float derv[n];
	float derv2[n];
	float dervHistory[n];
	for (int i = 0; i < n; i++) {
		x[i] = 0;
		v[i] = 0;
		derv[i] = 0;
		dervHistory[i] = 0;
	}
	float gamma_u = 2;
	float gamma_d = gamma_u;
	float quadSolTmp = 0;

	parallel_iterations = 0;
	t_start = clock();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	while (getTotalElapsetTime(&t_start) < executionTime) {
		int doRepeat = 1;
		while (doRepeat) {
			quadSolTmp = 2 * (1 + A_nest * mu) / L_nest;
			float a = (quadSolTmp + sqrt(quadSolTmp * quadSolTmp + 4 * A_nest
					* quadSolTmp)) / 2;
			float convCombA = A_nest / (a + A_nest);
			float convComba = a / (a + A_nest);
			for (int i = 0; i < n; i++) {
				y[i] = convCombA * x[i] + convComba * v[i];
			}
			for (int i = 0; i < n; i++) {
				derv[i] = 0;
			}
			for (int i = 0; i < m; i++) {
				g[i] = -b[i];
			}
			for (int i = 0; i < nnzElements; i++) // derv = Ax-b
			{
				g[Row_IDX[i] - 1] += A_TTD[i] * y[Col_IDX[i] - 1];
			}
			for (int i = 0; i < nnzElements; i++) {
				derv[Col_IDX[i] - 1] += g[Row_IDX[i] - 1] * A_TTD[i];
			}
			float LLocal = L_nest;
			float Li = 1 / LLocal;
			float lambdaOverL = lambda * Li;
			for (int j = 0; j < n; j++) {
				float xLocal = x[j];
				float alpha = derv[j] * Li;
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
			}
			for (int j = 0; j < n; j++) {
				derv2[j] = -L_nest * TVALUES[j] - derv[j];
				derv[j] = 0;
			}
			for (int i = 0; i < m; i++) {
				g[i] = -b[i];
			}
			for (int i = 0; i < nnzElements; i++) // derv = Ax-b
			{
				g[Row_IDX[i] - 1] += A_TTD[i] * (y[Col_IDX[i] - 1]
						+ TVALUES[Col_IDX[i] - 1]);
			}
			for (int i = 0; i < nnzElements; i++) {
				derv[Col_IDX[i] - 1] += g[Row_IDX[i] - 1] * A_TTD[i];
			}
			for (int i = 0; i < n; i++) {
				derv[i] += derv2[i];
			}
			float LS = 0;
			float RS = 0;
			for (int i = 0; i < n; i++) {
				LS -= derv[i] * TVALUES[i];
				RS += derv[i] * derv[i];
			}
			RS = RS / L_nest;
			if (LS < RS) {
				L_nest = L_nest * gamma_u;
				doRepeat = 1;
			} else {
				doRepeat = 0;
				A_nest += a;
				for (int i = 0; i < n; i++) {
					x[i] = y[i] + TVALUES[i];
				}
				L_nest = L_nest / gamma_d;
				float ratioA = (A_nest - a) / A_nest;
				float ratioB = (a) / A_nest;
				for (int i = 0; i < n; i++) {
					dervHistory[i] = ratioA * dervHistory[i] + ratioB
							* (derv[i] - derv2[i]);
				}

				float LLocal = 1 / A_nest;
				float Li = A_nest;
				float lambdaOverL = lambda * Li;
				for (int i = 0; i < n; i++) {
					float alpha = dervHistory[i] * Li;
					float deltaL = -alpha - lambdaOverL;
					float deltaR = -alpha + lambdaOverL;
					float t;
					if (deltaL > 0)
						t = deltaL;
					else if (deltaR < 0)
						t = deltaR;
					else
						t = 0;
					v[i] = t;

				}

			}
		}
		parallel_iterations++;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&setialExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("serial AC iterations:%d\n", parallel_iterations);

	printf("trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			parallel_iterations / setialExecutionTime * 1000);
	fprintf(fp, "serial AC iterations:%d\n", parallel_iterations);

	fprintf(fp, "trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			parallel_iterations / setialExecutionTime * 1000);
	saveSolutionIntoFile("/tmp/ttd_ac.txt", x, n, nodeDescription, treshHold);
	objVal = computeTTDObjectiveValue(A_TTD, Row_IDX, Col_IDX, x, n, m, b,
			nnzElements, lambda);
	printf("Obj Val: %1.16f\n", objVal);
	fprintf(fp, "Obj Val: %1.16f\n", objVal);

	//-------------------------------------------------------

	cudaFree(Li_dev);
	checkCUDAError("Dealocationa");
	cudaFree(R_idx_d);
	checkCUDAError("Dealocationb");
	cudaFree(redisuals_dev);
	checkCUDAError("Dealocationc");
	cudaFree(lambda_d);
	checkCUDAError("Dealocationd");
	cudaFree(A_d);
	checkCUDAError("Dealocatione");
	cudaFree(n_d);
	checkCUDAError("Dealocationg");
	cudaFree(x_d);
	checkCUDAError("Dealocationh");
	cudaFree(derivatives_d);
	checkCUDAError("Dealocationi");

	print_time_message(&t1, "Device memory DeAllocated");
	fclose(fp);
}

void oneDayPR(int row, int col, int exampleType, float executionTime,
		float lambda) {

	int dim1 = 14;
	int dim2 = 1;
	float PRCDMExecutionTime = 3600;
	float SRCDMExecutionTime = 0;

	int m, n;
	clock_t t1;
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
	//-----------------------------------------
	FILE *fp;
	//		fp = fopen("/tmp/asdf", "w");
	//
	//		for (int i = 0; i < nnzElements; i++) {
	//			fprintf(fp,"%d,%d,%f\n", Row_IDX[i], Col_IDX[i], A_TTD[i]);
	//		}
	//		fclose(fp);
	//		fp = fopen("/tmp/ttd_vectorB.csv", "w");
	//
	//		for (int i = 0; i < m; i++) {
	//			fprintf(fp,"%d,%f\n", i,-g[i]);
	//		}
	//		fclose(fp);
	//	print_time_message(&t1, "Data saved");
	//-----------------------------------------
	float x[n];
	float L[n];
	float Li[n];
	for (int i = 0; i < n; i++) {
		x[i] = 0;
		L[i] = 0;
		for (int k = 4 * i; k < 4 * i + 4; k++) {
			L[i] += A_TTD[k] * A_TTD[k];
		}
		Li[i] = 1 / L[i];
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
	fp = fopen("/tmp/ttd_log.txt", "w");
	//-------------------------------Inicializacia CuSpare Library
	/* allocate GPU memory and copy the matrix and vectors into it */
	cusparseStatus_t status;
	cusparseHandle_t handle = 0;
	/* initialize cusparse library */
	status = cusparseCreate(&handle);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("CUSPARSE Library initialization failed");
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
		float mlt2 = 1;
		if (deltaL > 0)
			t = deltaL - xLocal;
		else if (deltaR < 0)
			t = deltaR - xLocal;
		else {
			t = -xLocal;
			mlt2 = 0;
		}
		float tmpEnergy = t * t * LLocal;
		TVALUES[j] = t;
		//		EVALUES[j] = tmpEnergy / 2;//+ lambda*x[j]*( signbit(x[j]) -signbit(x[j]+TVALUES[j])    );

		EVALUES[j] = mlt2 * (tmpEnergy / 2 + lambda * xLocal * (signbit(xLocal)
				- signbit(xLocal + t))) + (1 - mlt2) * (lambda * xLocal
				* signbit(xLocal) - derivatives[j] * t - LLocal * t * t / 2);

		if (tmpEnergy > max_VAL) {
			optimalStep = t;
			max_VAL = tmpEnergy;
			max_IDX = j;
		}
	}
	print_time_message(&t1, "Initial Shrink End");
	print_time_message(&t1, "Start serial Code");
	float setialExecutionTime = 0;
	long long setialItetaions = 0;

	clock_t t_start;
	t_start = clock();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	while (getTotalElapsetTime(&t_start) < executionTime) {
		// 			printf("optimal t_idx=%d, tval=%f, elapsedTime:%f\n", max_IDX, TVALUES[max_IDX],getTotalElapsetTime(&t_start));
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
				float mlt2 = 1;
				if (deltaL > 0)
					t = deltaL - xLocal;
				else if (deltaR < 0)
					t = deltaR - xLocal;
				else {
					t = -xLocal;
					mlt2 = 0;
				}
				float tmpEnergy = t * t * LLocal;
				TVALUES[j] = t;
				//				EVALUES[j] = tmpEnergy / 2;//+ lambda*x[j]*( signbit(x[j]) -signbit(x[j]+TVALUES[j])    );

				EVALUES[j] = mlt2 * (tmpEnergy / 2 + lambda * xLocal
						* (signbit(xLocal) - signbit(xLocal + t))) + (1 - mlt2)
						* (lambda * xLocal * signbit(xLocal) - derivatives[j]
								* t - LLocal * t * t / 2);
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
		setialItetaions++;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&setialExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	float setialIterationPerSec = setialItetaions / setialExecutionTime * 1000;
	printf("Serial execution time:%f ms = %f it/sec \n", setialExecutionTime,
			setialIterationPerSec);
	fprintf(fp, "Serial execution time:%f ms = %f it/sec \n",
			setialExecutionTime, setialIterationPerSec);

	printf("serial greedy iterations:%d\n", setialItetaions);
	fprintf(fp, "serial greedy iterations:%d\n", setialItetaions);
	saveSolutionIntoFile("/tmp/ttd_g_serial.txt", x, n, nodeDescription, 0);
	//-------------------------------Computation
	float* T_dev;
	float* E_dev;
	cublasAlloc(n, sizeof(float), (void**) &T_dev);
	checkCUDAError("Alokacia T_dev");
	cublasAlloc(n, sizeof(float), (void**) &E_dev);
	checkCUDAError("Alokacia E_dev");

	float time;
	cusparseStatus_t copystatus;
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
	long long parallel_iterations = 0;
	t_start = clock();
	while (getTotalElapsetTime(&t_start) < executionTime) {
		for (int asdf = 0; asdf < 100; asdf++) {
			maxIndex = cublasIsamax(n, E_dev, 1);
			maxIndex = maxIndex - 1;
			//									printf("%d;Selected index:%d\n", i, maxIndex);
			IncreaseElementAndShrink<<< 1 ,1 >>>( maxIndex,T_dev, E_dev, derivatives_d, L_d,x_d,
					lambda, n );

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
			}
			parallel_iterations++;
		}
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("pridane  %f   \n", tPoint[0]);
	printf("trvanie %f ms ,  %f iter/sec speedUp:%f \n", time,
			parallel_iterations / time * 1000, parallel_iterations / time
					* 1000 / setialIterationPerSec);
	printf("parallel greedy iterations:%d\n", parallel_iterations);

	fprintf(fp, "trvanie %f ms ,  %f iter/sec speedUp:%f \n", time,
			parallel_iterations / time * 1000, parallel_iterations / time
					* 1000 / setialIterationPerSec);
	fprintf(fp, "parallel greedy iterations:%d\n", parallel_iterations);

	float treshHold = 0;
	cudaMemcpy(x, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("kernel execution compy ....");
	saveSolutionIntoFile("/tmp/ttd_g_parallel.txt", x, n, nodeDescription,
			treshHold);

	print_time_message(&t1, "Computation");

	//-------------------------------DeAllocation
	cudaFree(ATcount_d);
	print_time_message(&t1, "Device memory DeAllocated");

	float residuals[m];
	float b[m];
	float* redisuals_dev;
	for (int i = 0; i < m; i++) {
		residuals[i] = g[i];
		b[i] = -g[i];
	}

	for (int i = 0; i < n; i++)
		x[i] = 0;

	cudaMalloc((void**) &redisuals_dev, m * sizeof(float));
	checkCUDAError("kernel execution Alloc residuals_d");
	cudaMemcpy(redisuals_dev, &residuals[0], m * sizeof(float),
			cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy residuals_d");
	cudaMemcpy(x_d, &x[0], n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy x");

	setialItetaions = 0;
	t_start = clock();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	while (getTotalElapsetTime(&t_start) < SRCDMExecutionTime) {
		float tmp = (float) rand() / RAND_MAX;
		int idx = (int) (n * tmp);
		tmp = 0;
		for (int j = idx * 4; j < idx * 4 + 4; j++) {
			//printf("A=%f, res=%f,colid=%d\n",A_TTD[j],residuals[Col_IDX[j]],Col_IDX[j]);
			tmp += A_TTD[j] * residuals[Row_IDX[j] - 1];
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
		//		printf("ID:%d, value%f\n",idx, x[idx]);
		for (int j = idx * 4; j < idx * 4 + 4; j++) {
			residuals[Row_IDX[j] - 1] += tmp * A_TTD[j];
		}
		setialItetaions++;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&setialExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	float serialItPsec = (float) setialItetaions / setialExecutionTime;
	printf("trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			setialItetaions / setialExecutionTime * 1000);
	printf("serial random iterations:%d\n", setialItetaions);
	fprintf(fp, "trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			setialItetaions / setialExecutionTime * 1000);
	fprintf(fp, "serial random iterations:%d\n", setialItetaions);

	saveSolutionIntoFile("/tmp/ttd_r_serial.txt", x, n, nodeDescription, 0);
	//--------------------------------------------------
	// ============  Parallel Random


	dim3 dimBlockRCD2( TOTALTHREDSPERBLOCK);
	dim3 dimGridRCDM2( dim1, dim2);

	curandState *devStates;
	cudaMalloc((void **) &devStates, dim1 * dim2 * TOTALTHREDSPERBLOCK
			* sizeof(curandState));
	setup_kernel<<< dimGridRCDM2, dimBlockRCD2 >>>(devStates);
	checkCUDAError("Inicializacia ranom states");
	cudaFree(L_d);
	checkCUDAError("Dealocation");
	float* Li_dev;
	cudaMalloc((void**) &Li_dev, n * sizeof(float));
	checkCUDAError("kernel execution Alloc Li_d");
	cudaMemcpy(Li_dev, &Li[0], n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Coppy Li_d");

	cudaMemcpy(A_d, A_TTD, nnzElements * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy A_d");

	cudaFree(C_idx_d);
	checkCUDAError("Dealocation");

	int* R_idx_d;
	cudaMalloc((void**) &R_idx_d, nnzElements * sizeof(int));
	checkCUDAError("kernel execution Alloc R_idx_d");

	cudaMemcpy(R_idx_d, Row_IDX, nnzElements * sizeof(int),
			cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy Ridx_d");

	float l1sum = cublasSasum(n, x_d, 1);
	float gNorm = cublasSnrm2(m, redisuals_dev, 1);
	printf("k=%d,L1 sum part:%1.16f, \|Ax-b\|^2=%1.16f, fval=%1.16f", 0, l1sum,
			gNorm * gNorm, lambda * l1sum + 0.5 * gNorm * gNorm);
	fprintf(fp, "k=%d,L1 sum part:%1.16f, \|Ax-b\|^2=%1.16f, fval=%1.16f", 0,
			l1sum, gNorm * gNorm, lambda * l1sum + 0.5 * gNorm * gNorm);

	print_time_message(&t1, "Idem spustat paralelny RCDM");
	parallel_iterations = 0;
	float parallelExecutionTime;

	for (int k = 0; k < 24; k++) {
		t_start = clock();
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		while (getTotalElapsetTime(&t_start) < PRCDMExecutionTime) {

			RCDMKernel<<< dimGridRCDM2, dimBlockRCD2 >>>(A_d, R_idx_d, n_d, redisuals_dev, x_d, lambda_d, Li_dev,
					devStates);

			parallel_iterations += dim1 * dim2;

		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&parallelExecutionTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("k= %d\n", k);
		printf("parallel random iterations:%d totalMultiplier %d\n",
				parallel_iterations, NMAXITERKernel * TOTALTHREDSPERBLOCK);
		printf("trvanie %f ms ,  %f iter/sec , speedup:%f \n",
				parallelExecutionTime, parallel_iterations
						/ parallelExecutionTime * 1000 * NMAXITERKernel
						* TOTALTHREDSPERBLOCK, TOTALTHREDSPERBLOCK
						* NMAXITERKernel * parallel_iterations
						/ parallelExecutionTime / serialItPsec);
		fprintf(fp, "k= %d\n", k);
		fprintf(fp, "parallel random iterations:%d toitalMultiplies %d\n",
				parallel_iterations, NMAXITERKernel * TOTALTHREDSPERBLOCK);
		fprintf(fp, "trvanie %f ms ,  %f iter/sec , speedup:%f \n",
				parallelExecutionTime, parallel_iterations
						/ parallelExecutionTime * 1000 * NMAXITERKernel
						* TOTALTHREDSPERBLOCK, TOTALTHREDSPERBLOCK
						* NMAXITERKernel * parallel_iterations
						/ parallelExecutionTime / serialItPsec);
		print_time_message(&t1, "END spustat paralelny RCDM");
		cudaMemcpy(x, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("kernel execution compy ....");
		string fileName = "";
		fileName = "/tmp/ttd_r_parallel_";
		std::ostringstream oss;

		oss << k;
		fileName += oss.str();

		l1sum = cublasSasum(n, x_d, 1);
		gNorm = cublasSnrm2(m, redisuals_dev, 1);
		printf("k=%d,L1 sum part:%f, \|Ax-b\|^2=%f, fval=%f", k, l1sum, gNorm
				* gNorm, lambda * l1sum + 0.5 * gNorm * gNorm);
		fprintf(fp, "k=%d,L1 sum part:%f, \|Ax-b\|^2=%f, fval=%f", k, l1sum,
				gNorm * gNorm, lambda * l1sum + 0.5 * gNorm * gNorm);

		saveSolutionIntoFile(fileName.c_str(), x, n, nodeDescription, treshHold);
	}

	//------------------------------------------------------
	//===================== Nesterov Composite Accelerated
	float A_nest = 0;
	float L_nest = L[0];
	float mu = 0;
	float v[n];
	float y[n];
	float derv[n];
	float derv2[n];
	float dervHistory[n];
	for (int i = 0; i < n; i++) {
		x[i] = 0;
		v[i] = 0;
		derv[i] = 0;
		dervHistory[i] = 0;
	}
	float gamma_u = 2;
	float gamma_d = gamma_u;
	float quadSolTmp = 0;

	parallel_iterations = 0;
	t_start = clock();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	while (getTotalElapsetTime(&t_start) < executionTime) {
		int doRepeat = 1;
		while (doRepeat) {
			quadSolTmp = 2 * (1 + A_nest * mu) / L_nest;
			float a = (quadSolTmp + sqrt(quadSolTmp * quadSolTmp + 4 * A_nest
					* quadSolTmp)) / 2;
			float convCombA = A_nest / (a + A_nest);
			float convComba = a / (a + A_nest);
			for (int i = 0; i < n; i++) {
				y[i] = convCombA * x[i] + convComba * v[i];
			}
			for (int i = 0; i < n; i++) {
				derv[i] = 0;
			}
			for (int i = 0; i < m; i++) {
				g[i] = -b[i];
			}
			for (int i = 0; i < nnzElements; i++) // derv = Ax-b
			{
				g[Row_IDX[i] - 1] += A_TTD[i] * y[Col_IDX[i] - 1];
			}
			for (int i = 0; i < nnzElements; i++) {
				derv[Col_IDX[i] - 1] += g[Row_IDX[i] - 1] * A_TTD[i];
			}
			float LLocal = L_nest;
			float Li = 1 / LLocal;
			float lambdaOverL = lambda * Li;
			for (int j = 0; j < n; j++) {
				float xLocal = x[j];
				float alpha = derv[j] * Li;
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
			}
			for (int j = 0; j < n; j++) {
				derv2[j] = -L_nest * TVALUES[j] - derv[j];
				derv[j] = 0;
			}
			for (int i = 0; i < m; i++) {
				g[i] = -b[i];
			}
			for (int i = 0; i < nnzElements; i++) // derv = Ax-b
			{
				g[Row_IDX[i] - 1] += A_TTD[i] * (y[Col_IDX[i] - 1]
						+ TVALUES[Col_IDX[i] - 1]);
			}
			for (int i = 0; i < nnzElements; i++) {
				derv[Col_IDX[i] - 1] += g[Row_IDX[i] - 1] * A_TTD[i];
			}
			for (int i = 0; i < n; i++) {
				derv[i] += derv2[i];
			}
			float LS = 0;
			float RS = 0;
			for (int i = 0; i < n; i++) {
				LS -= derv[i] * TVALUES[i];
				RS += derv[i] * derv[i];
			}
			RS = RS / L_nest;
			if (LS < RS) {
				L_nest = L_nest * gamma_u;
				doRepeat = 1;
			} else {
				doRepeat = 0;
				A_nest += a;
				for (int i = 0; i < n; i++) {
					x[i] = y[i] + TVALUES[i];
				}
				L_nest = L_nest / gamma_d;
				float ratioA = (A_nest - a) / A_nest;
				float ratioB = (a) / A_nest;
				for (int i = 0; i < n; i++) {
					dervHistory[i] = ratioA * dervHistory[i] + ratioB
							* (derv[i] - derv2[i]);
				}

				float LLocal = 1 / A_nest;
				float Li = A_nest;
				float lambdaOverL = lambda * Li;
				for (int i = 0; i < n; i++) {
					float alpha = dervHistory[i] * Li;
					float deltaL = -alpha - lambdaOverL;
					float deltaR = -alpha + lambdaOverL;
					float t;
					if (deltaL > 0)
						t = deltaL;
					else if (deltaR < 0)
						t = deltaR;
					else
						t = 0;
					v[i] = t;

				}

			}
		}
		parallel_iterations++;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&setialExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("serial AC iterations:%d\n", parallel_iterations);

	printf("trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			parallel_iterations / setialExecutionTime * 1000);
	fprintf(fp, "serial AC iterations:%d\n", parallel_iterations);

	fprintf(fp, "trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			parallel_iterations / setialExecutionTime * 1000);
	saveSolutionIntoFile("/tmp/ttd_ac.txt", x, n, nodeDescription, treshHold);
	//-------------------------------------------------------

	cudaFree(Li_dev);
	checkCUDAError("Dealocationa");
	cudaFree(R_idx_d);
	checkCUDAError("Dealocationb");
	cudaFree(redisuals_dev);
	checkCUDAError("Dealocationc");
	cudaFree(lambda_d);
	checkCUDAError("Dealocationd");
	cudaFree(A_d);
	checkCUDAError("Dealocatione");
	cudaFree(n_d);
	checkCUDAError("Dealocationg");
	cudaFree(x_d);
	checkCUDAError("Dealocationh");
	cudaFree(derivatives_d);
	checkCUDAError("Dealocationi");

	print_time_message(&t1, "Device memory DeAllocated");
	fclose(fp);
}

void oneDayPG(int row, int col, int exampleType, float executionTime,
		float lambda) {

	int dim1 = 14;
	int dim2 = 1;
	float PRCDMExecutionTime = 1;
	float PGExecutionTime = 600;
	float SRCDMExecutionTime = 1;

	int m, n;
	clock_t t1;
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
	//-----------------------------------------
	FILE *fp;
	//		fp = fopen("/tmp/asdf", "w");
	//
	//		for (int i = 0; i < nnzElements; i++) {
	//			fprintf(fp,"%d,%d,%f\n", Row_IDX[i], Col_IDX[i], A_TTD[i]);
	//		}
	//		fclose(fp);
	//		fp = fopen("/tmp/ttd_vectorB.csv", "w");
	//
	//		for (int i = 0; i < m; i++) {
	//			fprintf(fp,"%d,%f\n", i,-g[i]);
	//		}
	//		fclose(fp);
	//	print_time_message(&t1, "Data saved");
	//-----------------------------------------
	float x[n];
	float L[n];
	float Li[n];
	for (int i = 0; i < n; i++) {
		x[i] = 0;
		L[i] = 0;
		for (int k = 4 * i; k < 4 * i + 4; k++) {
			L[i] += A_TTD[k] * A_TTD[k];
		}
		Li[i] = 1 / L[i];
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
	fp = fopen("/tmp/ttd_log.txt", "w");
	//-------------------------------Inicializacia CuSpare Library
	/* allocate GPU memory and copy the matrix and vectors into it */
	cusparseStatus_t status;
	cusparseHandle_t handle = 0;
	/* initialize cusparse library */
	status = cusparseCreate(&handle);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("CUSPARSE Library initialization failed");
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
		float mlt2 = 1;
		if (deltaL > 0)
			t = deltaL - xLocal;
		else if (deltaR < 0)
			t = deltaR - xLocal;
		else {
			t = -xLocal;
			mlt2 = 0;
		}
		float tmpEnergy = t * t * LLocal;
		TVALUES[j] = t;
		//		EVALUES[j] = tmpEnergy / 2;//+ lambda*x[j]*( signbit(x[j]) -signbit(x[j]+TVALUES[j])    );

		EVALUES[j] = mlt2 * (tmpEnergy / 2 + lambda * xLocal * (signbit(xLocal)
				- signbit(xLocal + t))) + (1 - mlt2) * (lambda * xLocal
				* signbit(xLocal) - derivatives[j] * t - LLocal * t * t / 2);

		if (tmpEnergy > max_VAL) {
			optimalStep = t;
			max_VAL = tmpEnergy;
			max_IDX = j;
		}
	}
	print_time_message(&t1, "Initial Shrink End");
	print_time_message(&t1, "Start serial Code");
	float setialExecutionTime = 0;
	long long setialItetaions = 0;

	clock_t t_start;
	t_start = clock();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	while (getTotalElapsetTime(&t_start) < executionTime) {
		// 			printf("optimal t_idx=%d, tval=%f, elapsedTime:%f\n", max_IDX, TVALUES[max_IDX],getTotalElapsetTime(&t_start));
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
				float mlt2 = 1;
				if (deltaL > 0)
					t = deltaL - xLocal;
				else if (deltaR < 0)
					t = deltaR - xLocal;
				else {
					t = -xLocal;
					mlt2 = 0;
				}
				float tmpEnergy = t * t * LLocal;
				TVALUES[j] = t;
				//				EVALUES[j] = tmpEnergy / 2;//+ lambda*x[j]*( signbit(x[j]) -signbit(x[j]+TVALUES[j])    );

				EVALUES[j] = mlt2 * (tmpEnergy / 2 + lambda * xLocal
						* (signbit(xLocal) - signbit(xLocal + t))) + (1 - mlt2)
						* (lambda * xLocal * signbit(xLocal) - derivatives[j]
								* t - LLocal * t * t / 2);
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
		setialItetaions++;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&setialExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	float setialIterationPerSec = setialItetaions / setialExecutionTime * 1000;
	printf("Serial execution time:%f ms = %f it/sec \n", setialExecutionTime,
			setialIterationPerSec);
	fprintf(fp, "Serial execution time:%f ms = %f it/sec \n",
			setialExecutionTime, setialIterationPerSec);

	printf("serial greedy iterations:%d\n", setialItetaions);
	fprintf(fp, "serial greedy iterations:%d\n", setialItetaions);
	saveSolutionIntoFile("/tmp/ttd_g_serial.txt", x, n, nodeDescription, 0);
	//-------------------------------Computation
	float* T_dev;
	float* E_dev;
	cublasAlloc(n, sizeof(float), (void**) &T_dev);
	checkCUDAError("Alokacia T_dev");
	cublasAlloc(n, sizeof(float), (void**) &E_dev);
	checkCUDAError("Alokacia E_dev");

	float time;
	cusparseStatus_t copystatus;
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

	long long parallel_iterations = 0;

	float treshHold = 0;
	for (int k = 0; k < 114; k++) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		t_start = clock();
		while (getTotalElapsetTime(&t_start) < PGExecutionTime) {
			for (int asdf = 0; asdf < 100; asdf++) {
				maxIndex = cublasIsamax(n, E_dev, 1);
				maxIndex = maxIndex - 1;
				//									printf("%d;Selected index:%d\n", i, maxIndex);
				IncreaseElementAndShrink<<< 1 ,1 >>>( maxIndex,T_dev, E_dev, derivatives_d, L_d,x_d,
						lambda, n );
				//				IncreaseElement<<< 1 ,1 >>>(x_d, maxIndex, T_dev);

				checkCUDAError("IncreaseElement");
				cublasGetVector(1, sizeof(float), &T_dev[maxIndex], 1, tPoint,
						1);
				checkCUDAError("get T");
				//			printf("optimal t_idx=%d and value = %f\n", maxIndex, tPoint[0]);
				for (int ki = maxIndex * 4; ki < maxIndex * 4 + 4; ki++) {
					//								printf("pridat %f   %f   %d\n", tPoint[0], A_TTD[ki],Row_IDX[ki]-1);
					copystatus = cusparseSaxpyi(handle, rowCounts[Row_IDX[ki]
							- 1], tPoint[0] * A_TTD[ki],
							&A_d[ATcount[Row_IDX[ki] - 1]],
							&C_idx_d[ATcount[Row_IDX[ki] - 1]], derivatives_d,
							CUSPARSE_INDEX_BASE_ZERO);
					if (copystatus != CUSPARSE_STATUS_SUCCESS)
						printf("Nastala chyba pri CuSparse!\n");

					//				cudaThreadSynchronize();
					ShrinkKernelSubset<<< dimGridShrinkSubset ,dimBlock >>>(T_dev, E_dev, derivatives_d,
							L_d, x_d, lambda, &ATcount_d[Row_IDX[ki] - 1], &C_idx_d[ATcount[Row_IDX[ki] - 1]]);
					//
					checkCUDAError("Shrink subset");
				}
				//				ShrinkKernel<<< dimGridRCDM ,dimBlock >>>(T_dev, E_dev, derivatives_d, L_d,x_d,
				//							lambda, n);
				parallel_iterations++;
			}
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("pridane  %f   \n", tPoint[0]);
		printf("trvanie %f ms ,  %f iter/sec speedUp:%f \n", time,
				parallel_iterations / time * 1000, parallel_iterations / time
						* 1000 / setialIterationPerSec);
		printf("parallel greedy iterations:%d\n", parallel_iterations);

		fprintf(fp, "trvanie %f ms ,  %f iter/sec speedUp:%f \n", time,
				parallel_iterations / time * 1000, parallel_iterations / time
						* 1000 / setialIterationPerSec);
		fprintf(fp, "parallel greedy iterations:%d\n", parallel_iterations);

		cudaMemcpy(x, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("kernel execution compy ....");

		string fileName = "";
		fileName = "/tmp/ttd_g_parallel_";
		std::ostringstream oss;

		oss << k;
		fileName += oss.str();

		//		printf("k=%d,L1 sum part:%f, \|Ax-b\|^2=%f, fval=%f", k, l1sum, gNorm
		//				* gNorm, lambda * l1sum + 0.5 * gNorm * gNorm);
		//		fprintf(fp, "k=%d,L1 sum part:%f, \|Ax-b\|^2=%f, fval=%f", k, l1sum,
		//				gNorm * gNorm, lambda * l1sum + 0.5 * gNorm * gNorm);

		saveSolutionIntoFile(fileName.c_str(), x, n, nodeDescription, treshHold);
	}

	print_time_message(&t1, "Computation");

	//-------------------------------DeAllocation
	cudaFree(ATcount_d);
	print_time_message(&t1, "Device memory DeAllocated");

	float residuals[m];
	float b[m];
	float* redisuals_dev;
	for (int i = 0; i < m; i++) {
		residuals[i] = g[i];
		b[i] = -g[i];
	}

	for (int i = 0; i < n; i++)
		x[i] = 0;

	cudaMalloc((void**) &redisuals_dev, m * sizeof(float));
	checkCUDAError("kernel execution Alloc residuals_d");
	cudaMemcpy(redisuals_dev, &residuals[0], m * sizeof(float),
			cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy residuals_d");
	cudaMemcpy(x_d, &x[0], n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy x");

	setialItetaions = 0;
	t_start = clock();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	while (getTotalElapsetTime(&t_start) < SRCDMExecutionTime) {
		float tmp = (float) rand() / RAND_MAX;
		int idx = (int) (n * tmp);
		tmp = 0;
		for (int j = idx * 4; j < idx * 4 + 4; j++) {
			//printf("A=%f, res=%f,colid=%d\n",A_TTD[j],residuals[Col_IDX[j]],Col_IDX[j]);
			tmp += A_TTD[j] * residuals[Row_IDX[j] - 1];
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
		//		printf("ID:%d, value%f\n",idx, x[idx]);
		for (int j = idx * 4; j < idx * 4 + 4; j++) {
			residuals[Row_IDX[j] - 1] += tmp * A_TTD[j];
		}
		setialItetaions++;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&setialExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	float serialItPsec = (float) setialItetaions / setialExecutionTime;
	printf("trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			setialItetaions / setialExecutionTime * 1000);
	printf("serial random iterations:%d\n", setialItetaions);
	fprintf(fp, "trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			setialItetaions / setialExecutionTime * 1000);
	fprintf(fp, "serial random iterations:%d\n", setialItetaions);

	saveSolutionIntoFile("/tmp/ttd_r_serial.txt", x, n, nodeDescription, 0);
	//--------------------------------------------------
	// ============  Parallel Random


	dim3 dimBlockRCD2( TOTALTHREDSPERBLOCK);
	dim3 dimGridRCDM2( dim1, dim2);

	curandState *devStates;
	cudaMalloc((void **) &devStates, dim1 * dim2 * TOTALTHREDSPERBLOCK
			* sizeof(curandState));
	setup_kernel<<< dimGridRCDM2, dimBlockRCD2 >>>(devStates);
	checkCUDAError("Inicializacia ranom states");
	cudaFree(L_d);
	checkCUDAError("Dealocation");
	float* Li_dev;
	cudaMalloc((void**) &Li_dev, n * sizeof(float));
	checkCUDAError("kernel execution Alloc Li_d");
	cudaMemcpy(Li_dev, &Li[0], n * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Coppy Li_d");

	cudaMemcpy(A_d, A_TTD, nnzElements * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy A_d");

	cudaFree(C_idx_d);
	checkCUDAError("Dealocation");

	int* R_idx_d;
	cudaMalloc((void**) &R_idx_d, nnzElements * sizeof(int));
	checkCUDAError("kernel execution Alloc R_idx_d");

	cudaMemcpy(R_idx_d, Row_IDX, nnzElements * sizeof(int),
			cudaMemcpyHostToDevice);
	checkCUDAError("kernel execution Copy Ridx_d");

	float l1sum = cublasSasum(n, x_d, 1);
	float gNorm = cublasSnrm2(m, redisuals_dev, 1);
	printf("k=%d,L1 sum part:%1.16f, \|Ax-b\|^2=%1.16f, fval=%1.16f", 0, l1sum,
			gNorm * gNorm, lambda * l1sum + 0.5 * gNorm * gNorm);
	fprintf(fp, "k=%d,L1 sum part:%1.16f, \|Ax-b\|^2=%1.16f, fval=%1.16f", 0,
			l1sum, gNorm * gNorm, lambda * l1sum + 0.5 * gNorm * gNorm);

	print_time_message(&t1, "Idem spustat paralelny RCDM");
	parallel_iterations = 0;
	float parallelExecutionTime;

	for (int k = 0; k < 0; k++) {
		t_start = clock();
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		while (getTotalElapsetTime(&t_start) < PRCDMExecutionTime) {

			RCDMKernel<<< dimGridRCDM2, dimBlockRCD2 >>>(A_d, R_idx_d, n_d, redisuals_dev, x_d, lambda_d, Li_dev,
					devStates);

			parallel_iterations += dim1 * dim2;

		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&parallelExecutionTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("k= %d\n", k);
		printf("parallel random iterations:%d totalMultiplier %d\n",
				parallel_iterations, NMAXITERKernel * TOTALTHREDSPERBLOCK);
		printf("trvanie %f ms ,  %f iter/sec , speedup:%f \n",
				parallelExecutionTime, parallel_iterations
						/ parallelExecutionTime * 1000 * NMAXITERKernel
						* TOTALTHREDSPERBLOCK, TOTALTHREDSPERBLOCK
						* NMAXITERKernel * parallel_iterations
						/ parallelExecutionTime / serialItPsec);
		fprintf(fp, "k= %d\n", k);
		fprintf(fp, "parallel random iterations:%d toitalMultiplies %d\n",
				parallel_iterations, NMAXITERKernel * TOTALTHREDSPERBLOCK);
		fprintf(fp, "trvanie %f ms ,  %f iter/sec , speedup:%f \n",
				parallelExecutionTime, parallel_iterations
						/ parallelExecutionTime * 1000 * NMAXITERKernel
						* TOTALTHREDSPERBLOCK, TOTALTHREDSPERBLOCK
						* NMAXITERKernel * parallel_iterations
						/ parallelExecutionTime / serialItPsec);
		print_time_message(&t1, "END spustat paralelny RCDM");
		cudaMemcpy(x, x_d, n * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("kernel execution compy ....");
		string fileName = "";
		fileName = "/tmp/ttd_r_parallel_";
		std::ostringstream oss;

		oss << k;
		fileName += oss.str();

		l1sum = cublasSasum(n, x_d, 1);
		gNorm = cublasSnrm2(m, redisuals_dev, 1);
		printf("k=%d,L1 sum part:%f, \|Ax-b\|^2=%f, fval=%f", k, l1sum, gNorm
				* gNorm, lambda * l1sum + 0.5 * gNorm * gNorm);
		fprintf(fp, "k=%d,L1 sum part:%f, \|Ax-b\|^2=%f, fval=%f", k, l1sum,
				gNorm * gNorm, lambda * l1sum + 0.5 * gNorm * gNorm);

		saveSolutionIntoFile(fileName.c_str(), x, n, nodeDescription, treshHold);
	}

	//------------------------------------------------------
	//===================== Nesterov Composite Accelerated
	float A_nest = 0;
	float L_nest = L[0];
	float mu = 0;
	float v[n];
	float y[n];
	float derv[n];
	float derv2[n];
	float dervHistory[n];
	for (int i = 0; i < n; i++) {
		x[i] = 0;
		v[i] = 0;
		derv[i] = 0;
		dervHistory[i] = 0;
	}
	float gamma_u = 2;
	float gamma_d = gamma_u;
	float quadSolTmp = 0;

	parallel_iterations = 0;
	t_start = clock();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	while (getTotalElapsetTime(&t_start) < executionTime) {
		int doRepeat = 1;
		while (doRepeat) {
			quadSolTmp = 2 * (1 + A_nest * mu) / L_nest;
			float a = (quadSolTmp + sqrt(quadSolTmp * quadSolTmp + 4 * A_nest
					* quadSolTmp)) / 2;
			float convCombA = A_nest / (a + A_nest);
			float convComba = a / (a + A_nest);
			for (int i = 0; i < n; i++) {
				y[i] = convCombA * x[i] + convComba * v[i];
			}
			for (int i = 0; i < n; i++) {
				derv[i] = 0;
			}
			for (int i = 0; i < m; i++) {
				g[i] = -b[i];
			}
			for (int i = 0; i < nnzElements; i++) // derv = Ax-b
			{
				g[Row_IDX[i] - 1] += A_TTD[i] * y[Col_IDX[i] - 1];
			}
			for (int i = 0; i < nnzElements; i++) {
				derv[Col_IDX[i] - 1] += g[Row_IDX[i] - 1] * A_TTD[i];
			}
			float LLocal = L_nest;
			float Li = 1 / LLocal;
			float lambdaOverL = lambda * Li;
			for (int j = 0; j < n; j++) {
				float xLocal = x[j];
				float alpha = derv[j] * Li;
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
			}
			for (int j = 0; j < n; j++) {
				derv2[j] = -L_nest * TVALUES[j] - derv[j];
				derv[j] = 0;
			}
			for (int i = 0; i < m; i++) {
				g[i] = -b[i];
			}
			for (int i = 0; i < nnzElements; i++) // derv = Ax-b
			{
				g[Row_IDX[i] - 1] += A_TTD[i] * (y[Col_IDX[i] - 1]
						+ TVALUES[Col_IDX[i] - 1]);
			}
			for (int i = 0; i < nnzElements; i++) {
				derv[Col_IDX[i] - 1] += g[Row_IDX[i] - 1] * A_TTD[i];
			}
			for (int i = 0; i < n; i++) {
				derv[i] += derv2[i];
			}
			float LS = 0;
			float RS = 0;
			for (int i = 0; i < n; i++) {
				LS -= derv[i] * TVALUES[i];
				RS += derv[i] * derv[i];
			}
			RS = RS / L_nest;
			if (LS < RS) {
				L_nest = L_nest * gamma_u;
				doRepeat = 1;
			} else {
				doRepeat = 0;
				A_nest += a;
				for (int i = 0; i < n; i++) {
					x[i] = y[i] + TVALUES[i];
				}
				L_nest = L_nest / gamma_d;
				float ratioA = (A_nest - a) / A_nest;
				float ratioB = (a) / A_nest;
				for (int i = 0; i < n; i++) {
					dervHistory[i] = ratioA * dervHistory[i] + ratioB
							* (derv[i] - derv2[i]);
				}

				float LLocal = 1 / A_nest;
				float Li = A_nest;
				float lambdaOverL = lambda * Li;
				for (int i = 0; i < n; i++) {
					float alpha = dervHistory[i] * Li;
					float deltaL = -alpha - lambdaOverL;
					float deltaR = -alpha + lambdaOverL;
					float t;
					if (deltaL > 0)
						t = deltaL;
					else if (deltaR < 0)
						t = deltaR;
					else
						t = 0;
					v[i] = t;

				}

			}
		}
		parallel_iterations++;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&setialExecutionTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("serial AC iterations:%d\n", parallel_iterations);

	printf("trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			parallel_iterations / setialExecutionTime * 1000);
	fprintf(fp, "serial AC iterations:%d\n", parallel_iterations);

	fprintf(fp, "trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			parallel_iterations / setialExecutionTime * 1000);
	saveSolutionIntoFile("/tmp/ttd_ac.txt", x, n, nodeDescription, treshHold);
	//-------------------------------------------------------

	cudaFree(Li_dev);
	checkCUDAError("Dealocationa");
	cudaFree(R_idx_d);
	checkCUDAError("Dealocationb");
	cudaFree(redisuals_dev);
	checkCUDAError("Dealocationc");
	cudaFree(lambda_d);
	checkCUDAError("Dealocationd");
	cudaFree(A_d);
	checkCUDAError("Dealocatione");
	cudaFree(n_d);
	checkCUDAError("Dealocationg");
	cudaFree(x_d);
	checkCUDAError("Dealocationh");
	cudaFree(derivatives_d);
	checkCUDAError("Dealocationi");

	print_time_message(&t1, "Device memory DeAllocated");
	fclose(fp);
}

void niceAC(int row, int col, int exampleType, float executionTime,
		float lambda) {

	int dim1 = 14;
	int dim2 = 1;
	float PRCDMExecutionTime = 0;
	float SRCDMExecutionTime = PRCDMExecutionTime;
	float setialExecutionTime = 0;
	int m, n;
	clock_t t1;
	float * A_TTD;
	int * Row_IDX;
	int * Col_IDX;
	int nnzElements;
	int* nodeDescription;
	clock_t t_start;
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
	float b[m];
	getForceVector(&g, m, row, col, exampleType);
	for (int i = 0; i < m; i++) {
		b[i] = g[i];
		g[i] = -g[i];

	}
	print_time_message(&t1, "Force vector obtained");
	//-----------------------------------------
	FILE *fp;
	//		fp = fopen("/tmp/asdf", "w");
	//
	//		for (int i = 0; i < nnzElements; i++) {
	//			fprintf(fp,"%d,%d,%f\n", Row_IDX[i], Col_IDX[i], A_TTD[i]);
	//		}
	//		fclose(fp);
	//		fp = fopen("/tmp/ttd_vectorB.csv", "w");
	//
	//		for (int i = 0; i < m; i++) {
	//			fprintf(fp,"%d,%f\n", i,-g[i]);
	//		}
	//		fclose(fp);
	//	print_time_message(&t1, "Data saved");
	//-----------------------------------------
	float x[n];
	float L[n];
	float Li[n];
	for (int i = 0; i < n; i++) {
		x[i] = 0;
		L[i] = 0;
		for (int k = 4 * i; k < 4 * i + 4; k++) {
			L[i] += A_TTD[k] * A_TTD[k];
		}
		Li[i] = 1 / L[i];
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
	fp = fopen("/tmp/ttd_log.txt", "w");
	//-------------------------------Inicializacia CuSpare Library
	/* allocate GPU memory and copy the matrix and vectors into it */

	float residuals[m];
	float* redisuals_dev;
	for (int i = 0; i < m; i++) {
		residuals[i] = g[i];
	}
	for (int i = 0; i < n; i++)
		x[i] = 0;
	long long setialItetaions = 0;

	t_start = clock();

	while (getTotalElapsetTime(&t_start) < SRCDMExecutionTime) {
		float tmp = (float) rand() / RAND_MAX;
		int idx = (int) (n * tmp);
		tmp = 0;
		for (int j = idx * 4; j < idx * 4 + 4; j++) {
			//printf("A=%f, res=%f,colid=%d\n",A_TTD[j],residuals[Col_IDX[j]],Col_IDX[j]);
			tmp += A_TTD[j] * residuals[Row_IDX[j] - 1];
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
		//		printf("ID:%d, value%f\n",idx, x[idx]);
		for (int j = idx * 4; j < idx * 4 + 4; j++) {
			residuals[Row_IDX[j] - 1] += tmp * A_TTD[j];
		}
		setialItetaions++;
	}
	float serialItPsec = (float) setialItetaions / setialExecutionTime;
	printf("trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			setialItetaions / setialExecutionTime * 1000);
	printf("serial random iterations:%d\n", setialItetaions);
	fprintf(fp, "trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			setialItetaions / setialExecutionTime * 1000);
	fprintf(fp, "serial random iterations:%d\n", setialItetaions);

	saveSolutionIntoFile("/tmp/ttd_r_serial.txt", x, n, nodeDescription, 0);

	//===================== Nesterov Composite Accelerated
	float A_nest = 0;
	float L_nest = L[0];
	float mu = 0;
	float v[n];
	float y[n];
	float derv[n];
	float derv2[n];
	float dervHistory[n];
	for (int i = 0; i < n; i++) {
		x[i] = 0;
		v[i] = 0;
		derv[i] = 0;
		dervHistory[i] = 0;
	}
	float gamma_u = 2;
	float gamma_d = gamma_u;
	float quadSolTmp = 0;
	float TVALUES[n];
	float treshHold = 0;
	int parallel_iterations = 0;
	t_start = clock();
	while (getTotalElapsetTime(&t_start) < executionTime) {
		int doRepeat = 1;
		while (doRepeat) {
			quadSolTmp = 2 * (1 + A_nest * mu) / L_nest;
			float a = (quadSolTmp + sqrt(quadSolTmp * quadSolTmp + 4 * A_nest
					* quadSolTmp)) / 2;
			float convCombA = A_nest / (a + A_nest);
			float convComba = a / (a + A_nest);
			for (int i = 0; i < n; i++) {
				y[i] = convCombA * x[i] + convComba * v[i];
			}
			for (int i = 0; i < n; i++) {
				derv[i] = 0;
			}
			for (int i = 0; i < m; i++) {
				g[i] = -b[i];
			}
			for (int i = 0; i < nnzElements; i++) // derv = Ax-b
			{
				g[Row_IDX[i] - 1] += A_TTD[i] * y[Col_IDX[i] - 1];
			}
			for (int i = 0; i < nnzElements; i++) {
				derv[Col_IDX[i] - 1] += g[Row_IDX[i] - 1] * A_TTD[i];
			}
			float LLocal = L_nest;
			float Li = 1 / LLocal;
			float lambdaOverL = lambda * Li;
			for (int j = 0; j < n; j++) {
				float xLocal = x[j];
				float alpha = derv[j] * Li;
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
			}
			for (int j = 0; j < n; j++) {
				derv2[j] = -L_nest * TVALUES[j] - derv[j];
				derv[j] = 0;
			}
			for (int i = 0; i < m; i++) {
				g[i] = -b[i];
			}
			for (int i = 0; i < nnzElements; i++) // derv = Ax-b
			{
				g[Row_IDX[i] - 1] += A_TTD[i] * (y[Col_IDX[i] - 1]
						+ TVALUES[Col_IDX[i] - 1]);
			}
			for (int i = 0; i < nnzElements; i++) {
				derv[Col_IDX[i] - 1] += g[Row_IDX[i] - 1] * A_TTD[i];
			}
			for (int i = 0; i < n; i++) {
				derv[i] += derv2[i];
			}
			float LS = 0;
			float RS = 0;
			for (int i = 0; i < n; i++) {
				LS -= derv[i] * TVALUES[i];
				RS += derv[i] * derv[i];
			}
			RS = RS / L_nest;
			if (LS < RS) {
				L_nest = L_nest * gamma_u;
				doRepeat = 1;
			} else {
				doRepeat = 0;
				A_nest += a;
				for (int i = 0; i < n; i++) {
					x[i] = y[i] + TVALUES[i];
				}
				L_nest = L_nest / gamma_d;
				float ratioA = (A_nest - a) / A_nest;
				float ratioB = (a) / A_nest;
				for (int i = 0; i < n; i++) {
					dervHistory[i] = ratioA * dervHistory[i] + ratioB
							* (derv[i] - derv2[i]);
				}

				float LLocal = 1 / A_nest;
				float Li = A_nest;
				float lambdaOverL = lambda * Li;
				for (int i = 0; i < n; i++) {
					float alpha = dervHistory[i] * Li;
					float deltaL = -alpha - lambdaOverL;
					float deltaR = -alpha + lambdaOverL;
					float t;
					if (deltaL > 0)
						t = deltaL;
					else if (deltaR < 0)
						t = deltaR;
					else
						t = 0;
					v[i] = t;

				}

			}
		}
		parallel_iterations++;
	}
	printf("serial AC iterations:%d\n", parallel_iterations);
	setialExecutionTime = getElapsetTime(&t_start);
	fprintf(fp, "serial AC iterations:%d\n", parallel_iterations);

	fprintf(fp, "trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			parallel_iterations / setialExecutionTime * 1000);
	saveSolutionIntoFile("/tmp/ttd_ac_big.txt", x, n, nodeDescription,
			treshHold);
	//-------------------------------------------------------

	print_time_message(&t1, "Device memory DeAllocated");
	fclose(fp);
}

void niceRCDM(int row, int col, int exampleType, float executionTime,
		float lambda) {

	int dim1 = 14;
	int dim2 = 1;
	float PRCDMExecutionTime = executionTime;
	float SRCDMExecutionTime = PRCDMExecutionTime;
	float setialExecutionTime = 0;
	int m, n;
	clock_t t1;
	float * A_TTD;
	int * Row_IDX;
	int * Col_IDX;
	int nnzElements;
	int* nodeDescription;
	clock_t t_start;
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
	float b[m];
	getForceVector(&g, m, row, col, exampleType);
	for (int i = 0; i < m; i++) {
		b[i] = g[i];
		g[i] = -g[i];

	}
	print_time_message(&t1, "Force vector obtained");
	//-----------------------------------------
	FILE *fp;
	//		fp = fopen("/tmp/asdf", "w");
	//
	//		for (int i = 0; i < nnzElements; i++) {
	//			fprintf(fp,"%d,%d,%f\n", Row_IDX[i], Col_IDX[i], A_TTD[i]);
	//		}
	//		fclose(fp);
	//		fp = fopen("/tmp/ttd_vectorB.csv", "w");
	//
	//		for (int i = 0; i < m; i++) {
	//			fprintf(fp,"%d,%f\n", i,-g[i]);
	//		}
	//		fclose(fp);
	//	print_time_message(&t1, "Data saved");
	//-----------------------------------------
	float x[n];
	float L[n];
	float Li[n];
	for (int i = 0; i < n; i++) {
		x[i] = 0;
		L[i] = 0;
		for (int k = 4 * i; k < 4 * i + 4; k++) {
			L[i] += A_TTD[k] * A_TTD[k];
		}
		Li[i] = 1 / L[i];
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
	fp = fopen("/tmp/ttd_log.txt", "w");
	//-------------------------------Inicializacia CuSpare Library
	/* allocate GPU memory and copy the matrix and vectors into it */

	float residuals[m];
	float* redisuals_dev;
	for (int i = 0; i < m; i++) {
		residuals[i] = g[i];
	}
	for (int i = 0; i < n; i++)
		x[i] = 0;
	long long setialItetaions = 0;

	t_start = clock();

	while (getTotalElapsetTime(&t_start) < SRCDMExecutionTime) {
		float tmp = (float) rand() / RAND_MAX;
		int idx = (int) (n * tmp);
		tmp = 0;
		for (int j = idx * 4; j < idx * 4 + 4; j++) {
			//printf("A=%f, res=%f,colid=%d\n",A_TTD[j],residuals[Col_IDX[j]],Col_IDX[j]);
			tmp += A_TTD[j] * residuals[Row_IDX[j] - 1];
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
		//		printf("ID:%d, value%f\n",idx, x[idx]);
		for (int j = idx * 4; j < idx * 4 + 4; j++) {
			residuals[Row_IDX[j] - 1] += tmp * A_TTD[j];
		}
		setialItetaions++;
	}
	float serialItPsec = (float) setialItetaions / setialExecutionTime;
	printf("trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			setialItetaions / setialExecutionTime * 1000);
	printf("serial random iterations:%d\n", setialItetaions);
	fprintf(fp, "trvanie %f ms ,  %f iter/sec  \n", setialExecutionTime,
			setialItetaions / setialExecutionTime * 1000);
	fprintf(fp, "serial random iterations:%d\n", setialItetaions);

	saveSolutionIntoFile("/tmp/ttd_r_serial_big.txt", x, n, nodeDescription, 0);

	fclose(fp);
}

int main(void) {

	//	timeComparison(50, 50, 7, 10000, 0.0001;);
//	timeComparison(100, 100, 2, 10000, 0.00001);
	timeComparison(100, 100, 8, 10000, 0.001);
	//	oneDayPG(60, 60, 2, 1, 0.0001);

//		niceRCDM(200, 200, 2, 80000, 0.0001);
	//	niceAC(200, 200, 2, 80000, 0.0001);
	//	niceProblem(200, 200, 2, 100000, 0.0001);
	//	greedyTTD();
	//	calculateTTDProblem();
	return 1;
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

}

__global__ void RCDMKernel(float *A, int*R_Idx, int* n, float*residuals,
		float*x, float * lambda, float* Li, curandState* cstate) {
	int j, i, k;
	float delta, tmp; //partialDetivative
	int id = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	curandState localState = cstate[id];
	float LambdaParameter = lambda[0];
	__shared__ float partialDetivative[TOTALTHREDSPERBLOCK];
	float xLocal;
	float LiLocal;
	float ALocal[4];
	int cidx;
	int RIDX[4];
	for (k = 0; k < NMAXITERKernel; k++) {
		double d = curand_uniform_double(&localState);
		int idx = (int) (d * n[0]);
		// LOAD A, R, residuals
		//		float* residualsAddress[COLUMNLENGTH];
		xLocal = x[idx];
		LiLocal = Li[idx];
		cidx = idx * 4;
		partialDetivative[threadIdx.x] = 0;
		//        #pragma unroll COLUMNLENGTH
		for (i = 0; i < 4; i++) {
			j = cidx + i;
			ALocal[i] = A[j];
			RIDX[i] = R_Idx[j] - 1;
			//			residualsAddress[i] = &residuals[RIDX[i]];
			partialDetivative[threadIdx.x] += ALocal[i] * residuals[RIDX[i]];
		}

		tmp = LiLocal * (partialDetivative[threadIdx.x] + LambdaParameter);
		if (xLocal > tmp) {
			delta = -tmp;
		} else {
			tmp = LiLocal * (partialDetivative[threadIdx.x] - LambdaParameter);
			if (xLocal < tmp) {
				delta = -tmp;
			} else {
				delta = -xLocal;
			}
		}
		atomicAdd(&x[idx], delta);
		//				atomicAdd(&x[idx], 1);
		for (i = 0; i < COLUMNLENGTH; i++) {
			atomicAdd(&residuals[RIDX[i]], ALocal[i] * delta);
			//			atomicAdd(residualsAddress[i], ALocal[i] * delta);
		}

	}
		cstate[id] = localState;
}
