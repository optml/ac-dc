#include <stdio.h>
#include <cutil.h>

/*
This file can be downloaded from supercomputingblog.com.
This is part of a series of tutorials that demonstrate how to use CUDA
The tutorials will also demonstrate the speed of using CUDA
*/

// IMPORTANT NOTE: for this data size, your graphics card should have at least 512 megabytes of memory.
// If your GPU has less memory, then you will need to decrease this data size.

#define MAX_DATA_SIZE		1024*1024*32		// about 32 million elements. 
// The max data size must be an integer multiple of 128*256, because each block will have 256 threads,
// and the block grid width will be 128. These are arbitrary numbers I choose.


void GoldenBrick(float *pA, float *pB, float *pResult, int count)
{
	for (int i=0; i < count; i++)
	{
		//pResult[count] = pA[count] * pB[count];
		//pResult[count] = pA[count] * pB[count] / 12.34567;
		//pResult[count] = sqrt(pA[count] * pB[count] / 12.34567);
		pResult[count] = sqrt(pA[count] * pB[count] / 12.34567) * sin(pA[count]);
	}
}

__global__ void multiplyNumbersGPU(float *pDataA, float *pDataB, float *pResult)
{
	// Because of the simplicity of this tutorial, we are going to assume that
	// every block has 256 threads. Each thread simply multiplies two numbers,
	// and then stores the result.

	// The grid of blocks is 128 blocks long.

	int tid = (blockIdx.y * 128 * 256) + blockIdx.x * 256 + threadIdx.x;	// This gives every thread a unique ID.
	// By no coincidence, we'll be using this thread ID to determine which data elements to multiply.

	//pResult[tid] = pDataA[tid] * pDataB[tid];		// Each thread only multiplies one data element.
	//pResult[tid] = pDataA[tid] * pDataB[tid] / 12.34567;
	//pResult[tid] = sqrt(pDataA[tid] * pDataB[tid] / 12.34567);
	pResult[tid] = sqrt(pDataA[tid] * pDataB[tid] / 12.34567) * sin(pDataA[tid]);

}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
	float *h_dataA, *h_dataB, *h_resultC;
	float *d_dataA, *d_dataB, *d_resultC;
	double gpuTime;
    int i;

    unsigned int hTimer;

    CUT_DEVICE_INIT(argc, argv);
    CUT_SAFE_CALL(cutCreateTimer(&hTimer));

    printf("Initializing data...\n");
	h_dataA     = (float *)malloc(sizeof(float) * MAX_DATA_SIZE);
	h_dataB     = (float *)malloc(sizeof(float) * MAX_DATA_SIZE);
	h_resultC = (float *)malloc(sizeof(float) * MAX_DATA_SIZE);
	CUDA_SAFE_CALL( cudaMalloc( (void **)&d_dataA, sizeof(float) * MAX_DATA_SIZE) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&d_dataB, sizeof(float) * MAX_DATA_SIZE) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&d_resultC , sizeof(float) * MAX_DATA_SIZE) );

	srand(123);
	for(i = 0; i < MAX_DATA_SIZE; i++)
	{
		h_dataA[i] = (float)rand() / (float)RAND_MAX;
		h_dataB[i] = (float)rand() / (float)RAND_MAX;
	}

	int firstRun = 1;	// Indicates if it's the first execution of the for loop
	const int useGPU = 1;	// When 0, only the CPU is used. When 1, only the GPU is used

	for (int dataAmount = MAX_DATA_SIZE; dataAmount > 128*256; dataAmount /= 2)
	{
		int blockGridWidth = 128;
		int blockGridHeight = (dataAmount / 256) / blockGridWidth;

		dim3 blockGridRows(blockGridWidth, blockGridHeight);
		dim3 threadBlockRows(256, 1);

		// Start the timer.
		// We want to measure copying data, running the kernel, and copying the results back to host
        CUT_SAFE_CALL( cutResetTimer(hTimer) );
        CUT_SAFE_CALL( cutStartTimer(hTimer) );

		if (useGPU == 1)
		{

			// Copy the data to the device
			CUDA_SAFE_CALL( cudaMemcpy(d_dataA, h_dataA, sizeof(float) * dataAmount, cudaMemcpyHostToDevice) );
			CUDA_SAFE_CALL( cudaMemcpy(d_dataB, h_dataB, sizeof(float) * dataAmount, cudaMemcpyHostToDevice) );

			// Do the multiplication on the GPU
			multiplyNumbersGPU<<<blockGridRows, threadBlockRows>>>(d_dataA, d_dataB, d_resultC);
			CUT_CHECK_ERROR("multiplyNumbersGPU() execution failed\n");
			CUDA_SAFE_CALL( cudaThreadSynchronize() );

			// Copy the data back to the host
			CUDA_SAFE_CALL( cudaMemcpy(h_resultC, d_resultC, sizeof(float) * dataAmount, cudaMemcpyDeviceToHost) );
		}
		else
		{
			// We're using the CPU only
			GoldenBrick(h_dataA, h_dataB, h_resultC, dataAmount);
		}

		// Stop the timer, print the total round trip execution time.
		CUT_SAFE_CALL(cutStopTimer(hTimer));
		gpuTime = cutGetTimerValue(hTimer);
		if (!firstRun || !useGPU)
		{
			printf("Elements: %d - convolution time : %f msec - %f Multiplications/sec\n", dataAmount, gpuTime, blockGridHeight * 128 * 256 / (gpuTime * 0.001));
		}
		else
		{
			firstRun = 0;
			// We discard the results of the first run because of the extra overhead incurred
			// during the first time a kernel is ever executed.
			dataAmount *= 2;	// reset to first run value
		}
	}

    printf("Cleaning up...\n");
	CUDA_SAFE_CALL( cudaFree(d_resultC ) );
	CUDA_SAFE_CALL( cudaFree(d_dataB) );
	CUDA_SAFE_CALL( cudaFree(d_dataA) );
	free(h_resultC);
	free(h_dataB);
	free(h_dataA);

    CUT_SAFE_CALL(cutDeleteTimer(hTimer));
    CUT_EXIT(argc, argv);
}

