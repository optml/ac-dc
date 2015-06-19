#include  <stdio.h>

//  Kernel definition
__global__ void vecAdd(float* A, float* B, float* C) {
	int i = threadIdx.x;
	A[i] = 0;
	B[i] = i;
	C[i] = i;
}
	int devcheck(int);


#define  SIZE 10
int main() {
	devcheck(0);

	int N = SIZE;
	float A[SIZE], B[SIZE], C[SIZE];
	// Kernel invocation

	int j;
	for (j=0;j<N;j++)
	{
		A[j]=1;
		B[j]=1;
		C[j]=1;
	}

	float *devPtrA;
	float *devPtrB;
	float *devPtrC;
	int memsize = SIZE * sizeof(float);

	cudaMalloc((void**) &devPtrA, memsize);
	cudaMalloc((void**) &devPtrB, memsize);
	cudaMalloc((void**) &devPtrC, memsize);
	cudaMemcpy(devPtrA, A, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(devPtrB, B, memsize, cudaMemcpyHostToDevice);
	vecAdd<<<1, N>>>(devPtrA, devPtrB, devPtrC);
	cudaMemcpy(C, devPtrC, memsize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < SIZE; i++)
		printf("C[%d]=%f\n", i, C[i]);

	cudaFree(devPtrA);
	cudaFree(devPtrA);
	cudaFree(devPtrA);
}

int devcheck(int gpudevice) {
	int device_count = 0;
	int device; // used with  cudaGetDevice() to verify cudaSetDevice()

	// get the number of non-emulation devices  detected
	cudaGetDeviceCount(&device_count);
	if (gpudevice > device_count) {
		printf("gpudevice >=  device_count ... exiting\n");
		exit(1);
	}
	cudaError_t cudareturn;
	cudaDeviceProp deviceProp;

	// cudaGetDeviceProperties() is also  demonstrated in the deviceQuery/ example
	// of the sdk projects directory
	cudaGetDeviceProperties(&deviceProp, gpudevice);
	printf("[deviceProp.major.deviceProp.minor] = [%d.%d]\n", deviceProp.major,
			deviceProp.minor);

	printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
	printf("  Number of cores:                               %d\n", deviceProp.major );

	printf("  Total amount of global memory:                 %u bytes\n", deviceProp.totalGlobalMem);
	printf("  Total amount of constant memory:               %u bytes\n", deviceProp.totalConstMem);
	printf("  Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
	printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
	printf("  Warp size:                                     %d\n", deviceProp.warpSize);
	printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
	printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
	printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
	printf("  Maximum memory pitch:                          %u bytes\n", deviceProp.memPitch);
	printf("  Texture alignment:                             %u bytes\n", deviceProp.textureAlignment);
	printf("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);

	if (deviceProp.major > 999) {
		printf("warning, CUDA Device  Emulation (CPU) detected, exiting\n");
		exit(1);
	}

	// choose a cuda device for kernel  execution
	cudareturn = cudaSetDevice(gpudevice);
	if (cudareturn == cudaErrorInvalidDevice) {
		perror("cudaSetDevice returned  cudaErrorInvalidDevice");
	} else {
		// double check that device was  properly selected
		cudaGetDevice(&device);
		printf("cudaGetDevice()=%d\n", device);
	}
	return 1;
}
