/*
 * Example of calling Random L2-L1 solver for 2D TTD problem
 */
#define COLUMNLENGTH 4
/* columLength - length of each column. We requite each column to have the same length
 *               one can fill them with zero values, but the input data has to fulfill this
 *               requirement!
 */
#include "headers.h" // All neccessary libs are included
using namespace std; // namespace for std;


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

	printf("  Number of multiprocessors:                     %d\n",
			deviceProp.multiProcessorCount);
	printf("  Number of cores:                               %d\n",
			deviceProp.major);

	printf("  Total amount of global memory:                 %u bytes\n",
			deviceProp.totalGlobalMem);
	printf("  Total amount of constant memory:               %u bytes\n",
			deviceProp.totalConstMem);
	printf("  Total amount of shared memory per block:       %u bytes\n",
			deviceProp.sharedMemPerBlock);
	printf("  Total number of registers available per block: %d\n",
			deviceProp.regsPerBlock);
	printf("  Warp size:                                     %d\n",
			deviceProp.warpSize);
	printf("  Maximum number of threads per block:           %d\n",
			deviceProp.maxThreadsPerBlock);
	printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
	printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
	printf("  Maximum memory pitch:                          %u bytes\n",
			deviceProp.memPitch);
	printf("  Texture alignment:                             %u bytes\n",
			deviceProp.textureAlignment);
	printf("  Clock rate:                                    %.2f GHz\n",
			deviceProp.clockRate * 1e-6f);

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
int main(void) {

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int device;
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		CUdevice cuDevice = device;
		int major, minor;

		printf("Device %d has compute capability %d.%d.\n", device,
				deviceProp.major, deviceProp.minor);

		//		cuDeviceGet(&cuDevice, device);
		int unifiedAddress = 9;
		//		 cuDeviceGetAttribute	(	&unifiedAddress,
		//				 CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
		//				 cuDevice
		//		)		;
		printf("share addess:%d\n", unifiedAddress);

	}

	int gpudevice = 3;
	cudaError_t cudareturn;

	cudareturn = cudaSetDevice(gpudevice);
	if (cudareturn == cudaErrorInvalidDevice) {
		perror("cudaSetDevice returned  cudaErrorInvalidDevice");
	} else {
		// double check that device was  properly selected
		cudaGetDevice(&device);
		printf("cudaGetDevice()=%d\n", device);
	}

	printf("idem alokovat data na host\n");
	thrust::host_vector<float> h_A_values(1024*1024*1024,0.2);

	printf("idem alokovat data na device\n");

	try {
//		float* d_f;
//		cudaMalloc(&d_f, 1024*1024*1024*sizeof(float));
		thrust::device_vector<float> d_A_values = h_A_values;
	}
	catch( std::bad_alloc &e) {
		std::cerr<<"Couldn't allocate d_a: " << e.what() <<std::endl;
		exit(-1);
	}

	checkCUDAError("Inicializacia ranom states");
	printf("done\n");
	return 1;
}
