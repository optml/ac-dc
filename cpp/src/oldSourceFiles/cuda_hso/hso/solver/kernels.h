void checkCUDAError(const char *msg);


__device__ __constant__ float c_params_lambda;
__device__ __constant__ int c_params_n;
__device__ __constant__ int c_params_alignedSize;
__device__ __constant__ int c_params_alignedBlocks;
__device__ __constant__ int c_params_ALIGNED_SIZE_FLOAT;
__device__ __constant__ int c_params_inner_kernel_iterations;

void init_constant_device_variables(float lambda, int n,
		int aligned_data_blocks, int aligned_data_size,
		int device_memmory_aligned_data_elements, int total_inner_iterations) {
	cudaMemcpyToSymbol(c_params_lambda, &lambda, sizeof(float));
	cudaMemcpyToSymbol(c_params_n, &n, sizeof(int));
	cudaMemcpyToSymbol(c_params_alignedBlocks, &aligned_data_blocks,
			sizeof(int));
	cudaMemcpyToSymbol(c_params_alignedSize, &aligned_data_size, sizeof(int));
	int datac_params_ALIGNED_SIZE_FLOAT = device_memmory_aligned_data_elements;
	cudaMemcpyToSymbol(c_params_ALIGNED_SIZE_FLOAT,
			&datac_params_ALIGNED_SIZE_FLOAT, sizeof(int));
	cudaMemcpyToSymbol(c_params_inner_kernel_iterations,
				&total_inner_iterations, sizeof(int));
	checkCUDAError("Copy data to constant memory");
}



__global__ void setup_kernel(curandState *state);

__global__ void ShrinkKernel(float *T, float * E, float* derivatives, float* L,
		float* x ) {
	int id = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	if (id <c_params_n) {
		float LLocal = L[id];
		float Li = 1 / LLocal;
		float xLocal = x[id];
		float lambdaOverL = c_params_lambda * Li;
		float alpha = derivatives[id] * Li;
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
		T[id] = t;
		E[id] = mlt2 * (t * t * LLocal / 2 + c_params_lambda * xLocal
				* (signbit(xLocal) - signbit(xLocal + t))) + (1 - mlt2)
				* (c_params_lambda * xLocal * signbit(xLocal) - derivatives[id]
						* t - LLocal * t * t / 2);

	}
}

__global__ void ShrinkKernelSubset(float *T, float * E, float* derivatives,
		float* L, float* x,   int ACounts, int* Indexes) {
	int id = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	if (id < ACounts) {
		id = Indexes[id];
		float LLocal = L[id];
		float Li = 1 / LLocal;
		float xLocal = x[id];
		float lambdaOverL = c_params_lambda * Li;
		float alpha = derivatives[id] * Li;
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
		T[id] = t;
		E[id] = mlt2 * (t * t * LLocal / 2 + c_params_lambda * xLocal
				* (signbit(xLocal) - signbit(xLocal + t))) + (1 - mlt2)
				* (c_params_lambda * xLocal * signbit(xLocal) - derivatives[id]
						* t - LLocal * t * t / 2);
	}
}

__global__ void IncreaseElementAndShrink(int id, float *T, float * E,
		float* derivatives, float* L, float* x ) {
	x[id] = x[id] + T[id];

	float LLocal = L[id];
	float Li = 1 / LLocal;
	float xLocal = x[id];
	float lambdaOverL = c_params_lambda * Li;
	float alpha = derivatives[id] * Li;
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
	T[id] = t;
	E[id] = mlt2 * (t * t * LLocal / 2 + c_params_lambda * xLocal * (signbit(
			xLocal) - signbit(xLocal + t))) + (1 - mlt2) * (c_params_lambda
			* xLocal * signbit(xLocal) - derivatives[id] * t - LLocal * t * t
			/ 2);

}
__global__ void IncreaseElement(float* x, int element, float* T) {
	x[element] = x[element] + T[element];

}
__global__ void setup_kernel(curandState *state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	/* Each thread gets different seed, a different sequence number, no offset */
	curand_init(i, i, 0, &state[i]);
}
void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}





