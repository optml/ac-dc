__global__ void setup_kernel(curandState *state);

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
		E[id] = mlt2 * (t * t * LLocal / 2 + lambdaParameter * xLocal
				* (signbit(xLocal) - signbit(xLocal + t))) + (1 - mlt2)
				* (lambdaParameter * xLocal * signbit(xLocal) - derivatives[id]
						* t - LLocal * t * t / 2);

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
		E[id] = mlt2 * (t * t * LLocal / 2 + lambdaParameter * xLocal
				* (signbit(xLocal) - signbit(xLocal + t))) + (1 - mlt2)
				* (lambdaParameter * xLocal * signbit(xLocal) - derivatives[id]
						* t - LLocal * t * t / 2);
	}
}

__global__ void IncreaseElementAndShrink(int id, float *T, float * E,
		float* derivatives, float* L, float* x, float lambdaParameter, int n) {
	x[id] = x[id] + T[id];

	float LLocal = L[id];
	float Li = 1 / LLocal;
	float xLocal = x[id];
	float lambdaOverL = lambdaParameter * Li;
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
	E[id] = mlt2 * (t * t * LLocal / 2 + lambdaParameter * xLocal * (signbit(
			xLocal) - signbit(xLocal + t))) + (1 - mlt2) * (lambdaParameter
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

