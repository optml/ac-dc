/*
 * This is solver for so called L1-L2 problem
 *
 * min_x 0.5* \|A*x-b\|_2^2 + \lambda * \|x\|_1
 *
 */
__global__ void setup_ordered_kernel(curandState *state) {
	int i = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	int j = blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x;
	j += 10 * ((int) (threadIdx.x / 32)); //Each wrap will have the same sequence of random numbers
	curand_init(j, j, 0, &state[i]);
}

__global__ void solveRandomL2L1Kernel(float *A, int*R_Idx, float*residuals,
		float*x, curandState* cstate, float* Li) {
	int j;
	//		float ALocal[4];
	int id = (blockIdx.y * blockDim.x * blockDim.y + blockIdx.x * blockDim.x
			+ threadIdx.x);
	curandState localState = cstate[id];
	for (unsigned int k = 0; k < c_params_inner_kernel_iterations; k++) {

		double d = curand_uniform_double(&localState);
		int idx = ((int) (d * c_params_alignedBlocks))
				* c_params_ALIGNED_SIZE_FLOAT + threadIdx.x;
		if (idx < c_params_n) {
			float partialDetivative = 0;
			j = idx;
			for (unsigned int i = 0; i < COLUMNLENGTH; i++) {
				//								ALocal[i] = A[j];
				//								partialDetivative += ALocal[i] * residuals[R_Idx[j] - 1];
				partialDetivative += A[j] * residuals[R_Idx[j] - 1];
				j += c_params_alignedSize;
			}
			float xLocal = x[idx];
			float LiLocal = Li[idx];
			float delta;
			//Computing optimal step
			float tmp = LiLocal * (partialDetivative + c_params_lambda);
			if (xLocal > tmp) {
				delta = -tmp;
			} else {
				tmp = LiLocal * (partialDetivative - c_params_lambda);
				if (xLocal < tmp) {
					delta = -tmp;
				} else {
					delta = -xLocal;
				}
			}
			atomicAdd(&x[idx], delta); // Attomic update of x[idx]
			j = idx;
			for (unsigned int i = 0; i < COLUMNLENGTH; i++) {
				// Attomic update of corresponding residuals
				atomicAdd(&residuals[R_Idx[j] - 1], A[j] * delta);
				//								atomicAdd(&residuals[R_Idx[j] - 1],ALocal[i]* delta);
				j += c_params_alignedSize;
			}
		}
	}
	cstate[id] = localState;
}

/*
 *
 * Solve problem \min \|A*x-b\|_2^2+\lambda \|x\|_1
 *
 * INPUT:
 * n - number of colums of matrix A
 * m - number of rows of matrix A
 * h_A_values -  values of matrix A. We assume that this is one long vector with elements
 *               A is stored in columwise order (at first elements from first column,
 *               then second column follows
 * h_b - vector b
 * lambda - parameter lambda
 * settings - settings to optimizer
 * h_x_result - pointer for result
 */
void runParallelRandomL2L1Solver(int n, int m,
		thrust::host_vector<float> h_A_values,
		thrust::host_vector<int> h_A_row_index,
		thrust::host_vector<int> h_A_col_index, thrust::host_vector<float> h_b,
		float lambda, optimization_setting settings,
		thrust::host_vector<float> *h_x_result) {
	cusparseStatus_t status;
	cusparseHandle_t handle = 0;
	/* initialize cusparse library */
	status = cusparseCreate(&handle);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("CUSPARSE Library initialization failed");
	}
	thrust::host_vector<float> h_residuals(m); // compute residuals = -b;
	thrust::transform(h_b.begin(), h_b.end(), h_residuals.begin(),
			thrust::negate<float>());
	thrust::device_vector<float> d_x(n, 0);
	//======================Permute data============================
	thrust::host_vector<int> hostPermutator(n);
	thrust::sequence(hostPermutator.begin(), hostPermutator.end());
	std::random_shuffle(hostPermutator.begin(), hostPermutator.end());
	//compute how to align data in memory for increase CUDA performance
	int aligned_data_blocks = (int) (n
			/ settings.device_memmory_aligned_data_elements) + 1;
	int aligned_data_size = aligned_data_blocks
			* settings.device_memmory_aligned_data_elements;

	//	printf("n:%d, aligned_data_size:%d,aligned_data_blocks:%d\n",n,aligned_data_size ,aligned_data_blocks);

	thrust::host_vector<float> h_A_values_permuted_Aligned(aligned_data_size
			* 4, 0);
	thrust::host_vector<int> h_A_row_index_permuted_Aligned(aligned_data_size
			* 4, 0);
	thrust::host_vector<float> h_Li(n, 0);
	for (unsigned int i = 0; i < n; i++) {
		int k = hostPermutator[i];
		for (unsigned int j = 0; j < COLUMNLENGTH; j++) {
			h_A_values_permuted_Aligned[i + j * aligned_data_size]
					= h_A_values[4 * k + j];
			h_A_row_index_permuted_Aligned[i + j * aligned_data_size]
					= h_A_row_index[4 * k + j];
			h_Li[i] += h_A_values_permuted_Aligned[i + j * aligned_data_size]
					* h_A_values_permuted_Aligned[i + j * aligned_data_size];
		}
		if (h_Li[i] > 0)
			h_Li[i] = 1 / h_Li[i]; // Compute reciprocal Lipschitz Constants
	}
	//initialize data on device
	thrust::device_vector<float> d_A_values = h_A_values_permuted_Aligned;
	thrust::device_vector<int> d_A_row_index = h_A_row_index_permuted_Aligned;
	thrust::device_vector<float> d_residuals = h_residuals;
	thrust::device_vector<float> d_Li = h_Li;
	// obtain raw pointer to device vector’s memory
	int * d_A_row_index_raw = thrust::raw_pointer_cast(&d_A_row_index[0]);
	float * d_A_values_raw = thrust::raw_pointer_cast(&d_A_values[0]);
	float * d_x_raw = thrust::raw_pointer_cast(&d_x[0]);
	float * d_residuals_raw = thrust::raw_pointer_cast(&d_residuals[0]);
	float * d_Li_raw = thrust::raw_pointer_cast(&d_Li[0]);
	dim3 dimBlockRCD2(settings.device_total_threads_per_block);
	dim3 dimGridRCDM2(settings.device_block_dim_1, settings.device_block_dim_2);

	//Initialize Random Generator for each thread
	curandState *devStates;
	cudaMalloc((void **) &devStates, settings.device_block_dim_1
			* settings.device_block_dim_2
			* settings.device_total_threads_per_block * sizeof(curandState));
	setup_ordered_kernel<<< dimGridRCDM2, dimBlockRCD2 >>>(devStates);
	checkCUDAError("Inicializacia ranom states");

	//Prepare counters
	long long parallel_iterations = 0;
	float measuredExecutionTime;
	float total_measuredExecutionTime = 0;

	//Initialize __constant__ variables on device
	init_constant_device_variables(lambda, n, aligned_data_blocks,
			aligned_data_size, settings.device_memmory_aligned_data_elements,
			settings.number_of_inner_iterations_per_kernel_run);

	if (settings.show_objective_value_at_the_beggining) {
		float norm_g = cublasSnrm2(m, d_residuals_raw, 1);
		float norm_x = cublasSasum(n, d_x_raw, 1);
		double rcdmObjectiveValue = norm_x * lambda + 0.5 * norm_g * norm_g;
		printf("Initial objective value = %1.16f \n", rcdmObjectiveValue);
	}
	if (settings.show_objective_values_during_optimization) {
		printf(
				"Method,Elapsed time[ms],Average Speed[it/ms],Current objective value\n");
	}
	while (total_measuredExecutionTime / 1000 < settings.total_execution_time) {
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		solveRandomL2L1Kernel <<< dimGridRCDM2, dimBlockRCD2 >>>(d_A_values_raw, d_A_row_index_raw,
				d_residuals_raw,
				d_x_raw,
				devStates,
				d_Li_raw
		);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&measuredExecutionTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		parallel_iterations += settings.device_block_dim_1
				* settings.device_block_dim_2
				* settings.number_of_inner_iterations_per_kernel_run
				* settings.device_total_threads_per_block;
		total_measuredExecutionTime += measuredExecutionTime;
		if (settings.show_objective_values_during_optimization) {
			float norm_g = cublasSnrm2(m, d_residuals_raw, 1);
			float norm_x = cublasSasum(n, d_x_raw, 1);
			double rcdmObjectiveValue = norm_x * lambda + 0.5 * norm_g * norm_g;
			printf(
					"PR,%f,%f,%1.16f\n",
					total_measuredExecutionTime, parallel_iterations
							/ total_measuredExecutionTime, rcdmObjectiveValue);
		}
		if (settings.do_restart_step_after_kernel_run) {
			//TODO

		}
	}

	if (settings.show_objective_value_at_the_end) {
		float norm_g = cublasSnrm2(m, d_residuals_raw, 1);
		float norm_x = cublasSasum(n, d_x_raw, 1);
		double rcdmObjectiveValue = norm_x * lambda + 0.5 * norm_g * norm_g;
		printf("================SUMMARY==================\n");
		printf("Total elapsed time: %f ms = %f sec = %f min = %f hours  \n",
				total_measuredExecutionTime,
				total_measuredExecutionTime / 1000, total_measuredExecutionTime
						/ 1000 / 60, total_measuredExecutionTime / 1000 / 60
						/ 60);
		printf("Average speed:  %f it/ms \n", parallel_iterations
				/ total_measuredExecutionTime);
		printf("Objective value: %1.16f \n", rcdmObjectiveValue);
		printf("Total iterations: %1.5f M \n", (float) parallel_iterations
				/ 1000000);
		printf("=========================================\n");
	}

	thrust::host_vector<float> h_x_computed = d_x; //get data from device to host
	(*h_x_result).resize(n);
	for (unsigned int i = 0; i < n; i++) {
		(*h_x_result)[hostPermutator[i]] = h_x_computed[i]; //unpermute data
	}
	cudaFree(devStates);
}
