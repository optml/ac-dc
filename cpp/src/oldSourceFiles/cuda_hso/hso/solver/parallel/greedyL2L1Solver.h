/*
 * This is solver for so called L1-L2 problem
 *
 * min_x 0.5* \|A*x-b\|_2^2 + \lambda * \|x\|_1
 *
 */

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
void runParallelGreedyL2L1Solver(int n, int m,
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
	//---------------------------- Compute r=-b
	thrust::host_vector<float> h_residuals(m); // compute residuals = -b;
	thrust::transform(h_b.begin(), h_b.end(), h_residuals.begin(),
			thrust::negate<float>());
	//-------------------------- Transformation to another sparse representation
	thrust::host_vector<float> h_A_transpose_values(h_A_values.size(), 0);
	thrust::host_vector<int> h_A_transpose_col_idx(h_A_values.size(), 0);
	thrust::host_vector<int> h_A_transpose_count(m + 1, 0);
	thrust::host_vector<int> h_actualCount(m + 1, 0);
	thrust::host_vector<int> h_rowCounts(m + 1, 0);
	for (unsigned int i = 0; i < h_A_values.size(); i++) {
		h_A_transpose_count[h_A_row_index[i] - 1]++; // Shift from 1-based to 0 based
	}
	h_rowCounts[0] = h_A_transpose_count[0];
	for (unsigned int i = 1; i < m + 1; i++) {
		int tmpCount = h_A_transpose_count[i];
		h_rowCounts[i] = h_A_transpose_count[i];
		h_A_transpose_count[i] += h_A_transpose_count[i - 1];
		h_actualCount[i] = h_A_transpose_count[i] - tmpCount;
	}
	h_A_transpose_count = h_actualCount;
	for (unsigned int i = 0; i < n; i++) {
		for (unsigned int j = COLUMNLENGTH * i; j < COLUMNLENGTH * i
				+ COLUMNLENGTH; j++) {
			int tmprow = h_A_row_index[j] - 1;
			h_A_transpose_col_idx[h_actualCount[tmprow]] = i;
			h_A_transpose_values[h_actualCount[tmprow]] = h_A_values[j];
			h_actualCount[tmprow]++;
		}
	}
	//------------------------- Compute Lipchitz Constants.
	thrust::host_vector<float> h_L(n, 0);
	for (unsigned int i = 0; i < n; i++) {
		for (unsigned int j = 0; j < 4; j++) {
			h_L[i] += h_A_values[4 * i + j] * h_A_values[4 * i + j];
		}
	}

	//--------------------------MOVE data to device
	thrust::device_vector<float> d_A_transpose_values = h_A_transpose_values;
	thrust::device_vector<float> d_derivatives(n, 0);
	thrust::device_vector<int> d_A_transpose_col_idx = h_A_transpose_col_idx;
	thrust::device_vector<float> d_L = h_L;
	thrust::device_vector<float> d_x(n, 0);
	thrust::device_vector<int> d_A_transpose_count = h_A_transpose_count;

	float * d_derivatives_raw = thrust::raw_pointer_cast(&d_derivatives[0]);
	int * d_A_transpose_col_idx_raw = thrust::raw_pointer_cast(
			&d_A_transpose_col_idx[0]);
	int * d_A_transpose_count_raw = thrust::raw_pointer_cast(
			&d_A_transpose_count[0]);
	float * d_A_transpose_values_raw = thrust::raw_pointer_cast(
			&d_A_transpose_values[0]);
	float * d_L_raw = thrust::raw_pointer_cast(&d_L[0]);
	float * d_x_raw = thrust::raw_pointer_cast(&d_x[0]);

	thrust::device_vector<float> d_T(n, 0);
	thrust::device_vector<float> d_Energy(n, 0);
	float * d_T_raw = thrust::raw_pointer_cast(&d_T[0]);
	float * d_Energy_raw = thrust::raw_pointer_cast(&d_Energy[0]);

	//-------------------------------Prepare of Derivatives vector on device
	for (unsigned int i = 0; i < m; i++) {
		if (h_b[i] != 0) {
			cusparseStatus_t copystatus = cusparseSaxpyi(handle,
					h_rowCounts[i], -h_b[i],
					&d_A_transpose_values_raw[h_A_transpose_count[i]],
					&d_A_transpose_col_idx_raw[h_A_transpose_count[i]],
					d_derivatives_raw, CUSPARSE_INDEX_BASE_ZERO);
			if (copystatus != CUSPARSE_STATUS_SUCCESS) {
				printf("Error in CuSparse!");
			}
		}
	}

	//-----------------------
	//Prepare counters
	long long parallel_iterations = 0;
	float measuredExecutionTime;
	float total_measuredExecutionTime = 0;

	cudaMemcpyToSymbol(c_params_lambda, &lambda, sizeof(float));
	cudaMemcpyToSymbol(c_params_n, &n, sizeof(int));

	cusparseStatus_t copystatus;

	dim3 dimBlock(settings.device_total_threads_per_block);
	dim3 dimGridRCDM(1, 1 + n / (settings.device_total_threads_per_block));

	int maxIndex = 0;
	ShrinkKernel<<< dimGridRCDM ,dimBlock >>>(d_T_raw, d_Energy_raw, d_derivatives_raw,
			d_L_raw,d_x_raw );
	int maxShrinkSubset = 0;
	for (unsigned int i = 0; i < m; i++) {
		if (h_rowCounts[i] > maxShrinkSubset)
			maxShrinkSubset = h_rowCounts[i];
	}
	printf("Max shrink subset %d\n", maxShrinkSubset);
	dim3 dimGridShrinkSubset(1, 1 + maxShrinkSubset
			/ (settings.device_total_threads_per_block));

	thrust::device_vector<float> d_residuals = h_residuals;
	float * d_residuals_raw = thrust::raw_pointer_cast(&d_residuals[0]);

	cusparseMatDescr_t descra = 0;
	status = cusparseCreateMatDescr(&descra);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("Matrix descriptor initialization failed");
	}
	cusparseSetMatType(descra, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ZERO);
	if (settings.show_objective_value_at_the_beggining) {
		status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n,
				1, descra, d_A_transpose_values_raw, d_A_transpose_count_raw,
				d_A_transpose_col_idx_raw, d_x_raw, 1, d_residuals_raw);
		if (status != CUSPARSE_STATUS_SUCCESS)
			printf("Error in  CuSparse!\n");
		float norm_g = cublasSnrm2(m, d_residuals_raw, 1);
		float norm_x = cublasSasum(n, d_x_raw, 1);
		double greedyObjectiveValue = norm_x * lambda + 0.5 * norm_g * norm_g;
		printf("Initial objective value = %1.16f \n", greedyObjectiveValue);
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
		for (unsigned int j = 0; j
				< settings.number_of_inner_iterations_per_kernel_run; j++) {
			maxIndex = cublasIsamax(n, d_Energy_raw, 1);
			checkCUDAError("Get point");
			maxIndex = maxIndex - 1;
			float optimalStep = d_T[maxIndex];
			IncreaseElementAndShrink<<< 1 ,1 >>>(
					maxIndex,
					d_T_raw,
					d_Energy_raw,
					d_derivatives_raw,
					d_L_raw,d_x_raw );
			checkCUDAError("IncreaseElement");
			for (unsigned int ki = maxIndex * COLUMNLENGTH; ki < maxIndex
					* COLUMNLENGTH + COLUMNLENGTH; ki++) {
				copystatus
						= cusparseSaxpyi(
								handle,
								h_rowCounts[h_A_row_index[ki] - 1],
								optimalStep * h_A_values[ki],
								&d_A_transpose_values_raw[h_A_transpose_count[h_A_row_index[ki]
										- 1]],
								&d_A_transpose_col_idx_raw[h_A_transpose_count[h_A_row_index[ki]
										- 1]], d_derivatives_raw,
								CUSPARSE_INDEX_BASE_ZERO);
				if (copystatus != CUSPARSE_STATUS_SUCCESS)
					printf("Error in  CuSparse!\n");
				ShrinkKernelSubset<<< dimGridShrinkSubset ,dimBlock >>>(
						d_T_raw,
						d_Energy_raw,
						d_derivatives_raw,
						d_L_raw, d_x_raw,
						h_rowCounts[h_A_row_index[ki] - 1],
						&d_A_transpose_col_idx_raw[h_A_transpose_count[h_A_row_index[ki] - 1]]);
				checkCUDAError("Shrink subset");
			}
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&measuredExecutionTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		total_measuredExecutionTime += measuredExecutionTime;
		parallel_iterations
				+= settings.number_of_inner_iterations_per_kernel_run;
		if (settings.show_objective_values_during_optimization) {
			d_residuals = h_residuals;
			d_residuals_raw = thrust::raw_pointer_cast(&d_residuals[0]);
			status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					m, n, 1, descra, d_A_transpose_values_raw,
					d_A_transpose_count_raw, d_A_transpose_col_idx_raw,
					d_x_raw, 1, d_residuals_raw);
			if (status != CUSPARSE_STATUS_SUCCESS)
				printf("Error in  CuSparse!\n");
			float norm_g = cublasSnrm2(m, d_residuals_raw, 1);
			float norm_x = cublasSasum(n, d_x_raw, 1);
			double greedyObjectiveValue = norm_x * lambda + 0.5 * norm_g
					* norm_g;
			printf(
					"PG,%f,%f,%1.16f\n",
					total_measuredExecutionTime, parallel_iterations
							/ total_measuredExecutionTime, greedyObjectiveValue);
		}
		if (settings.do_restart_step_after_kernel_run) {
			//TODO
		}
	}
	if (settings.show_objective_value_at_the_end) {
		d_residuals = h_residuals;
		d_residuals_raw = thrust::raw_pointer_cast(&d_residuals[0]);
		status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n,
				1, descra, d_A_transpose_values_raw, d_A_transpose_count_raw,
				d_A_transpose_col_idx_raw, d_x_raw, 1, d_residuals_raw);
		if (status != CUSPARSE_STATUS_SUCCESS)
			printf("Error in  CuSparse!\n");
		float norm_g = cublasSnrm2(m, d_residuals_raw, 1);
		float norm_x = cublasSasum(n, d_x_raw, 1);
		double greedyObjectiveValue = norm_x * lambda + 0.5 * norm_g * norm_g;
		printf("================SUMMARY==================\n");
		printf("Total elapsed time: %f ms = %f sec = %f min = %f hours  \n",
				total_measuredExecutionTime,
				total_measuredExecutionTime / 1000, total_measuredExecutionTime
						/ 1000 / 60, total_measuredExecutionTime / 1000 / 60
						/ 60);
		printf("Average speed:  %f it/ms \n", parallel_iterations
				/ total_measuredExecutionTime);
		printf("Objective value: %1.16f \n", greedyObjectiveValue);
		printf("Total iterations: %1.5f M \n", (float) parallel_iterations
				/ 1000000);
		printf("=========================================\n");
	}

	(*h_x_result) = d_x; //get data from device to host
}
