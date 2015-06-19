/*
 * This is solver for so called L1-L2 problem
 *
 * min_x 0.5* \|A*x-b\|_2^2 + \lambda * \|x\|_1
 *
 */

void runSerialRandomL2L1Solver(int n, int m,
		thrust::host_vector<float> h_A_values,
		thrust::host_vector<int> h_A_row_index,
		thrust::host_vector<int> h_A_col_index, thrust::host_vector<float> h_b,
		float lambda, optimization_setting settings,
		thrust::host_vector<float> *h_x_result) {
	thrust::host_vector<float> h_residuals(m); // compute residuals = -b;
	thrust::transform(h_b.begin(), h_b.end(), h_residuals.begin(),
			thrust::negate<float>());
	thrust::host_vector<float> h_x(n, 0);
	thrust::host_vector<float> h_Li(n, 0);
	for (unsigned int i = 0; i < n; i++) {
		for (unsigned int j = 0; j < 4; j++) {
			h_Li[i] += h_A_values[4 * i + j] * h_A_values[4 * i + j];
		}
		if (h_Li[i] > 0)
			h_Li[i] = 1 / h_Li[i]; // Compute reciprocal Lipschitz Constants
	}
	//Prepare counters
	long long serial_iterations = 0;
	float measuredExecutionTime;
	float total_measuredExecutionTime = 0;

	// setup arguments
	square<float> unary_op_square;
	absolute_value<float> unary_op_abs_value;
	thrust::plus<float> binary_op;
	float init = 0;
	if (settings.show_objective_value_at_the_beggining) {
		float norm_g_sq = thrust::transform_reduce(h_residuals. begin(),
				h_residuals . end(), unary_op_square, init, binary_op);
		float norm_x = thrust::transform_reduce(h_x. begin(), h_x. end(),
				unary_op_abs_value, init, binary_op);
		double rcdmObjectiveValue = norm_x * lambda + 0.5 * norm_g_sq;
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
		for (unsigned int j = 0; j
				< settings.number_of_inner_iterations_per_kernel_run; j++) {
			//--------------- One single RCDM Iteration
			float tmp = (float) rand() / RAND_MAX;
			int idx = (int) (n * tmp);
			tmp = 0;
			for (unsigned int j = idx * COLUMNLENGTH; j < idx * COLUMNLENGTH
					+ COLUMNLENGTH; j++) {
				tmp += h_A_values[j] * h_residuals[h_A_row_index[j] - 1];
			}
			float tmp1 = h_Li[idx] * (tmp + lambda);
			if (h_x[idx] > tmp1) {
				tmp = -tmp1;
			} else {
				tmp1 = h_Li[idx] * (tmp - lambda);
				if (h_x[idx] < tmp1) {
					tmp = -tmp1;
				} else {
					tmp = -h_x[idx];
				}
			}
			h_x[idx] += tmp;
			for (unsigned int j = idx * COLUMNLENGTH; j < idx * COLUMNLENGTH
					+ COLUMNLENGTH; j++) {
				h_residuals[h_A_row_index[j] - 1] += tmp * h_A_values[j];
			}
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&measuredExecutionTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		serial_iterations += settings.number_of_inner_iterations_per_kernel_run;
		total_measuredExecutionTime += measuredExecutionTime;
		if (settings.show_objective_values_during_optimization) {
			float norm_g_sq = thrust::transform_reduce(h_residuals. begin(),
					h_residuals . end(), unary_op_square, init, binary_op);
			float norm_x = thrust::transform_reduce(h_x. begin(), h_x. end(),
					unary_op_abs_value, init, binary_op);
			double rcdmObjectiveValue = norm_x * lambda + 0.5 * norm_g_sq;
			printf(
					"SR,%f,%f,%1.16f\n",
					total_measuredExecutionTime, serial_iterations
							/ total_measuredExecutionTime, rcdmObjectiveValue);
		}
		//		if (settings.do_restart_step_after_kernel_run) {
		//TODO
		//	}
	}
	if (settings.show_objective_value_at_the_end) {
		float norm_g_sq = thrust::transform_reduce(h_residuals. begin(),
				h_residuals . end(), unary_op_square, init, binary_op);
		float norm_x = thrust::transform_reduce(h_x. begin(), h_x. end(),
				unary_op_abs_value, init, binary_op);
		double rcdmObjectiveValue = norm_x * lambda + 0.5 * norm_g_sq;
		printf("================SUMMARY==================\n");
		printf("Total elapsed time: %f ms = %f sec = %f min = %f hours  \n",
				total_measuredExecutionTime,
				total_measuredExecutionTime / 1000, total_measuredExecutionTime
						/ 1000 / 60, total_measuredExecutionTime / 1000 / 60
						/ 60);
		printf("Average speed:  %f it/ms \n", serial_iterations
				/ total_measuredExecutionTime);
		printf("Objective value: %1.16f \n", rcdmObjectiveValue);
		printf("Total iterations: %1.5f M \n", (float) serial_iterations
				/ 1000000);
		printf("=========================================\n");
	}
	(*h_x_result) = h_x;
}
