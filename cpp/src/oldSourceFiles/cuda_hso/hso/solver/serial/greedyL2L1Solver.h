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

void runSerialGreedyL2L1Solver(int n, int m,
		thrust::host_vector<float> h_A_values,
		thrust::host_vector<int> h_A_row_index,
		thrust::host_vector<int> h_A_col_index, thrust::host_vector<float> h_b,
		float lambda, optimization_setting settings,
		thrust::host_vector<float> *h_x_result) {
	thrust::host_vector<float> x(n, 0);
	thrust::host_vector<float> L(n, 0);
	thrust::host_vector<float> derivatives(n, 0);
	//=======Computing derivatives =A'(-b)
	for (int i = 0; i < h_A_values.size(); i++) {
		derivatives[h_A_col_index[i] - 1] += -h_A_values[i]
				* h_b[h_A_row_index[i] - 1];
		L[h_A_col_index[i] - 1] += h_A_values[i] * h_A_values[i];
	}
	//=======Transformation===============//TODO Improve!
	int ATcount[m];
	int actualCount[m];
	int rowCounts[m];
	for (unsigned int i = 0; i < m; i++) {
		ATcount[i] = 0;
		actualCount[i] = 0;
	}
	for (unsigned int i = 0; i < h_A_values.size(); i++) {
		ATcount[h_A_row_index[i] - 1]++; // Shift from 1-based to 0 based
	}
	rowCounts[0] = ATcount[0];
	for (unsigned int i = 1; i < m; i++) {
		int tmpCount = ATcount[i];
		rowCounts[i] = ATcount[i];
		ATcount[i] += ATcount[i - 1];
		actualCount[i] = ATcount[i] - tmpCount;
	}
	for (unsigned int i = 0; i < m; i++) {
		ATcount[i] = actualCount[i];
	}
	float ATCB[h_A_values.size()];
	int ColAT[h_A_values.size()];
	for (unsigned int i = 0; i < n; i++) {
		for (unsigned int j = COLUMNLENGTH * i; j < COLUMNLENGTH * i
				+ COLUMNLENGTH; j++) {
			int tmprow = h_A_row_index[j] - 1;
			ColAT[actualCount[tmprow]] = i;
			ATCB[actualCount[tmprow]] = h_A_values[j];
			actualCount[tmprow]++;
		}
	}
	//========Initialization
	int max_IDX = 0;
	float max_VAL = 0;
	float optimalStep = 0;
	float T_optimal_steps[n];
	float Energy[n];
	for (unsigned int j = 0; j < n; j++) {
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
		T_optimal_steps[j] = t;
		Energy[j] = mlt2 * (tmpEnergy / 2 + lambda * xLocal * (signbit(xLocal)
				- signbit(xLocal + t))) + (1 - mlt2) * (lambda * xLocal
				* signbit(xLocal) - derivatives[j] * t - LLocal * t * t / 2);

		if (tmpEnergy > max_VAL) {
			optimalStep = t;
			max_VAL = tmpEnergy;
			max_IDX = j;
		}
	}
	float total_measuredExecutionTime = 0;
	float measuredExecutionTime = 0;
	int serial_iterations = 0;
	//============Computation
	if (settings.show_objective_value_at_the_beggining) {
		double greedyObjectiveValue = computeTTDObjectiveValue(&h_A_values[0],
				&h_A_row_index[0], &h_A_col_index[0], &x[0], n, m, &h_b[0],
				h_A_values.size(), lambda);
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
		for (unsigned int in_it = 0; in_it
				< settings.number_of_inner_iterations_per_kernel_run; in_it++) {
			x[max_IDX] += T_optimal_steps[max_IDX];
			optimalStep = T_optimal_steps[max_IDX];
			for (unsigned int ki = max_IDX * COLUMNLENGTH; ki < max_IDX
					* COLUMNLENGTH + COLUMNLENGTH; ki++) {
				for (unsigned int jj = ATcount[h_A_row_index[ki] - 1]; jj
						< ATcount[h_A_row_index[ki] - 1]
								+ rowCounts[h_A_row_index[ki] - 1]; jj++) {
					int j = ColAT[jj];
					derivatives[j] += optimalStep * h_A_values[ki] * ATCB[jj];
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
					T_optimal_steps[j] = t;
					Energy[j] = mlt2 * (tmpEnergy / 2 + lambda * xLocal
							* (signbit(xLocal) - signbit(xLocal + t))) + (1
							- mlt2) * (lambda * xLocal * signbit(xLocal)
							- derivatives[j] * t - LLocal * t * t / 2);
				}
			}
			max_VAL = 0;
			max_IDX = 0;
			for (int j = 0; j < n; j++) {
				if (Energy[j] > max_VAL) {
					max_VAL = Energy[j];
					max_IDX = j;
				}
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
			double greedyObjectiveValue = computeTTDObjectiveValue(
					&h_A_values[0], &h_A_row_index[0], &h_A_col_index[0],
					&x[0], n, m, &h_b[0], h_A_values.size(), lambda);
			printf(
					"SG,%f,%f,%1.16f\n",
					total_measuredExecutionTime, serial_iterations
							/ total_measuredExecutionTime, greedyObjectiveValue);
		}
		//		if (settings.do_restart_step_after_kernel_run) {
		//TODO
		//	}
	}
	if (settings.show_objective_value_at_the_end) {
		double greedyObjectiveValue = computeTTDObjectiveValue(&h_A_values[0],
				&h_A_row_index[0], &h_A_col_index[0], &x[0], n, m, &h_b[0],
				h_A_values.size(), lambda);
		printf("================SUMMARY==================\n");
		printf("Total elapsed time: %f ms = %f sec = %f min = %f hours  \n",
				total_measuredExecutionTime,
				total_measuredExecutionTime / 1000, total_measuredExecutionTime
						/ 1000 / 60, total_measuredExecutionTime / 1000 / 60
						/ 60);
		printf("Average speed:  %f it/ms \n", serial_iterations
				/ total_measuredExecutionTime);
		printf("Objective value: %1.16f \n", greedyObjectiveValue);
		printf("Total iterations: %1.5f M \n", (float) serial_iterations
				/ 1000000);
		printf("=========================================\n");
	}
	(*h_x_result) = x;
}
