/*
 * MultiCoreSolver.cpp
 *
 *  Created on: Sep 9, 2013
 *      Author: taki
 */

#include "../class/Context.h"
#include "../helpers/option_console_parser.h"
#include "../solver/settingsAndStatistics.h"
#include "../utils/file_reader.h"
#include "../solver/MulticoreEngineExecutor.h"
#include "../solver/Solver.h"
#include "../helpers/utils.h"
#include "../utils/distributed_instances_loader.h"
#include <math.h>

int main(int argc, char *argv[]) {
	OptimizationSettings settings;
	Context ctx(settings);
	consoleHelper::parseConsoleOptions(ctx, argc, argv);
	ProblemData<int, double> instance;

	double omega = 1;
//n = samples
	//m =features
	loadDistributedSparseSVMRowData(ctx.matrixAFile, -1, -1, instance, false);
	instance.sigma = 1;

	std::vector<double> x(instance.n);
	std::vector<double> y(instance.m);

	for (long i = 0; i < instance.n; i++) {
		x[i] = rand() / (RAND_MAX + 0.0);

		double L = 0;
		for (long j = instance.A_csr_row_ptr[i];
				j < instance.A_csr_row_ptr[i + 1]; j++) {
			L += instance.A_csr_values[j] * instance.A_csr_values[j];
		}

		double scale = sqrt(L);
		if (scale > 0) {
			scale=1/L;
			for (long j = instance.A_csr_row_ptr[i];
					j < instance.A_csr_row_ptr[i + 1]; j++) {
				instance.A_csr_values[j] = instance.A_csr_values[j] * scale;
			}
		}
	}

	double norm = cblas_l2_norm(instance.n, &x[0], 1);
	cblas_vector_scale(instance.n, &x[0], 1 / norm);

	for (int it = 0; it < 20; it++) {
		norm = cblas_l2_norm(instance.n, &x[0], 1);
		std::cout << "odhad je " << norm << std::endl;
		cblas_vector_scale(instance.n, &x[0], 1 / norm);

		cblas_vector_scale(instance.m, &y[0], (double) 0);
		for (long i = 0; i < instance.n; i++) {
			for (long j = instance.A_csr_row_ptr[i];
					j < instance.A_csr_row_ptr[i + 1]; j++) {
				y[instance.A_csr_col_idx[j]] += instance.A_csr_values[j] * x[i];
			}
		}
		cblas_vector_scale(instance.n, &x[0], (double) 0);
		for (long i = 0; i < instance.n; i++) {
			for (long j = instance.A_csr_row_ptr[i];
					j < instance.A_csr_row_ptr[i + 1]; j++) {
				x[i] += y[instance.A_csr_col_idx[j]] * instance.A_csr_values[j];
			}
		}

	}
}

