/*
 * MultiCoreSolver.cpp
 *
 *  Created on: Sep 9, 2013
 *      Author: taki
 */

#include "../../class/Context.h"
#include "../../helpers/option_console_parser.h"
#include "../../solver/settingsAndStatistics.h"
#include "../../utils/file_reader.h"
#include "../../solver/MulticoreEngineExecutor.h"
#include "../../solver/Solver.h"
#include "../../helpers/utils.h"
#include "../../utils/distributed_instances_loader.h"
#include <math.h>

int main(int argc, char *argv[]) {
	OptimizationSettings settings;
	Context ctx(settings);
	consoleHelper::parseConsoleOptions(ctx, argc, argv);
	ProblemData<int, double> instance;

	double omega = 1;
	std::vector<int> rowCounts;
	switch (ctx.settings.lossFunction) {
	case 0:
	case 1:
		InputOuputHelper::loadCSCData(ctx, instance);
		std::cout << "Data loaded with size: " << instance.m << " x "
				<< instance.n << std::endl;
		rowCounts.resize(instance.m, 0);
		for (unsigned long i = 0; i < instance.A_csc_row_idx.size(); i++) {
			rowCounts[instance.A_csc_row_idx[i]]++;
			if (rowCounts[instance.A_csc_row_idx[i]] > omega) {
				omega = rowCounts[instance.A_csc_row_idx[i]];
			}
		}
		break;
	case 2:
		loadDistributedSparseSVMRowData(ctx.matrixAFile, -1, -1, instance,
				false);

		if (ctx.isTestErrorFileAvailable) {
			ProblemData<int, double> testInstance;
			loadDistributedSparseSVMRowData(ctx.matrixATestFile, -1, -1,
					testInstance, false);
			instance.A_test_csr_col_idx = testInstance.A_csr_col_idx;
			instance.A_test_csr_row_ptr = testInstance.A_csr_row_ptr;
			instance.A_test_csr_values = testInstance.A_csr_values;
			instance.test_b = testInstance.b;
		}

		instance.lambda = ctx.lambda;
		rowCounts.resize(instance.m, 0);

		for (unsigned long i = 0; i < instance.A_csr_col_idx.size(); i++) {
			rowCounts[instance.A_csr_col_idx[i]]++;
			if (rowCounts[instance.A_csr_col_idx[i]] > omega) {
				omega = rowCounts[instance.A_csr_col_idx[i]];
			}
		}
		instance.omegaAvg = 0;
		instance.omegaMin = rowCounts[0];
		for (unsigned long i = 0; i < rowCounts.size(); i++) {
			instance.omegaAvg += rowCounts[i] / (0.0 + instance.n);
			if (rowCounts[i] < instance.omegaMin) {
				instance.omegaMin = rowCounts[i];
			}
		}
		std::cout << "Omega: " << omega << std::endl;
		std::cout << "Omega-avg: " << instance.omegaAvg << std::endl;
		std::cout << "Omega-min: " << instance.omegaMin << std::endl;
		break;
	case 3:
	case 4:
		loadDistributedSparseSVMRowData(ctx.matrixAFile, -1, -1, instance,
				false);
		instance.lambda = ctx.lambda;
		if (ctx.isTestErrorFileAvailable) {
			ProblemData<int, double> testInstance;
			loadDistributedSparseSVMRowData(ctx.matrixATestFile, -1, -1,
					testInstance, false);
			instance.A_test_csr_col_idx = testInstance.A_csr_col_idx;
			instance.A_test_csr_row_ptr = testInstance.A_csr_row_ptr;
			instance.A_test_csr_values = testInstance.A_csr_values;
			instance.test_b = testInstance.b;
		}
		getCSR_from_CSC(
				instance.A_csr_values, //Input
				instance.A_csr_col_idx, instance.A_csr_row_ptr,
				instance.A_csc_values, //Output
				instance.A_csc_row_idx, instance.A_csc_col_ptr, instance.m,
				instance.n);

		int tmp=instance.n;
		instance.n=instance.m;
		instance.m=tmp;
		for (int i=0;i<instance.m;i++){
			if (instance.A_csr_row_ptr[i+1]-instance.A_csr_row_ptr[i]>omega){
				omega=instance.A_csr_row_ptr[i+1]-instance.A_csr_row_ptr[i];
			}
		}
		std::cout<<"Omega is "<<omega<<std::endl;
		break;
	}

	instance.sigma = 1
			+ (omega - 1) * (ctx.settings.totalThreads - 1)
					/ (instance.n - 1.0);
	MulticoreEngineExecutor<int, double> executor(instance, &(ctx.settings));
	Solver<int, double> solver(executor);
	solver.runSolver();

}

