/*
 * This file contains an experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */

#include "../helpers/c_libs_headers.h"
#include "../utils/randomNumbersUtil.h"
#include "../solver/settingsAndStatistics.h"
#include "../solver/Solver.h"
#include "../solver/lossFunctions/multicoreLossFunctions.h"
#include "../utils/distributed_instances_loader.h"
#include "../solver/MulticoreEngineExecutor.h"

int oneProblem(string dataFile, string logFile, int epochs,int totalThreads, float lambda) {

	cout << "experiments starts" << endl;

	OptimizationSettings settings;
	//---------------------- Set output files
	settings.logFile.open(logFile.c_str());
	//---------------------- Setting parameters of experiment
	double fvalOpt = 0;

	ProblemData<int, double> instance;
	loadDistributedSparseSVMRowData(dataFile, -1, -1, instance, false);
	instance.sigma = 1;
//	instance.sigma = 1
//			+ (omega - 1) * (ctx.settings.totalThreads - 1)
//					/ (instance.n - 1.0);

	settings.lossFunction = 2;
	settings.showInitialObjectiveValue = true;
	settings.bulkIterations = true;
	settings.verbose = true;

	settings.totalThreads = 10;

	settings.innerIterations = instance.n;

	settings.iters_bulkIterations_count = epochs;

	instance.lambda = lambda;


	settings.logToFile=true;

	MulticoreEngineExecutor<int, double> executor(instance, &settings);
	Solver<int, double> solver(executor);
	solver.runSolver();

	settings.logFile.close();
	return 0;
}

int main(int argc, char * argv[]) {

	oneProblem("data/a1a", "results/test1.log", 100,1,
			0.01);
	oneProblem("data/a1a", "results/test2.log", 100,8,
			0.01);
	oneProblem("data/a1a", "results/test3.log", 100,32,
			0.01);

	return 0;
}

