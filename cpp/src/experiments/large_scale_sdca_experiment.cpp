/*
 * This file contains an experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */

#include "../helpers/c_libs_headers.h"
#include "../utils/randomNumbersUtil.h"
#include "../utils/largeScaleExperiment.h"
#include "../utils/distributed_instances_loader.h"
const int MAXIMUM_THREADS = 64;
int oneProblem(std::vector<gsl_rng *> &rs, string dataFile, string logFile,int EPOCHS, int PARTITION) {

	cout << "experiments starts" << endl;

	//---------------------- Set output files
	ofstream experimentLogFile;
	experimentLogFile.open(logFile.c_str());
	//---------------------- Setting parameters of experiment
	ProblemData<long long, double> inst;
	randomNumberUtil::init_random_seeds(rs);
	double fvalOpt = 0;

	loadDistributedSparseSVMRowData(dataFile, -1, -1, inst, false);


	inst.lambda = 1 / (0.0 + inst.n);
	inst.lambda=inst.lambda*1000;
	inst.lambda =1;

	inst.omega = 2;

	inst.total_n = inst.n;
	//--------------------- run experiment - one can change precission here
	largeScaleExperiment::run_experiment<long long, double, loss_hinge_dual>(
			inst, EPOCHS, PARTITION, rs, experimentLogFile, MAXIMUM_THREADS, fvalOpt);
	experimentLogFile.close();
	return 0;
}

int main(int argc, char * argv[]) {

	std::vector<gsl_rng *> rs = randomNumberUtil::inittializeRandomSeeds(
			MAXIMUM_THREADS);

	oneProblem(rs,"data/a1a","results/large_scale_sdca_expeiment_kddb.log",30,1);



	return 0;
}

