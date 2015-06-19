/*
 * This file contains an experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */

#include "../helpers/c_libs_headers.h"
#include "../utils/randomNumbersUtil.h"
#include "../utils/largeScaleExperiment.h"
#include "../problem_generator/generator_nesterov.h"

int main(int argc, char * argv[]) {
	const int MAXIMUM_THREADS = 24;
	std::vector<gsl_rng *> rs = randomNumberUtil::inittializeRandomSeeds(
			MAXIMUM_THREADS);
	//---------------------- Set output files
	ofstream histogramLogFile;
	histogramLogFile.open("results/large_scale_expeiment_histogram.log");
	ofstream experimentLogFile;
	experimentLogFile.open("results/large_scale_expeiment.log");
	//---------------------- Setting parameters of experiment
	long long n = 2000000000; // this value was used for experiments
	n = 100000; //we choose smaller value for you PC -  comment this line if you want reproduce our experiment
	long long m = n * 2;
	long long p = 30;

	//small fake to run it on local machine
	n = 1000;
	m = 10000;
	p = 20;

	ProblemData<long long, double> inst;
	randomNumberUtil::init_random_seeds(rs);
	double fvalOpt = nesterov_generator(inst, n, m, p, rs, histogramLogFile);
	inst.lambda = 1;
	//--------------------- run experiment - one can change precission here
	largeScaleExperiment::run_experiment<long long, double, square_loss_traits>(
			inst, 62,   1,rs, experimentLogFile, MAXIMUM_THREADS, fvalOpt);
	histogramLogFile.close();
	experimentLogFile.close();
	return 0;
}
