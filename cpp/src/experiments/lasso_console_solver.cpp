/*
 * This file contains an experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */

#include "../helpers/c_libs_headers.h"
#include "../helpers/gsl_random_helper.h"

#include "../solver/structures.h"
#include "../class/loss/losses.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

template<typename L, typename D>
void run_lasso_computation(ProblemData<L, D> &inst, std::vector<D> &h_Li,
		int omp_threads, D sigma, int N, int blockReduction,
		std::vector<gsl_rng *>& rs, ofstream& experimentLogFile,int multiply=100) {
	omp_set_num_threads(omp_threads);
	init_random_seeds(rs);
	L n = inst.n;
	L m = inst.m;
	std::vector<D> residuals(m);
	Losses<L, D, square_loss_traits>::bulkIterations(inst, residuals);
	D fvalInit = Losses<L, D, square_loss_traits>::compute_fast_objective(inst,
			residuals);
	double totalRunningTime = 0;
	double iterations = 0;
	L perPartIterations =multiply* n / blockReduction;
	double additional = perPartIterations / (0.0 + n);
	D fval = fvalInit;
// store initial objective value
	experimentLogFile << setprecision(16) << omp_threads << "," << n << "," << m
			<< "," << sigma << "," << totalRunningTime << "," << iterations
			<< "," << fval << endl;
	//iterate
	for (int totalIt = 0; totalIt < N; totalIt++) {
		double startTime = gettime_();
#pragma omp parallel for
		for (L it = 0; it < perPartIterations; it++) {
			// one step of the algorithm
			unsigned long int idx = gsl_rng_uniform_int(gsl_rng_r, n);
			Losses<L, D, square_loss_traits>::do_single_iteration_parallel(inst,
					idx, residuals, inst.x, h_Li);
		}
		double endTime = gettime_();
		iterations += additional;
		totalRunningTime += endTime - startTime;
		// recompute residuals  - this step is not necessary but if accumulation of rounding errors occurs it is useful
		if (totalIt % 3 == 0) {
			Losses<L, D, square_loss_traits>::bulkIterations(inst,
					residuals);
		}
		fval = Losses<L, D, square_loss_traits>::compute_fast_objective(inst,
				residuals);
		int nnz = 0;
#pragma omp parallel for reduction(+:nnz)
		for (L i = 0; i < n; i++)
			if (inst.x[i] != 0)
				nnz++;
		cout << omp_threads << "," << n << "," << m << "," << sigma << ","
				<< totalRunningTime << "," << iterations << "," << setprecision(15)<<fval << ","
				<< nnz << endl;

		experimentLogFile << setprecision(16) << omp_threads << "," << n << ","
				<< m << "," << sigma << "," << totalRunningTime << ","
				<< iterations << "," << setprecision(15)<<fval << "," << nnz << endl;
	}
}

#include "../utils/file_reader.h"
template<typename L, typename D>
void loadData(ProblemData<L, D> &inst, std::vector<D> &h_Li, char * dataMatrix,
		char * bVector) {

	unsigned int m;
	unsigned int n;
	unsigned int ld;
	std::vector<D> matrix;
	InputOuputHelper::readCSVFile(matrix, ld, m, n, dataMatrix);
	cout << "Loaded data with m=" << m << " and n=" << n << endl;
	inst.m = m;
	inst.n = n;
	int nnz = 0;
	inst.A_csc_col_ptr.resize(1);
	inst.A_csc_col_ptr[0] = nnz;

	inst.A_csc_row_idx.resize(0);
	inst.A_csc_values.resize(0);
	h_Li.resize(n);
	D val;
	for (int i = 0; i < n; i++) {
		h_Li[i] = 0;
		for (int j = 0; j < m; j++) {
			val = matrix[i * m + j];
			if (val != 0) {
				inst.A_csc_row_idx.push_back(j);
				inst.A_csc_values.push_back(val);
				nnz++;
				h_Li[i] += val * val;
			}
		}
		if (h_Li[i] != 0) {
			h_Li[i] = 1 / h_Li[i];
		}
		inst.A_csc_col_ptr.push_back(nnz);
	}
	InputOuputHelper::readCSVFile(inst.b, ld, m, n, bVector);
}

int main(int argc, char * argv[]) {
// setup GSL random generators
	gsl_rng_env_setup();
	const gsl_rng_type * T;
	T = gsl_rng_default;
	const int MAXIMUM_THREADS = 32;
	std::vector<gsl_rng *> rs(MAXIMUM_THREADS);
	for (int i = 0; i < MAXIMUM_THREADS; i++) {
		rs[i] = gsl_rng_alloc(T);
		gsl_rng_set(rs[i], i);
	}
	init_omp_random_seeds();

	ProblemData<int, float> inst;
	std::vector<float> h_Li;
	char c;
	char* inputFilePath;
	char * bVector;
	while ((c = getopt(argc, argv, "A:b:")) != -1) {
		switch (c) {
		break;
	case 'A':
		inputFilePath = optarg;
		break;
	case 'b':
		bVector = optarg;
		break;
		}
	}

	loadData(inst, h_Li, inputFilePath, bVector);
//---------------------- Set output files
	ofstream experimentLogFile;
	experimentLogFile.open("results/lasso.log");
//--------------------- run experiment - one can change precission here
	float lambdaHigh = 100;
	float lambdaLow = 0.00001;
	float sigma = 1.0;
	int pieces = 20;
	inst.x.resize(inst.n);
	for (int i = 0; i < inst.n; i++)
		inst.x[i] = 0;

	float delta = pow(lambdaLow / lambdaHigh, 1 / (pieces + 0.0));
	for (float lambda = lambdaHigh; lambda >= lambdaLow;
			lambda = lambda * delta) {
		cout << "lambda " << lambda << endl;
		inst.lambda = lambda;
		run_lasso_computation(inst, h_Li, 4, sigma, 20, 1, rs,
				experimentLogFile,100);
		ofstream resultFile;
		stringstream ss;
		ss << "results/lasso_x_" << lambda << ".log";
		resultFile.open(ss.str().c_str());
		for (int i=0;i<inst.n;i++){
			resultFile<<setprecision(15)<<inst.x[i]<<endl;
		}
		resultFile.close();
	}
	experimentLogFile.close();
	return 0;
}
