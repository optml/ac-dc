/*
 * Example of calling Random L2-L1 solver for Synthetic problem
 *
 * This can be compiled with g++ compiler
 *
 * We will compare the Serial Random and Shrinking Strategy for Serial Random algorithm
 *
 * PS: If one introduce shrinking too early or with high probability, the result can be worse.
 *
 *   ulimit -s unlimited
 *   g++ -O3    -fopenmp mc_netflix.cpp
 *
 */
using namespace std;

#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include "../../helpers/c_libs_headers.h"
#include "../../utils/data_loader.h"
//#include "mc/mc_problem_generation.h"
//#include "helpers/matrix_conversions.h"

#include "../../solver/matrixCompletion/parallel/parallel_mc_opemmp.h"

//======================== Solvers
#include <omp.h>

#include <iostream>
#include <fstream>
#include <sstream>

template<typename T, typename I>
void runExample(std::string &dataFile, std::string &testFile, bool base,
		float executionTime, float ex_mu, long ex_bulk, long ex_total,
		int ex_rank, int TT, T DELTA) {

	std::stringstream ss("");
	ss << dataFile << "_" << "result_" << ex_bulk <<"_"<<ex_total<< "_" << ex_rank << "_"<< ex_mu
			<< "_" << TT << "_" << DELTA << ".log";

	ofstream myfile(ss.str().c_str());
	if (!myfile.is_open()) {
		return;
	}

	// ======================GENERATION COO RADNOM DATA=======================================
	problem_mc_data<I, T> ProblemData_instance;
	I m;
	I n;

	int status = load_matrix_in_coo_format(dataFile.c_str(),
			ProblemData_instance.A_coo_values, //
			ProblemData_instance.A_coo_row_idx, //
			ProblemData_instance.A_coo_col_idx, //
			m, n, base);

	if (!status) {
		cout << "File " << dataFile << " not found" << endl;
		return;
	}
	cout << "Dimension of your problem is " << m << " x  " << n << " \n";
	cout << "Number of nonzero elements of matrix A: "
			<< ProblemData_instance.A_coo_values.size() << "\n";
	// ======================Configure Solver
	OptimizationStatistics statistics;
	OptimizationSettings settings; // Create setting for optimization procedure
	settings.total_execution_time = executionTime; //see solver/structures for more settings
	settings.bulkIterations = false;
	settings.showIntermediateObjectiveValue = false;
	settings.showLastObjectiveValue = true;
	settings.showInitialObjectiveValue = true;

	settings.verbose = true;

	settings.iters_bulkIterations_count = 1;

	ProblemData_instance.m = m;
	ProblemData_instance.n = n;
	I m_test;
	I n_test;
	problem_mc_data<I, T> test_data;
	status = load_matrix_in_coo_format(testFile.c_str(), test_data.A_coo_values, //
			test_data.A_coo_row_idx, //
			test_data.A_coo_col_idx, //
			m_test, n_test, base);
	if (!status) {

		cout << "File " << testFile << " not found" << endl;
		return;
	}

	I threads = TT;
	//	for (I threads = 1; threads <= 20; threads++) {
	settings.totalThreads = threads;

	//	for (int i = 200; i <= 20000; i = i * 10) {
	ProblemData_instance.mu = ex_mu;
	settings.iters_bulkIterations_count = ex_bulk;
	settings.iters_communicate_count = ex_total;
	ProblemData_instance.rank = ex_rank;
	settings.verbose = false;

	run_matrix_completion_openmp_solver(ProblemData_instance, test_data,
			settings, statistics, TT, DELTA,myfile);

	printf("Computation ends\n");

	printf("%d:Average speed:%f it/ms \n", settings.totalThreads,
			statistics.average_speed_iters_per_ms);
	printf("%d:Total execution time:%f ms \n", settings.totalThreads,
			statistics.time_wallclock);
	printf("%d:Total Iterations %f M\n", settings.totalThreads,
			statistics.number_of_iters_in_millions);
 		    myfile.close();

}

int main(int argc, char *argv[]) {

	char c;
	/*
	 * r - result file
	 */
	std::string dataFile;
	std::string testFile;
	bool base = 0;
	double delta = 0.;
	int IT = 20;
	int rank = 10;
	double mu=0.001;
	int threads = 1;
	while ((c = getopt(argc, argv, "A:t:T:d:I:r:m:h:")) != -1) {
		switch (c) {

		case 'A':
			dataFile = optarg;
			break;
		case 'T':
			testFile = optarg;
			break;
		case 't':
			base = atoi(optarg);
			break;
		case 'd':
			delta = atof(optarg);
			break;

		case 'I':
			IT = atoi(optarg);
			break;

		case 'r':
			rank = atoi(optarg);
			break;
		case 'm':
			mu = atof(optarg);
			cout << "MU "<<mu<<endl;
			break;
		case 'h':
			threads = atoi(optarg);
			break;
		}
	}

	srand(1);
	cout << "MC - start \n";
	runExample<double, unsigned long>(dataFile, testFile, base, 10000000000,
			mu, 2, IT, rank, threads, delta);

	cout << "MC - finish \n";
	return 0;
}
