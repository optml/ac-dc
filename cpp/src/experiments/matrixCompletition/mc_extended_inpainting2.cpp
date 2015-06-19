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
 *    g++ -fopenmp  -O3 mc_extended_inpainting.cpp && ./a.out
 *
 */
using namespace std;
#include <vector>
#include <omp.h>
#include "../../helpers/c_libs_headers.h"
#include "../../utils/data_loader.h"
#include "../../solver/settingsAndStatistics.h"
#include "../../utils/csv_writter.h"
#include "../../solver/structures.h"

//#include "../../problem_generator/matrixCompletition/mc_problem_generation.h"
//#include "helpers/matrix_conversions.h"
#include "../../problem_generator/matrixCompletition/inpainting_problem_generator.h"

#include "../../solver/matrixCompletion/parallel/parallel_mc_opemmp.h"

//======================== Solvers

template<typename T, typename I>
void runExample(int finalRank, double finalMu, double totalTime, int p) {
	std::string prefix = "/data/takac/Jakub/images/test";
//	std::string prefix = "/work/jointProjects/2015_MACO/matlab2/Jakub/images/test";

	problem_mc_data<I, T> ProblemData_instance;
	std:: stringstream ss;
	std:: stringstream ss2;
	std:: stringstream ss3;
	std:: stringstream ss4;
	ss << prefix << p << ".csv";
	std::string filename = ss.str();
	ss2 << prefix << p << ".sample.csv";
	const char* filename_sample = ss2.str().c_str();
	ss3 << prefix << p << ".out.csv";

	std::string filename_out = ss3.str().c_str();

	ProblemData_instance.m = 512;
	ProblemData_instance.n = 512;

	std::vector<T> imageInRowFormat;

	T sparsity = 0.50;
	generate_random_mc_data_from_image(ProblemData_instance, imageInRowFormat,
			filename, filename_sample, sparsity);
	I m = ProblemData_instance.m;
	I n = ProblemData_instance.n;
	//=============MAGIC CONSTANSA================================
	OptimizationSettings settings; // Create setting for optimization procedure
	settings.totalThreads = 1;
	ProblemData_instance.mu = finalMu;

	int lowRank = finalRank - 1;
	settings.total_execution_time = totalTime; //see solver/structures for more settings
	settings.iters_bulkIterations_count = 0.1 * (m + n);

	// ======================Configure Solver
	OptimizationStatistics statistics;

	settings.bulkIterations = false;
	settings.showIntermediateObjectiveValue = false;
	settings.showLastObjectiveValue = true;
	settings.showInitialObjectiveValue = true;

	settings.verbose = true;

	ProblemData_instance.rank = finalRank;

	std::vector < T > completed_matrix(m * n, 0);

	I max_rank = finalRank;
	float acuracy[max_rank];
	for (I rank = lowRank; rank < max_rank; rank++) {
		ProblemData_instance.rank = rank;

		ofstream myfile;
		ss4.str("");
		ss4 << prefix << p << ".log.csv_" << finalMu;
		myfile.open(ss4.str().c_str());
		run_matrix_completion_openmp_solver(ProblemData_instance,
				ProblemData_instance, settings, statistics, 1, 0.0, myfile);
		myfile.close();

		cout << "PROBLEM SOLVED\n";
		for (int row = 0; row < m; row++) {
			for (int col = 0; col < n; col++) {
				T tmp = 0;
				for (int i = 0; i < ProblemData_instance.rank; i++) {
					tmp += ProblemData_instance.L_mat[row
							* ProblemData_instance.rank + i]
							* ProblemData_instance.R_mat[col
									* ProblemData_instance.rank + i];
				}
				completed_matrix[row * m + col] = tmp;
			}
		}
		float totalError = 0;
		for (int i = 0; i < m * n; i++) {
			totalError += (completed_matrix[i] - imageInRowFormat[i])
					* (completed_matrix[i] - imageInRowFormat[i]);
		}
		printf("Rank:%d Total Error: %f\n", rank, totalError);
		acuracy[rank] = totalError;
	}
	saveDataToCSVFile(m, n, filename_out, completed_matrix);
	for (int rank = lowRank; rank < max_rank; rank++) {
		printf("Rank:%d Total Error: %f\n", rank, acuracy[rank]);
	}

}

int main(int argc, char *argv[]) {
	srand(1);
	cout << "Example for L2L1 - start \n";

	int finalRank = 1;
	double finalMu = 1;
	double totalTime = 1;
	int p = 1;
	char c;
	/*
	 * r - result file
	 */

	while ((c = getopt(argc, argv, "r:m:t:p:")) != -1) {
		switch (c) {

		case 'r':
			finalRank = atoi(optarg);
			break;
		case 'm':
			finalMu = atof(optarg);
			break;

		case 't':
			totalTime = atof(optarg);
			break;
		case 'p':
			p = atoi(optarg);
			break;
		}
	}

	runExample<double, int>(finalRank, finalMu, totalTime, p);
	cout << "Example for L2L1 - finish \n";

	return 0;
}
