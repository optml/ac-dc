#include "../../helpers/c_libs_headers.h"
#include "../../helpers/utils.h"
#include "../../helpers/gsl_random_helper.h"
#include "../../helpers/matrix_conversions.h"
#include "../../solver/matrixCompletion/parallel/parallel_mc_opemmp.h"
using namespace std;

template<typename T, typename I>
void runMatrixCompletitionExperiment(T executionTime, T lambda) {
	// ======================GENERATION COO RADNOM DATA=======================================
	problem_mc_data<I, T> ProblemData_instance;
	I m = 512;
	I n = 512;
	std::vector<float> imageInRowFormat;
//	const char* filename = "../../DATA/MC/Images/fingerprint.csv";
//	const char* filename_sample = "../../DATA/MC/Images/fingerprint.sample.csv";
//	const char* filename_out = "../../DATA/MC/Images/fingerprint.out.csv";

	const char* filename = "../data/matrixCompletition/images/lenna.csv";
	const char* filename_sample = "../data/matrixCompletition/images/lenna.sample.csv";
	const char* filename_out = "../data/matrixCompletition/images/lenna.out.csv";

//		const char* filename = "../../DATA/MC/Images/mri.csv";
//		const char* filename_sample = "../../DATA/MC/Images/mri.sample.csv";
//		const char* filename_out = "../../DATA/MC/Images/mri.out.csv";
//	m = 256;
//	n = 256;

	OptimizationSettings settings; // Create setting for optimization procedure
	int finalRank = 256;
	int lowRank = 255;
	executionTime = 100;
	ProblemData_instance.mu = 0.5;
	float sparsity = 0.50;
	settings.total_execution_time = executionTime; //see solver/structures for more settings
	settings.iters_bulkIterations_count = 1 * (m + n);
	generate_random_matrix_completion_problem(m, n, sparsity,
			ProblemData_instance.A_coo_values,
			ProblemData_instance.A_coo_row_idx,
			ProblemData_instance.A_coo_col_idx);

	cout << "START READING DATA FROM FILE\n";
	loadDataFromCSVFile(m, n, filename, imageInRowFormat);

	cout << "DATA LOADED TOTAL LENGTH "
			<< ProblemData_instance.A_coo_values.size() << "\n";
	for (int i = 0; i < ProblemData_instance.A_coo_values.size(); i++) {
		ProblemData_instance.A_coo_values[i] =
				imageInRowFormat[ProblemData_instance.A_coo_row_idx[i] * n
						+ ProblemData_instance.A_coo_col_idx[i]];
	}
	cout << "DATA SAMPLED\n";

	std::vector<float> initialSampledData(n * m, -1);
	for (int i = 0; i < ProblemData_instance.A_coo_row_idx.size(); i++) {
		initialSampledData[ProblemData_instance.A_coo_row_idx[i] * n
				+ ProblemData_instance.A_coo_col_idx[i]] =
				ProblemData_instance.A_coo_values[i];
	}
	saveDataToCSVFile(m, n, filename_sample, initialSampledData);

	// ======================Configure Solver
	OptimizationStatistics statistics;

	settings.bulkIterations = false;
	settings.showIntermediateObjectiveValue = false;
	settings.showLastObjectiveValue = true;
	settings.showInitialObjectiveValue = true;

	settings.verbose = true;

	ProblemData_instance.rank = finalRank;

	ProblemData_instance.m = m;
	ProblemData_instance.n = n;
	std::vector<float> completed_matrix(m * n, 0);
	I threads = 1;
	settings.totalThreads = threads;
	I max_rank = finalRank;
	float acuracy[max_rank];
	for (I rank = lowRank; rank < max_rank; rank++) {
		ProblemData_instance.rank = rank;
		//	for (int i = 200; i <= 20000; i = i * 10) {
		printf("check %d\n", ProblemData_instance.n);
		run_matrix_completion_openmp_solver(ProblemData_instance, settings,
				statistics);
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

int main(int argc, char * argv[]) {
	cout << "Experiment starts...." << endl;


	runMatrixCompletitionExperiment<float,int>(1., 0.1);

	cout << "Experiment end...." << endl;
}
