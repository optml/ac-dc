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
#include <vector>
#include "headers/c_libs_headers.h"
#include "helpers/data_loader.h"
#include "mc/mc_problem_generation.h"
#include "helpers/csv_writter.h"
//#include "helpers/matrix_conversions.h"

#include "solver/matrix_completion/parallel/parallel_mc_opemmp.h"

//======================== Solvers
#include <omp.h>

template<typename T, typename I>
void runExample(float executionTime, float lambda) {
	// ======================GENERATION COO RADNOM DATA=======================================
	problem_mc_data<I, T> ProblemData_instance;
	I m;
	I n;

	m = 512;
	n = 512;
	std::vector<float> imageInRowFormat;
//	const char* filename = "../../DATA/MC/Images/fingerprint.csv";
//	const char* filename_sample = "../../DATA/MC/Images/fingerprint.sample.csv";
//	const char* filename_out = "../../DATA/MC/Images/fingerprint.out.csv";

//		const char* filename = "../../DATA/MC/Images/lenna.csv";
//		const char* filename_sample = "../../DATA/MC/Images/lenna.sample.csv";
//		const char* filename_out = "../../DATA/MC/Images/lenna.out.csv";


		const char* filename = "../../DATA/MC/Images/mri.csv";
		const char* filename_sample = "../../DATA/MC/Images/mri.sample.csv";
		const char* filename_out = "../../DATA/MC/Images/mri.out.csv";
	 m=256;n=256;


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
		ProblemData_instance.A_coo_values[i]
				= imageInRowFormat[ProblemData_instance.A_coo_row_idx[i] * n
						+ ProblemData_instance.A_coo_col_idx[i]];
	}
	cout << "DATA SAMPLED\n";

	std::vector<float> initialSampledData(n * m, -1);
	for (int i = 0; i < ProblemData_instance.A_coo_row_idx.size(); i++) {
		initialSampledData[ProblemData_instance.A_coo_row_idx[i] * n
				+ ProblemData_instance.A_coo_col_idx[i]]
				= ProblemData_instance.A_coo_values[i];
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

	/*

	 cout << "Dimension of your problem is " << m << " x  " << n << " \n";
	 cout << "Number of nonzero elements of matrix A: "
	 << ProblemData_instance.A_coo_values.size() << "\n";

	 printf("Computation ends\n");
	 T prediction = 0;
	 long long total_test_points = 0;
	 long long total_correct_points = 0;
	 for (I j = 0; j < test_data.A_coo_row_idx.size(); j++) {
	 total_test_points++;
	 I row = test_data.A_coo_row_idx[j];
	 I col = test_data.A_coo_col_idx[j];
	 prediction = 0;
	 for (int i = 0; i < ProblemData_instance.rank; i++) {
	 prediction += ProblemData_instance.L_mat[row
	 * ProblemData_instance.rank + i]
	 * ProblemData_instance.R_mat[col
	 * ProblemData_instance.rank + i];
	 }
	 if (abs(test_data.A_coo_values[j] - prediction) < 0.5) {
	 total_correct_points++;
	 }

	 //		if (total_test_points < 10)
	 //			printf("D[%d,%d]=%f  p:%f\n", test_data.A_coo_col_idx[j],
	 //					test_data.A_coo_row_idx[j], test_data.A_coo_values[j],
	 //					prediction);
	 }
	 printf("T%d: Total points: %d, total successfull %d,  accuracy:%f\n",
	 settings.max_totalThreads, total_test_points,
	 test_data.A_coo_row_idx.size(), total_correct_points,
	 (float) total_correct_points / total_test_points);

	 T train_prediction = 0;
	 long long train_total_test_points = 0;
	 long long train_total_correct_points = 0;
	 for (I j = 0; j < ProblemData_instance.A_coo_row_idx.size(); j++) {
	 train_total_test_points++;
	 I row = ProblemData_instance.A_coo_row_idx[j];
	 I col = ProblemData_instance.A_coo_col_idx[j];
	 train_prediction = 0;
	 for (int i = 0; i < ProblemData_instance.rank; i++) {
	 train_prediction += ProblemData_instance.L_mat[row
	 * ProblemData_instance.rank + i]
	 * ProblemData_instance.R_mat[col
	 * ProblemData_instance.rank + i];
	 }
	 if (abs(ProblemData_instance.A_coo_values[j] - prediction) < 0.5) {
	 train_total_correct_points++;
	 }

	 //		if (train_total_test_points < 10)
	 //			printf("train: D[%d,%d]=%f  p:%f\n", ProblemData_instance.A_coo_col_idx[j],
	 //					ProblemData_instance.A_coo_row_idx[j], ProblemData_instance.A_coo_values[j],
	 //					train_prediction);
	 }

	 printf("Tr:%d:Total points:%d, total successfull %d,  accuracy:%f\n",
	 settings.max_totalThreads, train_total_test_points,
	 train_total_correct_points, (float) train_total_correct_points
	 / train_total_test_points);

	 printf("%d:Average speed:%f it/ms \n", settings.max_totalThreads,
	 statistics.average_speed_iters_per_ms);
	 printf("%d:Total execution time:%f ms \n", settings.max_totalThreads,
	 statistics.true_computation_time);
	 printf("%d:Total Iterations %f M\n", settings.max_totalThreads,
	 statistics.number_of_iters_in_millions);
	 //	}
	 //	}
	 // ======================Solve problem using Serial Random Coordinate Descent Method=========
	 cout << "Solve problem using Serial Random Coordinate Descent Method\n";
	 std::vector<float> h_x_result; // Vector where result will be stored


	 settings.use_shrinking = false;
	 settings.shrinking_probability = 0.9;
	 settings.shrinking_starting_iter = 3 * n;

	 settings.threshold = 1e-5;
	 settings.regulatization_path_length = 0;

	 settings.iters_bulkIterations_count = 10;
	 int numthreads = 2;
	 float averageSpeed[numthreads + 1];

	 ProblemData_instance.m = m;
	 ProblemData_instance.n = n;

	 ProblemData_instance.lambda = .01;
	 ProblemData_instance.sigma = 1;
	 ProblemData_instance.x.resize(n, 0);

	 for (int num_of_threads = 1; num_of_threads <= numthreads; num_of_threads++) {
	 settings.max_totalThreads = num_of_threads;

	 averageSpeed[num_of_threads]
	 = static_statsistics.average_speed_iters_per_ms;
	 }
	 for (int num_of_threads = 1; num_of_threads <= numthreads; num_of_threads++) {
	 printf("Thread %d, speed %f \n", num_of_threads,
	 averageSpeed[num_of_threads]);
	 }

	 */

}

int main(void) {
	srand(1);
	cout << "Example for L2L1 - start \n";
	runExample<double, int> (.10, .1);
	cout << "Example for L2L1 - finish \n";
	return 1;
}
