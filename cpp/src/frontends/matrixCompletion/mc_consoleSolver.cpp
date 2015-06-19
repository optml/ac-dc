/*
 * Example of calling Random L2-L1 solver for Synthetic problem
 *
 * This can be compiled with g++ compiler
 *
 * We will compare the Serial Random and Shrinking Strategy for Serial Random algorithm
 *
 * PS: If one introduce shrinking too early or with high probability, the result can be worse.
 *
 *
 *   g++ -O3 -g  -D_DEBUG  -lgslcblas    -fopenmp mc_consoleSolver.cpp
 *
 */
using namespace std;
#include <vector>
#include "headers/c_libs_headers.h"
#include "mc/mc_problem_generation.h"
//#include "helpers/matrix_conversions.h"

#include "solver/matrix_completion/parallel/parallel_mc_opemmp.h"

//======================== Solvers
#include <omp.h>

template<typename T, typename I>
void runExample(int m, int n, float sparsity, float executionTime, float lambda) {
	// ======================GENERATION COO RADNOM DATA=======================================
	problem_mc_data<I, T> ProblemData_instance;

	generate_random_matrix_completion_problem(m, n, sparsity,
			ProblemData_instance.A_coo_values,
			ProblemData_instance.A_coo_row_idx,
			ProblemData_instance.A_coo_col_idx);

	cout << "Dimension of your problem is " << m << " x  " << n << " \n";
	cout << "Number of nonzero elements of matrix A: "
			<< ProblemData_instance.A_coo_values.size() << "\n";
	// ======================Configure Solver
	OptimizationSettings settings; // Create setting for optimization procedure
	settings.total_execution_time = executionTime; //see solver/structures for more settings
	settings.bulkIterations = false;
	settings.showIntermediateObjectiveValue = false;
	settings.showLastObjectiveValue = true;
	settings.showInitialObjectiveValue = true;

	settings.iters_bulkIterations_count=1000;

	OptimizationStatistics statistics;

	ProblemData_instance.rank=10;
	ProblemData_instance.mu=0.1;
	ProblemData_instance.m=m;
	ProblemData_instance.n=n;

	run_matrix_completion_openmp_solver(ProblemData_instance, settings,
			statistics);

	/*
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
	runExample<double, int> (100, 100, 0.1, .10, .1);
	cout << "Example for L2L1 - finish \n";
	return 1;
}
