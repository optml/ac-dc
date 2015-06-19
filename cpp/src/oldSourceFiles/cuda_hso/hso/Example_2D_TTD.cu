/*
 * Example of calling Random L2-L1 solver for 2D TTD problem
 */
#define COLUMNLENGTH 4
/* columLength - length of each column. We requite each column to have the same length
 *               one can fill them with zero values, but the input data has to fulfill this
 *               requirement!
 */
#include "headers.h" // All neccessary libs are included
using namespace std; // namespace for std;

void runThrust2DExample(int row, int col, int exampleType, float executionTime,
		float lambda) {
	// ======================TTD 2D data Generation=======================================
	thrust::host_vector<float> h_A_values; // Matrix A values - sparse representation
	thrust::host_vector<int> h_A_row_index; // Matrix A row idx - sparse representation
	thrust::host_vector<int> h_A_col_index; // Matrix A col idx - sparse representation
	thrust::host_vector<int> h_fixedPoints; // Vector of fixed nodes
	thrust::host_vector<int> model_node_description; //Description of nodes
	getThrust2DFixedPointsVector(row, col, exampleType, &h_fixedPoints); //get fixed points
	cout << "Grid size is: " << row << " x " << col << "\n";
	cout << "Number of all points: " << row * col << "\n";
	cout << "Number of fixes points: " << h_fixedPoints.size() << "\n";
	int m = 0;
	int n = 0;
	getThrust2DProblemTTD(row, col, exampleType, &m, &n, &h_A_values,
			&h_A_col_index, &h_A_row_index, h_fixedPoints,
			&model_node_description, 5, 500000000);
	cout << "Dimension of your problem is " << m << " x  " << n << " \n";
	cout << "Number of nonzero elements of matrix A: " << h_A_values.size()
			<< "\n";
	thrust::host_vector<float> h_force(m);
	getThrus2DForceVector(&h_force, m, row, col, exampleType); // Get force vector

	// ======================Configure Solver
	optimization_setting settings; // Create setting for optimization procedure
	settings.total_execution_time = executionTime; //see solver/structures for more settings
	// ======================Solve problem using Parallel Random Coordinate Descent Method=========
	cout << "Solve problem using Parallel Random Coordinate Descent Method\n";
	thrust::host_vector<float> h_x_result_parallel_random; // Vector where result will be stored
	runParallelRandomL2L1Solver(n, m, h_A_values, h_A_row_index, h_A_col_index,
			h_force, lambda, settings, &h_x_result_parallel_random);
	// ======================Solve problem using Serial Random Coordinate Descent Method=========
	cout << "Solve problem using Serial Random Coordinate Descent Method\n";
	thrust::host_vector<float> h_x_result_serial_random; // Vector where result will be stored
	runSerialRandomL2L1Solver(n, m, h_A_values, h_A_row_index, h_A_col_index,
			h_force, lambda, settings, &h_x_result_serial_random);
	// ======================Solve problem using Parallel Greedy Coordinate Descent Method=========
	cout << "Solve problem using Parallel Greedy Coordinate Descent Method \n";
	thrust::host_vector<float> h_x_result_parallel_greedy; // Vector where result will be stored
	settings.device_total_threads_per_block = 512;
	settings.number_of_inner_iterations_per_kernel_run = 1000;
	runParallelGreedyL2L1Solver(n, m, h_A_values, h_A_row_index, h_A_col_index,
			h_force, lambda, settings, &h_x_result_parallel_greedy);
	// ======================Solve problem using Serial Greedy Coordinate Descent Method=========
	cout << "Solve problem using Serial Greedy Coordinate Descent Method \n";
	thrust::host_vector<float> h_x_result_serial_greedy; // Vector where result will be stored
	settings.number_of_inner_iterations_per_kernel_run = 100;
	runSerialGreedyL2L1Solver(n, m, h_A_values, h_A_row_index, h_A_col_index,
			h_force, lambda, settings, &h_x_result_serial_greedy);
	// ======================Save result into file=======================================
	saveSolutionIntoFile("/tmp/ttd_output.csv", //FILENAME where the result will be stored
			&h_x_result_serial_greedy[0], // Pointer to vector of result
			n, //Length of solution
			&model_node_description[0], // Pointer to node model description
			0// Tresh Hold - only bars with absolut value bigger than this will be stored
	);

}

