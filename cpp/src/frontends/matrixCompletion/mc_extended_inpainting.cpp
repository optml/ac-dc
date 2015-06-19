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
#include "headers/c_libs_headers.h"
#include "helpers/data_loader.h"
#include "mc/mc_problem_generation.h"
#include "helpers/csv_writter.h"
//#include "helpers/matrix_conversions.h"
#include "mc/inpainting_problem_generator.h"

#include "solver/matrix_completion/parallel/parallel_mc_opemmp_extended.h"

//======================== Solvers


template<typename T, typename I>
void runExample() {
	/*  CONSTRAINS
	 ProblemData_instance.A_coo_operator[i]   =1   is constrain that M_i <= val_i
	 ProblemData_instance.A_coo_operator[i]   =0   is constrain that M_i = val_i
	 ProblemData_instance.A_coo_operator[i]   =-1   is constrain that M_i >= val_i
	 */

	// ======================GENERATION COO RADNOM DATA=======================================
	problem_mc_data<I, T> ProblemData_instance;

	//	const char* filename = "../../DATA/MC/Images/fingerprint.csv";
	//	const char* filename_sample = "../../DATA/MC/Images/fingerprint.sample.csv";
	//	const char* filename_out = "../../DATA/MC/Images/fingerprint.out.csv";
	const char* filename = "../../DATA/MC/Images/lenna.csv";
	const char* filename_sample = "../../DATA/MC/Images/lenna.sample.csv";
	const char* filename_out = "../../DATA/MC/Images/lenna.out.csv";
	//		const char* filename = "../../DATA/MC/Images/mri.csv";
	//		const char* filename_sample = "../../DATA/MC/Images/mri.sample.csv";
	//		const char* filename_out = "../../DATA/MC/Images/mri.out.csv";
	//	 m=256;n=256;

	ProblemData_instance.m = 512;
	ProblemData_instance.n = 512;

	std::vector<T> imageInRowFormat;

	T sparsity = 0.90;
	generate_random_mc_data_from_image(ProblemData_instance, imageInRowFormat, filename, filename_sample,
			sparsity);
	I m = ProblemData_instance.m;
	I n = ProblemData_instance.n;

	//=============MAGIC CONSTANSA================================
	OptimizationSettings settings; // Create setting for optimization procedure
	settings.totalThreads = 1;

	ProblemData_instance.mu = 1;
	int finalRank = 50;
	int lowRank = finalRank - 1;
	settings.total_execution_time = 30; //see solver/structures for more settings
	settings.iters_bulkIterations_count = .1 * (m + n);

	// ======================Configure Solver
	OptimizationStatistics statistics;

	settings.bulkIterations = false;
	settings.showIntermediateObjectiveValue = false;
	settings.showLastObjectiveValue = true;
	settings.showInitialObjectiveValue = true;

	settings.verbose = true;

	ProblemData_instance.rank = finalRank;

	std::vector<T> completed_matrix(m * n, 0);

	I max_rank = finalRank;
	float acuracy[max_rank];
	for (I rank = lowRank; rank < max_rank; rank++) {
		ProblemData_instance.rank = rank;
		run_matrix_completion_extended_openmp_solver(ProblemData_instance, settings, statistics);
		cout << "PROBLEM SOLVED\n";
		for (int row = 0; row < m; row++) {
			for (int col = 0; col < n; col++) {
				T tmp = 0;
				for (int i = 0; i < ProblemData_instance.rank; i++) {
					tmp += ProblemData_instance.L_mat[row * ProblemData_instance.rank + i]
							* ProblemData_instance.R_mat[col * ProblemData_instance.rank + i];
				}
				completed_matrix[row * m + col] = tmp;
			}
		}
		float totalError = 0;
		for (int i = 0; i < m * n; i++) {
			totalError += (completed_matrix[i] - imageInRowFormat[i]) * (completed_matrix[i] - imageInRowFormat[i]);
		}
		printf("Rank:%d Total Error: %f\n", rank, totalError);
		acuracy[rank] = totalError;
	}
	saveDataToCSVFile(m, n, filename_out, completed_matrix);
	for (int rank = lowRank; rank < max_rank; rank++) {
		printf("Rank:%d Total Error: %f\n", rank, acuracy[rank]);
	}

}

int main(void) {
	srand(1);
	cout << "Example for L2L1 - start \n";
	runExample<double, int> ();
	cout << "Example for L2L1 - finish \n";
	return 1;
}
