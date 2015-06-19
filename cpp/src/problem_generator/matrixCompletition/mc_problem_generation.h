/*
 * This file contains functions to generate random Matrix Completion problem
 */

#ifndef MC_PROBLEM_GENERATION_H_
#define MC_PROBLEM_GENERATION_H_

using namespace std;
#include <vector>

/**
 * Generate random MC sample, this function can be also used to sample from existing known matrix
 * and then just rewrite Z_values vector
 *
 * @param m - number of rows of matrix
 * @param n - number of columns of matrix
 * @param sparsity - each row will contains cca  sparsity*n nonzero elements
 * @param Z_values - COO values
 * @param Z_row_idx - COO row index
 * @param Z_col_idx - COO column index
 */

template<typename T, typename I>
void generate_random_matrix_completion_problem(int m, int n, float sparsity,
		std::vector<T> &Z_values, std::vector<I> &Z_row_idx,
		std::vector<I> &Z_col_idx) {

	long long nnzPerRow = (int) (n * sparsity);
	unsigned long long nnz = nnzPerRow * m;
	Z_values.resize(nnz);
	Z_row_idx.resize(nnz);
	Z_col_idx.resize(nnz);
	//Generate some data
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < nnzPerRow; j++) {
			bool stillNot = false;
			int newIDX = -1;
			while (!stillNot) {
				stillNot = true;
				newIDX = (int) (n * ((float) rand() / RAND_MAX));
				for (int k = 0; k < j; k++) {
					if (Z_col_idx[i * nnzPerRow + k] == newIDX)
						stillNot = false;
				}
			}

			Z_col_idx[i * nnzPerRow + j] = newIDX;
			Z_row_idx[i * nnzPerRow + j] = i;
			Z_values[i * nnzPerRow + j] = (float) rand() / RAND_MAX;
		}
	}
	printf("data DONE done\n");
}

template<typename T, typename I>
void generate_random_matrix_completion_problem_uniform_per_column(int m, int n,
		int nnz_per_column, std::vector<T> &Z_csc_values,
		std::vector<I> &Z_csc_row_idx, std::vector<I> &Z_col_ptr) {
	unsigned long long nnz = nnz_per_column * n;
	Z_csc_values.resize(nnz);
	Z_csc_row_idx.resize(nnz);
	Z_col_ptr.resize(n + 1);
	//Generate some data
	Z_col_ptr[0] = 0;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < nnz_per_column; j++) {
			bool stillNot = false;
			int newIDX = -1;
			while (!stillNot) {
				stillNot = true;
				newIDX = (int) (m * ((float) rand() / RAND_MAX));
				for (int k = 0; k < j; k++) {
					if (Z_csc_row_idx[i * nnz_per_column + k] == newIDX)
						stillNot = false;
				}
			}

			Z_csc_row_idx[i * nnz_per_column + j] = newIDX;
			Z_csc_values[i * nnz_per_column + j] = (float) rand() / RAND_MAX;
		}
		Z_col_ptr[i + 1] = nnz_per_column * i;
	}
	printf("data DONE done\n");
}

#endif /* MC_PROBLEM_GENERATION_H_ */
