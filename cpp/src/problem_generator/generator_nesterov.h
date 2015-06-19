/*
 * This file contains an functions for experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */

#ifndef GENERATOR_NESTEROV_H_
#define GENERATOR_NESTEROV_H_

#include "../helpers/c_libs_headers.h"
#include "../class/loss/losses.h"
#include "../parallel/parallel_essentials.h"
#include "../helpers/openmp_helper.h"
#include "../helpers/gsl_random_helper.h"
#include "../solver/structures.h"

/*
 *  This function generate a random LASSO problem with known optimal value (is returned). The input is
 *  n - number of columns
 *  m - number of rows
 *  p - number of nonzero elements in each row
 */
template<typename D, typename L>
double nesterov_generator(ProblemData<L, D> &inst, L n, L m, L p,
		std::vector<gsl_rng *>& rs, ofstream& histrogramLogFile) {
	bool sc = true;
	if (n>m) sc=false;
	double lambda = 1;
	double rho = 10000;
	rho=0.01;
	int n_nonzero = 100000;
	if (n_nonzero > n)
		n_nonzero = n / 100;
	n_nonzero = n / 2;



	double sqrtofnonzeros = sqrt(n_nonzero + 0.0);
	L nn = n;
	L nnz = nn * p;
	inst.lambda = lambda;
	inst.m = m;
	inst.n = n;
	std::vector<D> dataToSort(n);
	inst.A_csc_col_ptr.resize(n + 1, 0);
	inst.A_csc_row_idx.resize(nnz, 0);
	inst.A_csc_values.resize(nnz, 0);
	inst.b.resize(m, 0);
	L i;
	randomNumberUtil::init_random_seeds(rs);
	L IDX_h[TOTAL_THREADS][p]; // host A-index matrix pointers
	double optimalvalue = 0;
	inst.x.resize(n, 0);
	double tmp;
	nnz = 0;
#pragma omp parallel for schedule(static,1)
	for (i = 0; i < n; i++) {
		inst.A_csc_col_ptr[i] = i * p;
		L idx = 0;
		for (L j = 0; j < p; j++) {
			int notfinished = 1;
			double val = (D) (rand_r(&myseed) / (RAND_MAX + 1.0));
			while (notfinished) {
				notfinished = 0;
				idx = gsl_rng_uniform_int(gsl_rng_r, m);
				if (j == 0 && sc) {
					idx = i;
					val = p;
				}
				for (L k = 0; k < j; k++) {
					if (IDX_h[my_thread_id][k] == idx) {
						notfinished = 1;
					}
				}
			}
			IDX_h[my_thread_id][j] = idx;
			parallel::atomic_add(inst.b[idx], 1.0);
			inst.A_csc_values[i * p + j] = 2 * val - 1;
			inst.A_csc_row_idx[i * p + j] = idx;
		}
	}
	inst.A_csc_col_ptr[n] = n * p;
	L min = n;
	L max = 0;
	for (L i = 0; i < m; i++) {
		if (inst.b[i] > max)
			max = inst.b[i];
		if (inst.b[i] < min)
			min = inst.b[i];
	}
	inst.omega = max;

	std::vector<L> histogram(max + 1, 0);
	for (L i = 0; i < m; i++) {
		L tmp = inst.b[i];
		histogram[tmp]++;
	}
	histrogramLogFile << "row cardinality,count" << endl;
	for (unsigned int i = 0; i < histogram.size(); i++) {
		histrogramLogFile << i << "," << histogram[i] << endl;
	}
	tmp = 0;
#pragma omp parallel for schedule(static,1) reduction(+ : tmp)
	for (L j = 0; j < m; j++) {
		inst.b[j] = (D) (rand_r(&myseed) / (RAND_MAX + 1.0));
		tmp += inst.b[j] * inst.b[j];
	}
#pragma omp parallel for schedule(static,1)
	for (L j = 0; j < m; j++)
		inst.b[j] = inst.b[j] / tmp;
#pragma omp parallel for schedule(static,1)
	for (L col = 0; col < n; col++) {
		dataToSort[col] = 0;
		for (L rowId = inst.A_csc_col_ptr[col];
				rowId < inst.A_csc_col_ptr[col + 1]; rowId++) {
			dataToSort[col] += inst.b[inst.A_csc_row_idx[rowId]]
					* inst.A_csc_values[rowId];
		}
	}
	//Sorting B
	inst.x.resize(n);
	for (i = 0; i < n; i++) {
		inst.x[i] = dataToSort[i];
		dataToSort[i] = abs(dataToSort[i]);
	}
	std::sort(dataToSort.begin(), dataToSort.end(), greater<D>());
	D treshHoldValue = dataToSort[n_nonzero];

#pragma omp parallel for schedule(static,1)
	for (i = 0; i < n; i++) { // creating a final matrix A
		L idx = i;

		double alpha = 1;
		D oldVal = inst.x[idx];
		inst.x[idx] = 0;
		if (dataToSort[i] > treshHoldValue) {
			alpha = (double) abs(1 / oldVal);
			inst.x[idx] = ((D) (rand_r(&myseed) / (RAND_MAX + 1.0))) * rho
					/ (sqrtofnonzeros);
			if (oldVal < 0) {
				inst.x[idx] = -inst.x[idx];
			}
		} else if (dataToSort[idx] > 0.1 || dataToSort[i] < -0.1) {
			alpha = (double) abs(1 / oldVal)
					* ((D) (rand_r(&myseed) / (RAND_MAX + 1.0)));
		}
		L begining = inst.A_csc_col_ptr[idx];
		for (L j = 0; j < p; j++) {
			inst.A_csc_values[begining + j] = inst.A_csc_values[begining + j]
					* alpha;
		}
	}
#pragma omp parallel for schedule(static,1) reduction(+ : optimalvalue )
	for (i = 0; i < m; i++) {
		optimalvalue += inst.b[i] * inst.b[i];
	}
	optimalvalue = optimalvalue * 0.5;
	D sum_of_x = 0;
#pragma omp parallel for schedule(static,1) reduction(+ : sum_of_x )
	for (i = 0; i < n; i++) {
		if (inst.x[i] > 0)
			sum_of_x += inst.x[i];
		else
			sum_of_x -= inst.x[i];
	}
	optimalvalue += lambda * sum_of_x;
#pragma omp parallel for schedule(static,1)
	for (L col = 0; col < n; col++) {
		for (L rowId = inst.A_csc_col_ptr[col];
				rowId < inst.A_csc_col_ptr[col + 1]; rowId++) {

			parallel::atomic_add(inst.b[inst.A_csc_row_idx[rowId]],
					inst.x[col] * inst.A_csc_values[rowId]);

		}
	}
//check if the "x" gave me the optimal value....
	std::vector<D> residuals(m);
	Losses<L, D, square_loss_traits>::bulkIterations(inst, residuals);
	D fval = Losses<L, D, square_loss_traits>::compute_fast_objective(inst,
			residuals);
	cout << "Computed and generated optimal value:" << fval << "    "
			<< optimalvalue << endl;
	return optimalvalue;
}

template<typename D, typename L>
double nesterov_generator_row_binary(ProblemData<L, D> &inst, L n, L m,
		L p_row_min, L p_row_max, std::vector<gsl_rng *>& rs,
		const double probability = 0) {
	double lambda = 1;
	double rho = .1;
	int n_nonzero = 100000;
	if (n_nonzero > n)
		n_nonzero = n / 100;
	double sqrtofnonzeros = sqrt(n_nonzero + 0.0);

	inst.A_csr_row_ptr.resize(m + 1, 0);
	inst.A_csr_row_ptr[0] = 0;
	double averageNNZPerRow = 0;
	for (int i = 1; i <= m; i++) {
		int npr = p_row_min;

		if (gsl_rng_uniform(gsl_rng_r) > probability)
			npr = p_row_max;

		averageNNZPerRow += npr;
		inst.A_csr_row_ptr[i] = inst.A_csr_row_ptr[i - 1] + npr;
	}
	averageNNZPerRow = averageNNZPerRow / m;
	cout << "Average nnz per row " << averageNNZPerRow << endl;
	cout << "Need " << 8*averageNNZPerRow * m / (1024 * 1024*1024 + 0.0) << " GB" << endl;
	L nnz = inst.A_csr_row_ptr[m];
	inst.A_csc_col_ptr.resize(n + 1, 0);
	std::vector<L> colcounts(n + 1, 0);
	for (int i = 0; i < colcounts.size(); i++)
		colcounts[i] = 0;
	inst.A_csr_col_idx.resize(nnz);
#pragma omp parallel for schedule(static,1)
	for (int row = 0; row < m; row++) {
		for (int rpt = inst.A_csr_row_ptr[row];
				rpt < inst.A_csr_row_ptr[row + 1]; rpt++) {
			L idx;
			int notfinished = 1;
			while (notfinished) {
				notfinished = 0;
//				if (gsl_rng_uniform(gsl_rng_r) > factorization) {
				idx = gsl_rng_uniform_int(gsl_rng_r, n);
//				} else {
//					idx = gsl_rng_uniform_int(gsl_rng_r, n / TOTAL_THREADS)
//							+ n / TOTAL_THREADS * my_thread_id;
//				}

				for (L k = inst.A_csr_row_ptr[row]; k < rpt; k++) {
					if (inst.A_csr_col_idx[k] == idx) {
						notfinished = 1;
						break;
					}
				}
			}
			inst.A_csr_col_idx[rpt] = idx;
			colcounts[idx + 1]++;
		}
	}
	for (int i = 1; i <= colcounts.size(); i++) {
		colcounts[i] += colcounts[i - 1];
		inst.A_csc_col_ptr[i] = colcounts[i];
	}
	inst.A_csc_col_ptr[0] = 0;
	L nn = n;
	inst.lambda = lambda;
	inst.m = m;
	inst.n = n;
	std::vector<D> dataToSort(n);
	inst.A_csc_row_idx.resize(nnz, 0);
	inst.A_csc_values.resize(nnz, 0);
	inst.b.resize(m, 0);
	L i;
	randomNumberUtil::init_random_seeds(rs);

	double optimalvalue = 0;
	inst.x.resize(n, 0);
	double tmp;
	for (int row = 0; row < m; row++) {
		for (int rpt = inst.A_csr_row_ptr[row];
				rpt < inst.A_csr_row_ptr[row + 1]; rpt++) {
			double val = (D) (rand_r(&myseed) / (RAND_MAX + 1.0));
			L idx = inst.A_csr_col_idx[rpt];
			L colPtr = colcounts[idx];
			inst.A_csc_values[colPtr] = 2 * val - 1;
			inst.A_csc_row_idx[colPtr] = row;
			colcounts[idx]++;
		}
	}
	inst.A_csc_col_ptr[n] = nnz;

	tmp = 0;
#pragma omp parallel for schedule(static,1) reduction(+ : tmp)
	for (L j = 0; j < m; j++) {
		inst.b[j] = (D) (rand_r(&myseed) / (RAND_MAX + 1.0));
		tmp += inst.b[j] * inst.b[j];
	}
#pragma omp parallel for schedule(static,1)
	for (L j = 0; j < m; j++)
		inst.b[j] = inst.b[j] / tmp;
#pragma omp parallel for schedule(static,1)
	for (L col = 0; col < n; col++) {
		dataToSort[col] = 0;
		for (L rowId = inst.A_csc_col_ptr[col];
				rowId < inst.A_csc_col_ptr[col + 1]; rowId++) {
			dataToSort[col] += inst.b[inst.A_csc_row_idx[rowId]]
					* inst.A_csc_values[rowId];
		}
	}
	//Sorting B
	inst.x.resize(n);
	for (i = 0; i < n; i++) {
		inst.x[i] = dataToSort[i];
		dataToSort[i] = abs(dataToSort[i]);
	}
	std::sort(dataToSort.begin(), dataToSort.end(), greater<D>());
	D treshHoldValue = dataToSort[n_nonzero];

#pragma omp parallel for schedule(static,1)
	for (i = 0; i < n; i++) { // creating a final matrix A
		L idx = i;

		double alpha = 1;
		D oldVal = inst.x[idx];
		inst.x[idx] = 0;
		if (dataToSort[i] > treshHoldValue) {
			alpha = (double) abs(1 / oldVal);
			inst.x[idx] = ((D) (rand_r(&myseed) / (RAND_MAX + 1.0))) * rho
					/ (sqrtofnonzeros);
			if (oldVal < 0) {
				inst.x[idx] = -inst.x[idx];
			}
		} else if (dataToSort[idx] > 0.1 || dataToSort[i] < -0.1) {
			alpha = (double) abs(1 / oldVal)
					* ((D) (rand_r(&myseed) / (RAND_MAX + 1.0)));
		}
		for (L j = inst.A_csc_col_ptr[idx]; j < inst.A_csc_col_ptr[idx + 1];
				j++) {
			inst.A_csc_values[j] = inst.A_csc_values[j] * alpha;
		}
	}
#pragma omp parallel for schedule(static,1) reduction(+ : optimalvalue )
	for (i = 0; i < m; i++) {
		optimalvalue += inst.b[i] * inst.b[i];
	}
	optimalvalue = optimalvalue * 0.5;
	D sum_of_x = 0;
#pragma omp parallel for schedule(static,1) reduction(+ : sum_of_x )
	for (i = 0; i < n; i++) {
		if (inst.x[i] > 0)
			sum_of_x += inst.x[i];
		else
			sum_of_x -= inst.x[i];
	}
	optimalvalue += lambda * sum_of_x;
#pragma omp parallel for schedule(static,1)
	for (L col = 0; col < n; col++) {
		for (L rowId = inst.A_csc_col_ptr[col];
				rowId < inst.A_csc_col_ptr[col + 1]; rowId++) {

			parallel::atomic_add(inst.b[inst.A_csc_row_idx[rowId]],
					inst.x[col] * inst.A_csc_values[rowId]);

		}
	}
	//check if the "x" gave me the optimal value....
	std::vector<D> residuals(m);
	Losses<L, D, square_loss_traits>::bulkIterations(inst, residuals);
	D fval = Losses<L, D, square_loss_traits>::compute_fast_objective(inst,
			residuals);
	cout << "Computed and generated optimal value:" << fval << "    "
			<< optimalvalue << endl;
	return optimalvalue;
}

template<typename D, typename L>
double nesterov_generator_row_unbalanced(ProblemData<L, D> &inst, L n, L m,
		L p_row_min, L p_row_max, std::vector<gsl_rng *>& rs,
		const double factorization = 0) {
	double lambda = 1;
	double rho = .1;
	int n_nonzero = 100000;
	if (n_nonzero > n)
		n_nonzero = n / 100;
	double sqrtofnonzeros = sqrt(n_nonzero + 0.0);

	inst.A_csr_row_ptr.resize(m + 1, 0);
	inst.A_csr_row_ptr[0] = 0;
	double averageNNZPerRow = 0;
	for (int i = 1; i <= m; i++) {
		int npr = p_row_min;
		if (p_row_min < p_row_max)
			npr += gsl_rng_uniform_int(gsl_rng_r, p_row_max - p_row_min);
		averageNNZPerRow += npr;
		inst.A_csr_row_ptr[i] = inst.A_csr_row_ptr[i - 1] + npr;
	}
	averageNNZPerRow = averageNNZPerRow / m;
	cout << "Average nnz per row " << averageNNZPerRow << endl;

	L nnz = inst.A_csr_row_ptr[m];
	inst.A_csc_col_ptr.resize(n + 1, 0);
	std::vector<L> colcounts(n + 1, 0);
	for (int i = 0; i < colcounts.size(); i++)
		colcounts[i] = 0;
	inst.A_csr_col_idx.resize(nnz);
#pragma omp parallel for schedule(static,1)
	for (int row = 0; row < m; row++) {
		for (int rpt = inst.A_csr_row_ptr[row];
				rpt < inst.A_csr_row_ptr[row + 1]; rpt++) {
			L idx;
			int notfinished = 1;
			while (notfinished) {
				notfinished = 0;
				if (gsl_rng_uniform(gsl_rng_r) > factorization) {
					idx = gsl_rng_uniform_int(gsl_rng_r, n);
				} else {
					idx = gsl_rng_uniform_int(gsl_rng_r, n / TOTAL_THREADS)
							+ n / TOTAL_THREADS * my_thread_id;
				}

				for (L k = inst.A_csr_row_ptr[row]; k < rpt; k++) {
					if (inst.A_csr_col_idx[k] == idx) {
						notfinished = 1;
						break;
					}
				}
			}
			inst.A_csr_col_idx[rpt] = idx;
			colcounts[idx + 1]++;
		}
	}
	for (int i = 1; i <= colcounts.size(); i++) {
		colcounts[i] += colcounts[i - 1];
		inst.A_csc_col_ptr[i] = colcounts[i];
	}
	inst.A_csc_col_ptr[0] = 0;
	L nn = n;
	inst.lambda = lambda;
	inst.m = m;
	inst.n = n;
	std::vector<D> dataToSort(n);
	inst.A_csc_row_idx.resize(nnz, 0);
	inst.A_csc_values.resize(nnz, 0);
	inst.b.resize(m, 0);
	L i;
	randomNumberUtil::init_random_seeds(rs);

	double optimalvalue = 0;
	inst.x.resize(n, 0);
	double tmp;
	for (int row = 0; row < m; row++) {
		for (int rpt = inst.A_csr_row_ptr[row];
				rpt < inst.A_csr_row_ptr[row + 1]; rpt++) {
			double val = (D) (rand_r(&myseed) / (RAND_MAX + 1.0));
			L idx = inst.A_csr_col_idx[rpt];
			L colPtr = colcounts[idx];
			inst.A_csc_values[colPtr] = 2 * val - 1;
			inst.A_csc_row_idx[colPtr] = row;
			colcounts[idx]++;
		}
	}
	inst.A_csc_col_ptr[n] = nnz;

	tmp = 0;
#pragma omp parallel for schedule(static,1) reduction(+ : tmp)
	for (L j = 0; j < m; j++) {
		inst.b[j] = (D) (rand_r(&myseed) / (RAND_MAX + 1.0));
		tmp += inst.b[j] * inst.b[j];
	}
#pragma omp parallel for schedule(static,1)
	for (L j = 0; j < m; j++)
		inst.b[j] = inst.b[j] / tmp;
#pragma omp parallel for schedule(static,1)
	for (L col = 0; col < n; col++) {
		dataToSort[col] = 0;
		for (L rowId = inst.A_csc_col_ptr[col];
				rowId < inst.A_csc_col_ptr[col + 1]; rowId++) {
			dataToSort[col] += inst.b[inst.A_csc_row_idx[rowId]]
					* inst.A_csc_values[rowId];
		}
	}
	//Sorting B
	inst.x.resize(n);
	for (i = 0; i < n; i++) {
		inst.x[i] = dataToSort[i];
		dataToSort[i] = abs(dataToSort[i]);
	}
	std::sort(dataToSort.begin(), dataToSort.end(), greater<D>());
	D treshHoldValue = dataToSort[n_nonzero];

#pragma omp parallel for schedule(static,1)
	for (i = 0; i < n; i++) { // creating a final matrix A
		L idx = i;

		double alpha = 1;
		D oldVal = inst.x[idx];
		inst.x[idx] = 0;
		if (dataToSort[i] > treshHoldValue) {
			alpha = (double) abs(1 / oldVal);
			inst.x[idx] = ((D) (rand_r(&myseed) / (RAND_MAX + 1.0))) * rho
					/ (sqrtofnonzeros);
			if (oldVal < 0) {
				inst.x[idx] = -inst.x[idx];
			}
		} else if (dataToSort[idx] > 0.1 || dataToSort[i] < -0.1) {
			alpha = (double) abs(1 / oldVal)
					* ((D) (rand_r(&myseed) / (RAND_MAX + 1.0)));
		}
		for (L j = inst.A_csc_col_ptr[idx]; j < inst.A_csc_col_ptr[idx + 1];
				j++) {
			inst.A_csc_values[j] = inst.A_csc_values[j] * alpha;
		}
	}
#pragma omp parallel for schedule(static,1) reduction(+ : optimalvalue )
	for (i = 0; i < m; i++) {
		optimalvalue += inst.b[i] * inst.b[i];
	}
	optimalvalue = optimalvalue * 0.5;
	D sum_of_x = 0;
#pragma omp parallel for schedule(static,1) reduction(+ : sum_of_x )
	for (i = 0; i < n; i++) {
		if (inst.x[i] > 0)
			sum_of_x += inst.x[i];
		else
			sum_of_x -= inst.x[i];
	}
	optimalvalue += lambda * sum_of_x;
#pragma omp parallel for schedule(static,1)
	for (L col = 0; col < n; col++) {
		for (L rowId = inst.A_csc_col_ptr[col];
				rowId < inst.A_csc_col_ptr[col + 1]; rowId++) {

			parallel::atomic_add(inst.b[inst.A_csc_row_idx[rowId]],
					inst.x[col] * inst.A_csc_values[rowId]);

		}
	}
//check if the "x" gave me the optimal value....
	std::vector<D> residuals(m);
	Losses<L, D, square_loss_traits>::bulkIterations(inst, residuals);
	D fval = Losses<L, D, square_loss_traits>::compute_fast_objective(inst,
			residuals);
	cout << "Computed and generated optimal value:" << fval << "    "
			<< optimalvalue << endl;
	return optimalvalue;
}
#endif // GENERATOR_NESTEROV_
