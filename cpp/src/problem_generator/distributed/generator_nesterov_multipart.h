#ifndef GENERATOR_NESTEROV_FILE_H_
#define GENERATOR_NESTEROV_FILE_H_
#include <vector>
//#ifndef _OPENMP
//#include "../solver/structures.h"
//#else
//#include "../distributed/distributed_structures.h"
//#endif
#include <iostream>
#include <iomanip>
#include "../../parallel/openmp_helper.h"
class st_sortingByAbsWithIndex {
public:
	double value;
	int idx;
};

class st_sortingByAbs {
public:
	double value;
};

/* qsort struct comparision function (price double field) */
template<typename D>
int struct_cmp_by_abs_value(const void *a, const void *b) {
	return (abs(*(D*) b) - abs(*(D*) a));
//
//	double aa = *(double *) a;
//	double bb = *(double *) b;
//	if (aa * aa > bb * bb)
//		return -1;
//	else if (aa * aa < bb * bb)
//		return 1;
//	else
//		return 0;

	/* double comparison: returns negative if b > a
	 and positive if a > b. We multiplied result by 100.0
	 to preserve decimal fraction */

}

/* qsort struct comparision function (price double field) */
int struct_cmp_by_value(const void *a, const void *b) {
	st_sortingByAbsWithIndex *ia = (st_sortingByAbsWithIndex *) a;
	st_sortingByAbsWithIndex *ib = (st_sortingByAbsWithIndex *) b;
	double aa = ia->value;
	double bb = ib->value;
	if (aa * aa > bb * bb)
		return -1;
	else if (aa * aa < bb * bb)
		return 1;
	else
		return 0;

	/* double comparison: returns negative if b > a
	 and positive if a > b. We multiplied result by 100.0
	 to preserve decimal fraction */

}

char* getStreamToFile(const char* base, string extra, int i = 0) {
	stringstream ss;
	string finalFileName = base;
	ss << finalFileName;
	ss << "_" << extra;
	ss << "_" << i;
	finalFileName = ss.str();
	char* cstr = new char[finalFileName.size() + 1];
	strcpy(cstr, finalFileName.c_str());
	return cstr;
}

#include "../../helpers/gsl_random_helper.h"

template<typename D, typename L>
double nesterovDistributedGenerator(ProblemData<L, D>& part, int n, int m,
		int p, int nnz_of_solution,std::vector<gsl_rng *>& rs) {
	mpi::communicator world;




	double lambda = 1;
	double rho = 1;
	int n_nonzero = nnz_of_solution;
	double sqrtofnonzeros = sqrt(nnz_of_solution + 0.0);
	L nn = n;
	L nnz = nn * p;
	part.n = n;
	part.m = m;
	std::cout << "TOTAL NNZ per node: " << nnz << " and n= " << n << " and p="
			<< p << std::endl;

	cout << "Start initializint 7" << endl;
	cout << "size of structure" << endl;

	std::vector<D> dataToSort(n);

	cout << "size of structure" << sizeof(dataToSort) << "  " << n << endl;

	part.b.resize(m, 0);
	part.A_csc_col_ptr.resize(n + 1, 0);
	part.A_csc_values.resize(n * p);
	part.A_csc_row_idx.resize(n * p);

	cout << "Start initializint 4" << endl;
	L i, j, k;
	double optimalvalue = 0;
	cout << "Start initializint 5" << endl;
	double tmp;

	tmp = 0;
	cout << "Start initializint 6" << endl;
	if (world.rank() == 0) {
#pragma omp parallel for reduction(+:tmp)
		for (j = 0; j < m; j++) {
			part.b[j] = (double) rand_r(&myseed) / RAND_MAX;
			tmp += part.b[j] * part.b[j];
		}
		for (j = 0; j < m; j++) {
			part.b[j] = part.b[j] / tmp;
		}
	}
	vbroadcast(world, part.b, 0);
	cout << "Start initializint sasda6" << endl;
	//Generovanie problemu-------------------------------------------------------------------
	nnz = 0;
//	init_random_seeds(world.rank());

#pragma omp parallel for schedule(static,1)
	for (i = 0; i < n; i++) {
		dataToSort[i] = 0;
		L idx = 0;
		for (L jj = 0; jj < p; jj++) {
			int notfinished = 1;
			double val = (double) rand_r(&myseed) / RAND_MAX;
			while (notfinished) {
				notfinished = 0;
				idx = gsl_rng_uniform_int(gsl_rng_r, m);
//				idx = ((int) ((m) * (rand_r(&myseed) / (RAND_MAX + 1.0))));
				for (k = 0; k < jj; k++) {
					if (part.A_csc_row_idx[i * p + k] == idx) {
						notfinished = 1;
					}
				}
			}

			part.A_csc_row_idx[i * p + jj] = idx;
			val = 2 * val - 1;
			part.A_csc_values[i * p + jj] = val;
			dataToSort[i] += part.b[idx] * val;
			nnz++;
		}
	}
	cout << "Start initializint 8" << endl;
	//Sorting B
	printf("SORTING START\n");
//	qsort(&dataToSort[0], n, sizeof(D), struct_cmp_by_abs_value<D>);
	part.x.resize(n, 0);
	for (i = 0; i < n; i++) {
		part.x[i] = dataToSort[i];
		dataToSort[i] = abs(dataToSort[i]);
	}
	std::sort(dataToSort.begin(), dataToSort.end(), greater<D>());

	printf("SORTING END\n");
	cout << "Start initializint 9" << endl;

	cout << dataToSort[0] << "   -  " << dataToSort[n / 2] << "  -  "
			<< dataToSort[n - 1] << endl;
	int r = 0;
	mpi::request reqs[world.size()];
	//gether the sorter values and sort again....
	std::vector<D> finalSorting(0);
	if (world.rank() == 0) {
		dataToSort.resize(nnz_of_solution * world.size());
		for (int tmpi = 1; tmpi < world.size(); tmpi++) {
			//receive
			reqs[r++] = virecv(world, tmpi, 0,
					&dataToSort[tmpi * nnz_of_solution], nnz_of_solution);
		}
	} else {
		//sent Data
		reqs[r++] = visend(world, 0, 0, &dataToSort[0], nnz_of_solution);
	}
	mpi::wait_all(reqs, reqs + r);
	D treshHoldValue = 0;
	if (world.rank() == 0) {
		//sort and get tresholdqsort(&dataToSort[0], n, sizeof(D), struct_cmp_by_abs_value);
		std::sort(dataToSort.begin(), dataToSort.end(), greater<D>());
		treshHoldValue = dataToSort[nnz_of_solution];
	}
	vbroadcast(world, &treshHoldValue, 1, 0);
	for (i = 0; i < m; i++) {
		optimalvalue += part.b[i] * part.b[i];
	}

	if (world.rank() != 0) {
		for (i = 0; i < m; i++)
			part.b[i] = 0;
	}

	init_random_seeds(world.rank());
#pragma omp parallel for schedule(static,1)
	for (i = 0; i < n; i++) { // vytvaranie matice A
		D vals = abs(part.x[i]);
		D orogival = part.x[i];
		double alpha = 1;
		part.x[i] = 0;
		if (vals >= treshHoldValue) {
			alpha = (double) abs(1 / vals);
			//			//			printf("alpha = %f \n", alpha);
			part.x[i] = ((double) rand_r(&myseed2) / RAND_MAX) * rho
					/ (sqrtofnonzeros);
			if (orogival < 0) {
				part.x[i] = -part.x[i];
			}
		} else if (vals > 0.1) {
			alpha = (double) abs(1 / vals)
					* (double) rand_r(&myseed2) / RAND_MAX;
			//			//			printf("alpha = %f \n", alpha);
		}
		// load p values and scale them
		for (L jj = 0; jj < p; jj++) {
			part.A_csc_values[jj + i * p] = alpha
					* part.A_csc_values[jj + i * p];
			parallel::atomic_add(part.b[part.A_csc_row_idx[jj + i * p]],
					part.x[i] * part.A_csc_values[jj + i * p]);
//			part.b[part.A_csc_row_idx[jj + i * p]] += part.x[i]
//					* part.A_csc_values[jj + i * p];
		}

	}
	cout << "Start initializint 10" << endl;
	D initialObjective = 0;
	std::vector<D> tmpvector(m, 0);
	vreduce(world, part.b, tmpvector, 0);
	if (world.rank() == 0) {
#pragma omp parallel for
		for (i = 0; i < m; i++) {
			part.b[i] = tmpvector[i];
		}
	}
	vbroadcast(world, part.b, 0);
#pragma omp parallel for reduction(+:initialObjective)
	for (j = 0; j < m; j++) {
		initialObjective += part.b[j] * part.b[j];
	}
	cout << "Initial Objective " << setprecision(16) << initialObjective
			<< endl;
	cout << "b[0]=" << part.b[0] << endl;
	optimalvalue = optimalvalue * 0.5;
	D sum_of_x = 0;
	for (i = 0; i < n; i++) {
		if (part.x[i] > 0)
			sum_of_x += part.x[i];
		else
			sum_of_x -= part.x[i];
	}
	D tmp_sum_of_x = 0;
	vreduce(world, &sum_of_x, &tmp_sum_of_x, 1, 0);
	sum_of_x = tmp_sum_of_x;
	optimalvalue += lambda * sum_of_x;
	if (world.rank() == 0) {
		printf("optval %1.16f   (|x|=%f)\n", optimalvalue, sum_of_x);
	}

	part.lambda = 1;
	for (i = 0; i <= n; i++)
		part.A_csc_col_ptr[i] = i * p;
	return optimalvalue;
}

//template<typename D, typename L>
//void nesterov_generator(ProblemData<L, D> &inst) {
//	int n = 5000;
//	int m = n / 2;
//	int p = 10;
//	nesterov_generator(inst, n, m, p);
//}

#endif // GENERATOR_NESTEROV_FILE_H_
