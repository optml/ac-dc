#ifndef GENERATOR_NESTEROV_H_
#define GENERATOR_NESTEROV_H_

//#ifndef _OPENMP
//#include "../solver/structures.h"
//#else
//#include "../distributed/distributed_structures.h"
//#endif

template<typename D, typename L>
class st_sortingByAbsWithIndex {
public:
	D value;
	L idx;
};

/* qsort struct comparision function (price double field) */
template<typename D, typename L>
int struct_cmp_by_value(const void *a, const void *b) {
	st_sortingByAbsWithIndex<D, L> *ia = (st_sortingByAbsWithIndex<D, L> *) a;
	st_sortingByAbsWithIndex<D, L> *ib = (st_sortingByAbsWithIndex<D, L> *) b;
	D aa = ia->value;
	D bb = ib->value;
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

template<typename D, typename L>
double nesterov_generator(ProblemData<L, D> &inst, L n, L m, L p, std::vector<gsl_rng *>& rs) {
	double lambda = 1;
	double rho = .1;
	int n_nonzero = 100000;
	if (n_nonzero > n)
		n_nonzero = n / 100;
	double sqrtofnonzeros = sqrt(n_nonzero + 0.0);
	L nn = n;
	L nnz = nn * p;
	std::cout << "TOTAL NNZ " << nnz << " and n= " << n << " and p=" << p << std::endl;
	inst.lambda = lambda;
	inst.m = m;
	inst.n = n;

	cout << "Start initializint 7" << endl;

	cout << "size of structure" << endl;

//  dataToSort= (st_sortingByAbsWithIndex*) malloc (n);
//  if (dataToSort==NULL) exit (1);

//	st_sortingByAbsWithIndex* dataToSort;
//	dataToSort = (st_sortingByAbsWithIndex*) calloc(n,	sizeof(st_sortingByAbsWithIndex));
	std::vector<D> dataToSort(n);

//	std::vector<st_sortingByAbsWithIndex> dataToSort(n);

	cout << "Start initializint " << endl;
	inst.A_csc_col_ptr.resize(n + 1, 0);
	cout << "Start initializint 1" << endl;
	inst.A_csc_row_idx.resize(nnz, 0);
	cout << "Start initializint 2" << endl;
	inst.A_csc_values.resize(nnz, 0);
	cout << "Start initializint 3" << endl;
	inst.b.resize(m, 0);
	cout << "Start initializint 4" << endl;
	L i;
	init_random_seeds(rs);
	L IDX_h[TOTAL_THREADS][p]; // host Aindex matrix pointers
	double optimalvalue = 0;
	cout << "Start initializint 5" << endl;
	inst.x.resize(n, 0);
	cout << "Start initializint 6" << endl;
	double tmp;
	//Generovanie problemu-------------------------------------------------------------------
	nnz = 0;
	cout << "start generate matrix" << endl;
#pragma omp parallel for schedule(static,1)
	for (i = 0; i < n; i++) {
		inst.A_csc_col_ptr[i] = i * p;
		L idx = 0;
		for (L j = 0; j < p; j++) {
			int notfinished = 1;
			double val = (D) (rand_r(&myseed) / (RAND_MAX + 1.0));
			while (notfinished) {
				notfinished = 0;
//				idx = ((L) ((m) * (rand_r(&myseed) / (RAND_MAX + 1.0))));
				idx = gsl_rng_uniform_int(gsl_rng_r, m);
				if (j == 0) {
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
	cout << "data generated" << endl;
	L min = n;
	L max = 0;
	for (L i = 0; i < m; i++) {
		if (inst.b[i] > max)
			max = inst.b[i];
		if (inst.b[i] < min)
			min = inst.b[i];
	}
	cout << "Row statistics: MINIMUM " << min << endl;
	cout << "Row statistics: MAXIMUM " << max << endl;

	std::vector<L> histogram(max + 1, 0);
	for (L i = 0; i < m; i++) {
		L tmp = inst.b[i];
		histogram[tmp]++;
	}
	for (int i = 0; i < histogram.size(); i++) {
		cout << "Histrogram " << i << ":" << histogram[i] << endl;
	}

	cout << "Historgam obtained" << endl;
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
		for (L rowId = inst.A_csc_col_ptr[col]; rowId < inst.A_csc_col_ptr[col + 1]; rowId++) {
			dataToSort[col] += inst.b[inst.A_csc_row_idx[rowId]] * inst.A_csc_values[rowId];
		}
	}
	cout << "Start initializint 8" << endl;
	//Sorting B
	printf("SORTING START\n");
//	size_t structs_len = sizeof(dataToSort) / sizeof(st_sortingByAbsWithIndex);
// FIXME	qsort(&dataToSort[0], n, sizeof(st_sortingByAbsWithIndex<L, D> ), struct_cmp_by_value<L, D>);
	inst.x.resize(n);
	cout << "x is resized" << endl;
	for (i = 0; i < n; i++) {
		inst.x[i] = dataToSort[i];
		dataToSort[i] = abs(dataToSort[i]);
	}
	cout << "daa filed" << endl;
	std::sort(dataToSort.begin(), dataToSort.end(), greater<D>());
	cout << "sorted" << endl;
	D treshHoldValue = dataToSort[n_nonzero];

	cout << dataToSort[0] << "   -  " << dataToSort[n / 2] << "  -  " << dataToSort[n - 1] << endl;

	// TODO

	printf("SORTING END\n");
	cout << "Start initializint 9" << endl;
#pragma omp parallel for schedule(static,1)
	for (i = 0; i < n; i++) { // vytvaranie matice A
		L idx = i;

		double alpha = 1;
		D oldVal = inst.x[idx];
		inst.x[idx] = 0;
		if (abs(oldVal) > treshHoldValue) {
			alpha = (double) abs(1 / oldVal);
			//			printf("alpha = %f \n", alpha);
			inst.x[idx] = ((D) (rand_r(&myseed) / (RAND_MAX + 1.0))) * rho / (sqrtofnonzeros);
			if (oldVal < 0) {
				inst.x[idx] = -inst.x[idx];
			}
		} else if (abs(oldVal) > 0.1 ) {
			alpha = (double) abs(1 / oldVal) * ((D) (rand_r(&myseed) / (RAND_MAX + 1.0)));
			//			printf("alpha = %f \n", alpha);
		}
		L begining = inst.A_csc_col_ptr[idx];
		for (L j = 0; j < p; j++) {
			inst.A_csc_values[begining + j] = inst.A_csc_values[begining + j] * alpha;
		}
	}
	cout << "Start initializint 10" << endl;
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
	printf("optval %1.16f   (|x|=%f)\n", optimalvalue, sum_of_x);
#pragma omp parallel for schedule(static,1)
	for (L col = 0; col < n; col++) {
		for (L rowId = inst.A_csc_col_ptr[col]; rowId < inst.A_csc_col_ptr[col + 1]; rowId++) {

			parallel::atomic_add(inst.b[inst.A_csc_row_idx[rowId]], inst.x[col] * inst.A_csc_values[rowId]);

		}
	}
//check if the "x" gave me the optimal value....
	std::vector<D> residuals(m);
	MyTMPLosses<L, D, square_loss_traits>::bulkIterations(inst, residuals);
	D fval = MyTMPLosses<L, D, square_loss_traits>::compute_fast_objective(inst, residuals);
	cout << "Computed and generated val:" << fval << "    " << optimalvalue << endl;

//	free (dataToSort);
	return optimalvalue;
	/*


	 //DEBUG
	 inst.x.resize(n);
	 std::vector<D> residuals(m, 0);
	 for (int i = 0; i < n; i++)
	 inst.x[i] = x_optimal[i];
	 Losses<L, D, square_loss_traits>::bulkIterations(inst, residuals);
	 D sum_of_residuals = 0;
	 for (L i = 0; i < m; i++) {
	 sum_of_residuals += residuals[i] * residuals[i];
	 }
	 D objective = Losses<L, D, square_loss_traits>::compute_fast_objective(inst, residuals);
	 printf("At termination:\nObjective %f, \t |residuals|^2 %f\n", objective, sum_of_residuals);
	 for (int i = 0; i < n; i++)
	 inst.x[i] = 0;
	 */
}

//template<typename D, typename L>
//void nesterov_generator(ProblemData<L, D> &inst) {
//	int n = 5000;
//	int m = n / 2;
//	int p = 10;
//	nesterov_generator(inst, n, m, p);
//}

#endif // GENERATOR_NESTEROV_
