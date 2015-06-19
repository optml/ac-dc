#ifndef GENERATOR_FAT_H_
#define GENERATOR_FAT_H_

//#ifndef _OPENMP
//#include "../solver/structures.h"
//#else
#include "../distributed/distributed_structures.h"
//#endif


template<typename D, typename L>
void generate_fat_problem(int m, int n, ProblemData<L, D> &instOut, mpi::communicator &world,
		DistributedSettings &settings, int p) {
	ProblemData<L, D> inst;
	srand(2 + world.rank());
	double lambda = 1;
	L nnz = n * p;
	instOut.lambda = lambda;
	inst.m = m;
	inst.n = n;
	world.barrier();
	inst.A_csc_col_ptr.resize(n + 1, 0);
	inst.A_csc_row_idx.resize(nnz, 0);
	inst.A_csc_values.resize(nnz, 0);
	instOut.b.resize(inst.m, 0);

	int i, j, k;
	std::vector<int> IDX_h(p); // host Aindex matrix pointers
	double x_optimal[n];
	double optimalvalue = 0;
	double tmp;

	//Generovanie problemu-------------------------------------------------------------------
	nnz = 0;
	for (i = 0; i < n; i++) {
		inst.A_csc_col_ptr[i] = nnz;
		int idx = 0;
		for (j = 0; j < p; j++) {
			int notfinished = 1;
			double val = (double) rand() / RAND_MAX;
			while (notfinished) {
				notfinished = 0;

				idx = ((int) ((inst.m) * (rand() / (RAND_MAX + 0.0))));

				if (idx >= inst.m) {
					idx--;
				}

				//				idx = ((int) ((inst.m) * (rand() / (RAND_MAX + 1.0))));
				for (k = 0; k < j; k++) {
					if (IDX_h[k] == idx) {
						notfinished = 1;
					}
				}
			}
			IDX_h[j] = idx;
			inst.A_csc_row_idx[nnz] = idx;
			inst.A_csc_values[nnz] = 2 * val - 1;
			nnz++;
		}
	}

	inst.A_csc_col_ptr[n] = nnz;
	instOut.x.resize(n, 0);
	for (j = 0; j < inst.m; j++) {
		instOut.b[j] = (rand() / (RAND_MAX + 1.0));
	}
	instOut.n = inst.n;
	instOut.m = inst.m;
	getCSR_from_CSC(inst.A_csc_values, inst.A_csc_row_idx, inst.A_csc_col_ptr, instOut.A_csr_values,
			instOut.A_csr_col_idx, instOut.A_csr_row_ptr, inst.m, inst.n);
}

#endif // GENERATOR_FAT_H_
