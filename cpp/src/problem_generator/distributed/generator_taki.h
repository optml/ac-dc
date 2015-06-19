#ifndef GENERATOR_TAKI_H_
#define GENERATOR_TAKI_H_

//#ifndef _OPENMP
//#include "../solver/structures.h"
//#else
#include "../distributed/distributed_structures.h"
#include "../parallel/openmp_helper.h"
//#endif


template<typename D, typename L>
void generate_problem_for_taki(int m, int n, ProblemData<L, D> &inst, mpi::communicator &world,
		DistributedSettings &settings, int p) {
//	omp_set_num_threads(settings.totalThreads);
	srand(2 + world.rank());
	double lambda = 1;
	L nnz = n * p;
	inst.lambda = lambda;
	inst.m = m;
	inst.n = n;
	world.barrier();
	inst.A_csc_col_ptr.resize(n + 1, 0);
	inst.A_csc_row_idx.resize(nnz, 0);
	inst.A_csc_values.resize(nnz, 0);
	inst.b.resize(inst.m, 0);

	std::vector<int> IDX_h(p*settings.totalThreads); // host Aindex matrix pointers
	double x_optimal[n];
	double optimalvalue = 0;
	double tmp;
	//Generovanie problemu-------------------------------------------------------------------
	nnz = 0;


	init_random_seeds();



#pragma omp parallel for schedule(static,1024)
	for (int i = 0; i < n; i++) {
		inst.A_csc_col_ptr[i] = i*p;
		int idx = 0;
		for (int j = 0; j < p; j++) {
			int notfinished = 1;
			double val =  (double) rand_r(&myseed) / RAND_MAX;
			while (notfinished) {
				notfinished = 0;

				idx = ((int) ((inst.m) * (rand_r(&myseed) / (RAND_MAX + 0.0))));
				if (idx >= inst.m) {
					idx--;
				}
				//				idx = ((int) ((inst.m) * (rand() / (RAND_MAX + 1.0))));
				for (int k = 0; k < j; k++) {
					if (IDX_h[k+my_thread_id*p] == idx) {
						notfinished = 1;
					}
				}
			}
			IDX_h[j+my_thread_id*p] = idx;
			inst.A_csc_row_idx[i*p+j] = idx;
			inst.A_csc_values[i*p+j] = 2 * val - 1;
		}
	}

	inst.A_csc_col_ptr[n] = p*n;
	inst.x.resize(n, 0);
	for (int j = 0; j < inst.m; j++) {
		inst.b[j] = (rand() / (RAND_MAX + 1.0));
	}
}

#endif // GENERATOR_TAKI_H_
