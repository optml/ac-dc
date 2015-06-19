#ifndef GENERATOR_BLOCKED_H_
#define GENERATOR_BLOCKED_H_


template<typename D, typename L>
void generate_block_problem(ProblemData<L, D> &inst, int n, int m, int k, int p = 20, float densityOfCommonPart = 0.2) {

	// k is world.size()

	srand(2 + k);
	L nnz = n * p;
	inst.lambda = 1;
	inst.m = m / k * (k + 1);
	inst.n = n;

	inst.A_csc_col_ptr.resize(n + 1, 0);
	inst.A_csc_row_idx.resize(nnz, 0);
	inst.A_csc_values.resize(nnz, 0);
	inst.b.resize(inst.m, 0);
	int i, j;
	std::vector<int> IDX_h(p); // host Aindex matrix pointers
	double x_optimal[n];
	double optimalvalue = 0;
	double b[inst.m];
	double x[n];
	double tmp;

	int mblock = inst.m / (k + 1);

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

				if ((rand() / (RAND_MAX + 1.0)) < densityOfCommonPart) {
					idx = ((int) ((mblock) * (rand() / (RAND_MAX + 0.0))));
//				printf("idx %d \n", idx);
				} else {

					idx = ((int) ((mblock) * (rand() / (RAND_MAX + 0.0))));
					idx += mblock * (1+ (i*k)/n );
//					printf(">>> idx %d \n", idx);
				}

				if (idx>=inst.m){
					idx--;
					printf("FUUUUUUUUUJ!!!!\n");
				}



//				idx = ((int) ((inst.m) * (rand() / (RAND_MAX + 1.0))));
				for (int cnt = 0; cnt < j; cnt++) {
					if (IDX_h[cnt] == idx) {
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
	inst.x.resize(n, 0);
	for (j = 0; j < inst.m; j++) {
		inst.b[j] = (rand() / (RAND_MAX + 1.0));
	}

}

#endif // GENERATOR_BLOCKED
