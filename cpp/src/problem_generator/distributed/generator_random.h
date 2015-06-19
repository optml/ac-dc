#ifndef GENERATOR_RANDOM_H_
#define GENERATOR_RANDOM_H_

#include <ctime>





// NOTE p is an integer fro the number of columns, rather than the probability each entry is nnz!
template<typename D, typename L>
void generate_random_problem(ProblemData<L, D> &inst, int n, int m, int p, int seed = -1) {

	if (seed < 0)	srand(time(NULL));
	if (seed >= 0)	srand(seed);
	L nnz = n * p;
	inst.lambda = 1;
	inst.m = m;
	inst.n = n;
//	cout << "allocate " <<n <<" "<< p << " "<< nnz << " end"<<endl;
	inst.A_csc_col_ptr.resize(n + 1, 0);
//	cout << "allocate " <<n <<" "<< p << " "<< nnz << " end"<<endl;
	inst.A_csc_row_idx.resize(nnz, 0);
//	cout << "allocate " <<n <<" "<< p << " "<< nnz << " end"<<endl;
	inst.A_csc_values.resize(nnz, 0);
//	cout << "allocate " <<n <<" "<< p << " "<< nnz << " end"<<endl;
	inst.b.resize(m, 0);
//	cout << "allocate " <<n <<" "<< p << " "<< nnz << " end"<<endl;
	L i, j, k;
	int IDX_h[p]; // host Aindex matrix pointers
	double x_optimal[n];
	double optimalvalue = 0;
	double b[m];
	double x[n];
	double tmp;

	// Generate the instance
	nnz = 0;
	for (i = 0; i < n; i++) {
		inst.A_csc_col_ptr[i] = nnz;
		int idx = 0;
		for (j = 0; j < p; j++) {
			int notfinished = 1;
			double val = (double) rand() / RAND_MAX;
			while (notfinished) {
				notfinished = 0;
				idx = ((int) ((m) * (rand() / (RAND_MAX + 1.0))));
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
	inst.x.resize(n,0);
	for (j = 0; j < m; j++) {
		inst.b[j] = (rand() / (RAND_MAX + 1.0));
	}
}

template<typename D, typename L>
void generate_random_known(ProblemData<L, D> &inst) {
	int n = 1000;
	int m = 500;
	int p = 50;
//CVX   47.031631044949251




//	n = 10;
//	m = 7;
//	p = 5;



	int seed = 1;
	generate_random_problem(inst, n, m, p, seed);
}


#endif // GENERATOR_RANDOM_
