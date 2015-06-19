#ifndef GENERATOR_K_DIAGONAL_SYSTEM_H_
#define GENERATOR_K_DIAGONAL_SYSTEM_H_


template<typename D, typename L>
void generate_k_diagonal(ProblemData<L, D> &inst, int &m, int &n, int _n, int _k) {
	int k = _k;
	n = _n;
	long long nnz = k * n;

	inst.A_csc_col_ptr.resize(n + 1);
	inst.A_csc_row_idx.resize(nnz);
	inst.A_csc_values.resize(nnz);
	inst.x.resize(n, 0);

	inst.m = n;
	inst.n = n;
	m = n;

	D value = 1;
	nnz = 0;
	for (int i = 0; i < n; i++) {
		inst.x[i] = (float) rand() / RAND_MAX;
		inst.A_csc_col_ptr[i] = nnz;
		for (int j = i - k / 2; j <= i + k / 2; j++) {
			if (j >= 0 and j < n) {
				inst.A_csc_row_idx[nnz] = j;
				inst.A_csc_values[nnz] = (float) rand() / RAND_MAX;
			} else {
				int shift = k;
				if (i > k) {
					shift = -k;
				}
				inst.A_csc_row_idx[nnz] = j + shift;
				inst.A_csc_values[nnz] = 0;
			}
			nnz++;
		}
	}

	inst.A_csc_col_ptr[n] = nnz;
	inst.b.resize(m, 0);
	for (int i = 0; i < m; i++) {
		inst.b[i] = 0;
	}
	for (int col = 0; col < n; col++) {
		for (int j = inst.A_csc_col_ptr[col]; j <= inst.A_csc_col_ptr[col + 1]; j++) {
			inst.b[inst.A_csc_row_idx[j]] += inst.A_csc_values[j] * inst.x[col];
		}
		inst.x[col] = 0; //vynulovanie
	}
	inst.lambda = 0;
	inst.sigma = 1;
}

template<typename D, typename L>
void generate_k_diagonal_with_few_full_columns(ProblemData<L, D> &inst, int &m, int &n, int _n, int _k, int p = 4) {
	int k = _k;
	n = _n;
	long long nnz = k * n + (n - k) * p;

	inst.A_csc_col_ptr.resize(n + 1);
	inst.A_csc_row_idx.resize(nnz);
	inst.A_csc_values.resize(nnz);
	inst.x.resize(n, 0);

	inst.m = n;
	inst.n = n;
	m = n;

	D value = 1;
	nnz = 0;
	for (int i = 0; i < n; i++) {
		inst.x[i] = (float) rand() / RAND_MAX;
		inst.A_csc_col_ptr[i] = nnz;
		if (i < p) {
			for (int j = 0; j < n; j++) {
				inst.A_csc_row_idx[nnz] = j;
				inst.A_csc_values[nnz] = (float) rand() / RAND_MAX;
				nnz++;
			}

		} else {

			for (int j = i - k / 2; j <= i + k / 2; j++) {
				if (j >= 0 and j < n) {
					inst.A_csc_row_idx[nnz] = j;
					inst.A_csc_values[nnz] = (float) rand() / RAND_MAX;
				} else {
					int shift = k;
					if (i > k) {
						shift = -k;
					}
					inst.A_csc_row_idx[nnz] = j + shift;
					inst.A_csc_values[nnz] = 0;
				}
				nnz++;
			}
		}
	}

	inst.A_csc_col_ptr[n] = nnz;
	inst.b.resize(m, 0);
	for (int i = 0; i < m; i++) {
		inst.b[i] = 0;
	}
	for (int col = 0; col < n; col++) {
		for (int j = inst.A_csc_col_ptr[col]; j <= inst.A_csc_col_ptr[col + 1]; j++) {
			inst.b[inst.A_csc_row_idx[j]] += inst.A_csc_values[j] * inst.x[col];
		}
		inst.x[col] = 0; //vynulovanie
	}
	inst.lambda = 0;
	inst.sigma = 1;
}

#endif // GENERATOR_K_DIAGONAL_SYSTEM_H_
