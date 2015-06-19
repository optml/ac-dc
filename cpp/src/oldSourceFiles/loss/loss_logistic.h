/*
 * This file contains an functions for experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */


/*
 * abstract_loss.h
 *
 *  Created on: 19 Jan 2012
 *      Author: jmarecek and taki
 */

#ifndef LOGISTIC_LOSS_H
#define LOGISTIC_LOSS_H

#include "loss_abstract.h"


struct logistics_loss_traits: public loss_traits {
};

/*******************************************************************/
// a partial specialisation for logistic loss
template<typename L, typename D>
class Losses<L, D, logistics_loss_traits> {

public:

	static inline void bulkIterations_for_my_instance_data(
			const ProblemData<L, D> &inst, std::vector<D> &residuals) {
		for (L row = 0; row < inst.m; row++) {
			residuals[row] = 0;
			for (L col_tmp = inst.A_csr_row_ptr[row];
					col_tmp < inst.A_csr_row_ptr[row + 1]; col_tmp++) {
				residuals[row] -= inst.b[row] * inst.A_csr_values[col_tmp]
						* inst.x[inst.A_csr_col_idx[col_tmp]];
			}
		}
	}


	static inline void bulkIterations(const ProblemData<L, D> &inst,
			std::vector<D> &residuals) {
		for (L i = 0; i < inst.m; i++) {
			residuals[i] = 0;
		}
		for (L i = 0; i < inst.n; i++) {
			for (unsigned int j = inst.A_csc_col_ptr[i];
					j < inst.A_csc_col_ptr[i + 1]; j++) {
				residuals[inst.A_csc_row_idx[j]] -= inst.x[i]
						* inst.b[inst.A_csc_row_idx[j]] * inst.A_csc_values[j];
			}
		}
	}

	static inline void set_residuals_for_zero_x(const ProblemData<L, D> &inst,
			std::vector<D> &residuals) {
		for (L i = 0; i < inst.m; i++) {
			residuals[i] = 0;
		}
	}

	static inline D do_single_iteration_serial(const ProblemData<L, D> &inst,
			const L &idx, std::vector<D> &residuals, std::vector<D> &x,
			const std::vector<D> &Li) {
		D tmp = compute_update(inst, residuals, idx, Li);
		if (tmp != 0) {
			x[idx] += tmp;
			for (unsigned int j = inst.A_csc_col_ptr[idx];
					j < inst.A_csc_col_ptr[idx + 1]; j++) {
				L row_id = inst.A_csc_row_idx[j];
				residuals[row_id] -= tmp * inst.b[row_id]
						* inst.A_csc_values[j];
			}
		}
		return abs(tmp);
	}

	static inline D do_single_iteration_parallel(const ProblemData<L, D> &inst,
			const L &idx, std::vector<D> &residuals, std::vector<D> &x,
			const std::vector<D> &Li) {
		D tmp = compute_update(inst, residuals, idx, Li);
		parallel::atomic_add(x[idx], tmp);
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(residuals[inst.A_csc_row_idx[j]],
					-tmp * inst.b[inst.A_csc_row_idx[j]]
							* inst.A_csc_values[j]);
		}
		return tmp;
	}

	static inline D do_single_iteration_parallel_for_distributed(
			const ProblemData<L, D> &inst, const L &idx,
			std::vector<D> &residuals, std::vector<D> &x,
			const std::vector<D> &Li, D* residual_updates) {
		D tmp = compute_update(inst, residuals, idx, Li);
		parallel::atomic_add(x[idx], tmp);
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(residual_updates[inst.A_csc_row_idx[j]],
					-tmp * inst.b[inst.A_csc_row_idx[j]]
							* inst.A_csc_values[j]);
		}
		return tmp;
	}

	static inline D compute_update(const ProblemData<L, D> &inst,
			const std::vector<D> &residuals, const L &idx,
			const std::vector<D> &Li) {
		D partialDetivative = 0;
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			L row_id = inst.A_csc_row_idx[j];
			D tmp = exp(residuals[row_id]);
			partialDetivative += tmp / (1 + tmp) * inst.b[row_id]
					* inst.A_csc_values[j];
		}
		partialDetivative = -0.5 * inst.lambda * partialDetivative;
		return compute_soft_treshold(Li[idx],
				inst.x[idx] - Li[idx] * partialDetivative) - inst.x[idx];
	}

	static inline D compute_fast_objective(const ProblemData<L, D> &part,
			const std::vector<D> &residuals) {
		D resids = 0;
		D sumx = 0;
		for (L i = 0; i < part.m; i++) {
			resids += log(1 + exp(residuals[i]));
		}
		for (L j = 0; j < part.n; j++) {
			sumx += abs(part.x[j]);
		}
		return 0.5 * part.lambda * resids + sumx;
	}

	static inline void compute_reciprocal_lipschitz_constants(
			const ProblemData<L, D> &inst, std::vector<D> &h_Li) {
		for (unsigned int i = 0; i < inst.n; i++) {
			h_Li[i] = 0;
			for (unsigned int j = inst.A_csc_col_ptr[i];
					j < inst.A_csc_col_ptr[i + 1]; j++) {
				D tmp = inst.A_csc_values[j] * inst.b[inst.A_csc_row_idx[j]];
				h_Li[i] += tmp * tmp;
			}
			if (h_Li[i] > 0) //TODO Check is there should be "4" or not!!!
				h_Li[i] = 8 / (inst.sigma * inst.lambda * h_Li[i]); // Compute reciprocal Lipschitz Constants
		}
	}
};

#endif /* LOGISTIC_LOSS_H */
