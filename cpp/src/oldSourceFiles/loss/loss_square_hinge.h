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

#ifndef SQUARE_HINGE_LOSS_H
#define SQUARE_HINGE_LOSS_H

#include "loss_abstract.h"


struct square_hinge_loss_traits: public loss_traits {
};

/*******************************************************************/
// a partial specialiasation for square hinge loss
template<typename L, typename D>
class Losses<L, D, square_hinge_loss_traits> {

public:

	static inline void set_residuals_for_zero_x(const ProblemData<L, D> &inst,
			std::vector<D> &residuals) {
		for (L i = 0; i < inst.m; i++) {
			residuals[i] = 0;
		}
	}

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

	static inline D do_single_iteration_serial(const ProblemData<L, D> &inst,
			const L &idx, std::vector<D> &residuals, std::vector<D> &x,
			const std::vector<D> &Li) {
		D tmp = 0;
		tmp = compute_update(inst, residuals, idx, Li);
		x[idx] += tmp;
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			residuals[inst.A_csc_row_idx[j]] += -tmp
					* inst.b[inst.A_csc_row_idx[j]] * inst.A_csc_values[j];
		}
		return abs(tmp);
	}

	static inline D do_single_iteration_parallel(const ProblemData<L, D> &inst,
			const L &idx, std::vector<D> &residuals, std::vector<D> &x,
			const std::vector<D> &Li) {
		D tmp = 0;
		tmp = compute_update(inst, residuals, idx, Li);
		parallel::atomic_add(x[idx], tmp);
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(residuals[inst.A_csc_row_idx[j]],
					-tmp * inst.b[inst.A_csc_row_idx[j]]
							* inst.A_csc_values[j]);
		}
		return abs(tmp);
	}

	static inline D do_single_iteration_parallel_for_distributed(
			const ProblemData<L, D> &inst, const L &idx,
			std::vector<D> &residuals, std::vector<D> &x,
			const std::vector<D> &Li, D* &residual_update) {
		D tmp = 0;
		tmp = compute_update(inst, residuals, idx, Li);
		parallel::atomic_add(x[idx], tmp);
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(residual_update[inst.A_csc_row_idx[j]],
					-tmp * inst.b[inst.A_csc_row_idx[j]]
							* inst.A_csc_values[j]);
		}
		return abs(tmp);
	}

	static inline D compute_update(const ProblemData<L, D> &inst,
			const std::vector<D> &residuals, const L &idx,
			const std::vector<D> &Li) {
		D delta = 0;
		D tmp = 0; //compute partial derivative f_idx'(x)
		D partialDetivative = 0;
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			int i = inst.A_csc_row_idx[j];
			if (residuals[i] > -1) {
				partialDetivative += (1 + residuals[i]) * inst.b[i]
						* inst.A_csc_values[j];
			}
		}
		partialDetivative = -inst.lambda * partialDetivative;
		return compute_soft_treshold(Li[idx],
				inst.x[idx] - Li[idx] * partialDetivative) - inst.x[idx];
	}

	static inline D compute_fast_objective(const ProblemData<L, D> &part,
			const std::vector<D> &residuals) {
		D resids = 0;
		D sumx = 0;
		for (L i = 0; i < part.m; i++) {
			D tmp = (residuals[i] + 1);
			if (tmp > 0)
				resids += tmp * tmp;
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
			if (h_Li[i] > 0)
				h_Li[i] = 1 / (inst.lambda * inst.sigma * h_Li[i]); // Compute reciprocal Lipschitz Constants
		}

	}
};
#endif /* SQUARE_HINGE_LOSS_H */
