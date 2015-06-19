/*
 * This file contains an functions for experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */

/*
 *  Created on: 19 Jan 2012
 *      Author: jmarecek and taki
 */

#ifndef HINGE_DUAL_LOSS_H
#define HINGE_DUAL_LOSS_H

#include "loss_abstract.h"

struct loss_hinge_dual: public loss_traits {
};

/*******************************************************************/
// a partial specialisation for square loss
template<typename L, typename D>
class Losses<L, D, loss_hinge_dual> {

public:

	static inline void set_residuals_for_zero_x(const ProblemData<L, D> &inst,
			std::vector<D> &residuals) {
		for (L i = 0; i < inst.m; i++) {
			residuals[i] = 0;
		}
	}

	static inline void bulkIterations(ProblemData<L, D> &part,
			std::vector<D> &residuals) {
		for (int i = 0; i < part.m; i++) {
			residuals[i] = 0; //FIXME use CBLAS
		}
		const D delta = 1 / (part.lambda * part.n + 0.0);
		for (L sample = 0; sample < part.n; sample++) {
			if (part.x[sample] > 1)
				part.x[sample] = 1;
			if (part.x[sample] < 0)
				part.x[sample] = 0;
			for (L tmp = part.A_csr_row_ptr[sample];
					tmp < part.A_csr_row_ptr[sample + 1]; tmp++) {
				residuals[part.A_csr_col_idx[tmp]] += delta * part.b[sample]
						* part.A_csr_values[tmp] * part.x[sample];
			}
		}
	}
	static inline void bulkIterations(const ProblemData<L, D> &part,
			std::vector<D> &residuals, std::vector<D> &x) {

		for (int i = 0; i < part.m; i++) {
			residuals[i] = 0; //FIXME use CBLAS
		}
		const D delta = 1 / (part.lambda * part.n + 0.0);
		for (L sample = 0; sample < part.n; sample++) {
			for (L tmp = part.A_csr_row_ptr[sample];
					tmp < part.A_csr_row_ptr[sample + 1]; tmp++) {
				residuals[part.A_csr_col_idx[tmp]] = delta * part.b[sample]
						* part.A_csr_values[tmp] * x[sample];
			}
		}

	}
	static inline void bulkIterations_for_my_instance_data(
			const ProblemData<L, D> &inst, std::vector<D> &residuals) {
		for (L row = 0; row < inst.m; row++) {
			residuals[row] = -inst.b[row];
			for (L col_tmp = inst.A_csr_row_ptr[row];
					col_tmp < inst.A_csr_row_ptr[row + 1]; col_tmp++) {
				residuals[row] += inst.A_csr_values[col_tmp]
						* inst.x[inst.A_csr_col_idx[col_tmp]];
			}
		}
	}

	static inline D do_single_iteration_serial(const ProblemData<L, D> &inst,
			const L idx, std::vector<D> &residuals, std::vector<D> &x,
			const std::vector<D> &Li) {
		D tmp = 0;
		tmp = compute_update(inst, residuals, idx, Li);
		x[idx] += tmp;
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			residuals[inst.A_csc_row_idx[j]] += tmp * inst.A_csc_values[j];
		}
		return abs(tmp);
	}

	static inline D do_single_iteration_parallel(const ProblemData<L, D> &inst,
			const L idx, std::vector<D> &residuals, std::vector<D> &x,
			const std::vector<D> &Li) {

		D tmp = 0;
		for (L i = inst.A_csr_row_ptr[idx]; i < inst.A_csr_row_ptr[idx + 1];
				i++) {
			tmp += inst.A_csr_values[i] * residuals[inst.A_csr_col_idx[i]];
		}
		tmp = (1 - tmp * inst.b[idx]) * inst.lambda * inst.n * Li[idx];




		if (tmp < -x[idx]) {
			tmp = -x[idx];
		} else if (tmp > 1 - x[idx]) {
			tmp = 1 - x[idx];
		}

		parallel::atomic_add(x[idx], tmp);
		const D delta = 1 / (inst.lambda * inst.n);
		for (L j = inst.A_csr_row_ptr[idx]; j < inst.A_csr_row_ptr[idx + 1];
				j++) {
			parallel::atomic_add(residuals[inst.A_csr_col_idx[j]],
					tmp * inst.A_csr_values[j] * inst.b[idx] * delta);
		}

		return abs(tmp);
	}

	static inline D do_single_iteration_parallel_for_distributed(
			const ProblemData<L, D> &inst, const L idx,
			std::vector<D> &residuals, std::vector<D> &x,
			const std::vector<D> &Li, D* residual_updates) {
		D tmp = 0;
		tmp = compute_update(inst, residuals, idx, Li);
		parallel::atomic_add(x[idx], tmp);
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(residual_updates[inst.A_csc_row_idx[j]],
					tmp * inst.A_csc_values[j]);
		}
		return abs(tmp);
	}

	static inline D do_single_iteration_parallel_for_distributed(
			const ProblemData<L, D> &inst,
			const ProblemData<L, D> &inst_local, const L idx,
			std::vector<D> &residuals, std::vector<D> &residuals_local,
			std::vector<D> &x, const std::vector<D> &Li, D* residual_updates) {

		D tmp = 0;
		for (L i = inst.A_csr_row_ptr[idx]; i < inst.A_csr_row_ptr[idx + 1];
				i++) {
			tmp += inst.A_csr_values[i] * residuals[inst.A_csr_col_idx[i]];
		}
		tmp = (1 - tmp * inst.b[idx]) * inst.lambda * inst.total_n * Li[idx];

		if (tmp < -x[idx]) {
			tmp = -x[idx];
		} else if (tmp > 1 - x[idx]) {
			tmp = 1 - x[idx];
		}

		parallel::atomic_add(x[idx], tmp);
		const D delta = 1 / (inst.lambda * inst.total_n);
		for (L j = inst.A_csr_row_ptr[idx]; j < inst.A_csr_row_ptr[idx + 1];
				j++) {
			parallel::atomic_add(residual_updates[inst.A_csr_col_idx[j]],
					tmp * inst.A_csr_values[j] * inst.b[idx] * delta);
		}

//		for (unsigned int j = inst_local.A_csc_col_ptr[idx];
//				j < inst_local.A_csc_col_ptr[idx + 1]; j++) {
//			parallel::atomic_add(residuals_local[inst_local.A_csc_row_idx[j]],
//					tmp * inst_local.A_csc_values[j]);
//		}

		return abs(tmp);
	}

	static inline D compute_update(const ProblemData<L, D> &inst,
			const std::vector<D> &residuals, const L idx,
			const std::vector<D> &Li) {
		D tmp = 0; //compute partial derivative f_idx'(x)
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			tmp += inst.A_csc_values[j] * residuals[inst.A_csc_row_idx[j]];
		}
		tmp = compute_soft_treshold(Li[idx] * inst.lambda,
				inst.x[idx] - Li[idx] * tmp) - inst.x[idx];
		return tmp;
	}

	static inline D compute_update(const ProblemData<L, D> &inst,
			const ProblemData<L, D> &inst_local,
			const std::vector<D> &residuals,
			const std::vector<D> &residuals_local, const L idx,
			const std::vector<D> &Li) {
		D tmp = 0; //compute partial derivative f_idx'(x)
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			tmp += inst.A_csc_values[j] * residuals[inst.A_csc_row_idx[j]];
		}
		for (unsigned int j = inst_local.A_csc_col_ptr[idx];
				j < inst_local.A_csc_col_ptr[idx + 1]; j++) {
			tmp += inst_local.A_csc_values[j]
					* residuals_local[inst_local.A_csc_row_idx[j]];
		}
		tmp = compute_soft_treshold(Li[idx] * inst.lambda,
				inst.x[idx] - Li[idx] * tmp) - inst.x[idx];
		return tmp;
	}

	static inline D compute_fast_objective(ProblemData<L, D> &part,
			const std::vector<D> &residuals) {
		D resids = 0;
		D sumx = 0;
		D sumLoss = 0;
		for (L i = 0; i < part.m; i++) {
			resids += residuals[i] * residuals[i];
		}
		L good = 0;
		part.dualObjective = 0;
		for (L j = 0; j < part.n; j++) {
			D error = 0;
			for (L i = part.A_csr_row_ptr[j]; i < part.A_csr_row_ptr[j + 1];
					i++) {
				error += part.A_csr_values[i]
						* residuals[part.A_csr_col_idx[i]];
			}
			if (part.b[j] * error >  0) {
				good++;
			}
			error = 1 - part.b[j] * error;
			if (error < 0) {
				error = 0;
			}
			sumLoss += error  ;
			sumx+= part.x[j];
		}


		part.primalObjective = part.lambda * 0.5 * resids
				+ (sumLoss) / (0.0 + part.n);
		part.dualObjective = -part.lambda * 0.5 * resids
				+ sumx / (0.0 + part.n);

		part.oneZeroAccuracy = good / (0.0 + part.n);
		//		D residOut = 0;
		//		reduce(world, resids, residOut, std::plus<D>(), 0);
		//		cout << "XXXXXXXXXX   " << 1 / (0.0 + part.total_n) * sumxOut << endl;
		//		return part.lambda * resids + 1 / (0.0 + part.m) * sumxOut;
		return part.primalObjective -part.dualObjective;
	}

	static inline void compute_reciprocal_lipschitz_constants(
			const ProblemData<L, D> &inst, std::vector<D> &h_Li) {
		for (L i = 0; i < inst.n; i++) {
			h_Li[i] = 0;
			for (L j = inst.A_csr_row_ptr[i]; j < inst.A_csr_row_ptr[i + 1];
					j++) {
				h_Li[i] += inst.A_csr_values[j] * inst.A_csr_values[j];
			}
			if (h_Li[i] > 0)
				h_Li[i] = 1 / (inst.sigma * h_Li[i]); // Compute reciprocal Lipschitz Constants
		}
	}

	static inline void compute_reciprocal_lipschitz_constants(
			const ProblemData<L, D> &inst,
			const ProblemData<L, D> &inst_local, std::vector<D> &h_Li) {
#pragma omp parallel for
		for (unsigned int i = 0; i < inst.n; i++) {
			h_Li[i] = 0;
			for (unsigned int j = inst.A_csr_row_ptr[i];
					j < inst.A_csr_row_ptr[i + 1]; j++) {
				h_Li[i] += inst.A_csr_values[j] * inst.A_csr_values[j];
			}
			for (unsigned int j = inst_local.A_csr_row_ptr[i];
					j < inst_local.A_csr_row_ptr[i + 1]; j++) {
				h_Li[i] += inst_local.A_csr_values[j]
						* inst_local.A_csr_values[j];
			}
			if (h_Li[i] > 0)
				h_Li[i] = 1 / (inst.sigma * h_Li[i]); // Compute reciprocal Lipschitz Constants
		}
	}

};

#endif /* SQUARE_LOSS_H */
