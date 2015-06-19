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

#ifndef ABSTRACT_LOSS_H
#define ABSTRACT_LOSS_H


#include "../../parallel/parallel_essentials.h"
#include "../../solver/treshhold_functions.h"
#include "../../solver/structures.h"


// "Traits" defining what loss functions to use
struct loss_traits {
};

// An "abstract" template class
// We use only its partial specialisations
template<typename L, typename D, typename Traits>
class Losses {
public:
	static inline void bulkIterations(const ProblemData<L, D> inst, std::vector<D> &residuals) {
	}
	static inline D do_single_iteration_serial(const ProblemData<L, D> inst, const L idx,
	std::vector<D> &residuals, std::vector<D> &x, const std::vector<D> Li) {
		D tmp = 0;
		return abs(tmp);
	}
	static inline D do_single_iteration_parallel(const ProblemData<L, D> inst, const L idx,
	std::vector<D> &residuals, std::vector<D> &x, const std::vector<D> Li) {
		D tmp = 0;
		return abs(tmp);
	}
	static inline D compute_update(const ProblemData<L, D> &inst, const std::vector<D> &residuals,
	const L idx, const std::vector<D> &Li) {
		D delta = 0;
		return delta;
	}
	static inline D compute_fast_objective(const ProblemData<L, D> &part, const std::vector<D> &residuals) {
		D resids = 0;
		return resids;
	}
	static inline void compute_reciprocal_lipschitz_constants(const ProblemData<L, D> &inst,
	std::vector<D> &h_Li) {
	}
};

#endif /* ABSTRACT_LOSS_H */
