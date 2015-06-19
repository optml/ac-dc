/*
 * QuadraticLossCD.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef QUADRATICLOSSCD_H_
#define QUADRATICLOSSCD_H_
#include "QuadraticLoss.h"

template<typename L, typename D>
class QuadraticLossCD: public QuadraticLoss<L, D> {
public:
	QuadraticLossCD() {

	}

	virtual void init(ProblemData<L, D> & instance) {

		instance.Li.resize(instance.n);

		for (L idx = 0; idx < instance.n; idx++) {
			D norm = cblas_l2_norm(
					instance.A_csr_row_ptr[idx + 1]
							- instance.A_csr_row_ptr[idx],
					&instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
			instance.Li[idx] = 1
					/ (norm * norm * instance.penalty * instance.oneOverLambdaN
							+ 1);
		}

	}

	virtual void solveLocalProblem(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &w,
			std::vector<D> &deltaW, DistributedSettings & distributedSettings) {
		for (unsigned int it = 0; it < distributedSettings.iterationsPerThread;
				it++) {

			L idx = rand() / (0.0 + RAND_MAX) * instance.n;

			// compute "delta alpha" = argmin

			D dotProduct = 0;
			for (L i = instance.A_csr_row_ptr[idx];
					i < instance.A_csr_row_ptr[idx + 1]; i++) {

				dotProduct += (w[instance.A_csr_col_idx[i]]
						+ instance.penalty * deltaW[instance.A_csr_col_idx[i]])
						* instance.A_csr_values[i];
			}

			D alphaI = instance.x[idx] + deltaAlpha[idx];

			D deltaAl = 0; // FINISH

//			D norm = cblas_l2_norm(
//					instance.A_csr_row_ptr[idx + 1]
//							- instance.A_csr_row_ptr[idx],
//					&instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);

//			D denom = norm * norm * instance.penalty * instance.oneOverLambdaN
//					+ 1;

			deltaAl = (instance.b[idx] - alphaI - dotProduct) * instance.Li[idx];

			deltaAlpha[idx] += deltaAl;
			for (L i = instance.A_csr_row_ptr[idx];
					i < instance.A_csr_row_ptr[idx + 1]; i++) {

				deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
						* instance.A_csr_values[i] * deltaAl;

			}

		}
	}
};

#endif /* QUADRATICLOSSCD_H_ */
