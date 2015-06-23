/*
 * HingeLossCD.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef HINGELOSSCD_H_
#define HINGELOSSCD_H_

#include "HingeLoss.h"

template<typename L, typename D>

class HingeLossCD: public HingeLoss<L, D> {
public:
	HingeLossCD() {

	}
	virtual void init(ProblemData<L, D> & instance) {

		instance.Li.resize(instance.n);

		for (L idx = 0; idx < instance.n; idx++) {
			D norm = cblas_l2_norm(
					instance.A_csr_row_ptr[idx + 1]
							- instance.A_csr_row_ptr[idx],
					&instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
			instance.Li[idx] = 1 / (norm * norm * instance.penalty);
		}

	}

	virtual void SDCA(ProblemData<L, D> &instance,
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

			D part = (instance.lambda * instance.total_n)
					* (1.0 - instance.b[idx] * dotProduct) * instance.Li[idx];

			deltaAl =
					(part > 1 - alphaI) ?
							1 - alphaI : (part < -alphaI ? -alphaI : part);
			deltaAlpha[idx] += deltaAl;
			for (L i = instance.A_csr_row_ptr[idx];
					i < instance.A_csr_row_ptr[idx + 1]; i++) {

				deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
						* instance.A_csr_values[i] * deltaAl * instance.b[idx];

			}

		}

	}

	virtual void fast_SDCA(ProblemData<L, D> &instance,
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

			D part = (instance.lambda * instance.total_n)
					* (1.0 - instance.b[idx] * dotProduct) * instance.Li[idx];

			deltaAl =
					(part > 1 - alphaI) ?
							1 - alphaI : (part < -alphaI ? -alphaI : part);
			deltaAlpha[idx] += deltaAl;
			for (L i = instance.A_csr_row_ptr[idx];
					i < instance.A_csr_row_ptr[idx + 1]; i++) {

				deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
						* instance.A_csr_values[i] * deltaAl * instance.b[idx];

			}

		}

	}

};

#endif /* HINGELOSSCD_H_ */
