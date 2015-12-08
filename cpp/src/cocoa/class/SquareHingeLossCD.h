/*
 * SquareHingeLossCD.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef SQUAREHINGELOSSCD_H_
#define SQUAREHINGELOSSCD_H_
#include "SquaredHingeLoss.h"

template<typename L, typename D>
class SquareHingeLossCD : public SquaredHingeLoss<L,D>{
public:
	SquareHingeLossCD(){

	}


	virtual void init(ProblemData<L, D> & instance) {

		instance.Li.resize(instance.n);

		for (L idx = 0; idx < instance.n; idx++) {
			D norm = cblas_l2_norm(
					instance.A_csr_row_ptr[idx + 1]
							- instance.A_csr_row_ptr[idx],
					&instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
			instance.Li[idx] =   1/(norm * norm * instance.penalty* instance.oneOverLambdaN + 0.5);
		}

	}


	virtual void subproblem_solver_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
			std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
			mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {
		
			double start = 0;
			double finish = 0;
			double elapsedTime = 0;

			for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {

				start = gettime_();

				for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
					cblas_set_to_zero(deltaW);
					cblas_set_to_zero(deltaAlpha);

					for (unsigned int it = 0; it < distributedSettings.iterationsPerThread;	it++) {

						L idx = rand() / (0.0 + RAND_MAX) * instance.n;
	// compute "delta alpha" = argmin
						D dotProduct = 0;
						for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
							dotProduct += (w[instance.A_csr_col_idx[i]]
								+ instance.penalty*deltaW[instance.A_csr_col_idx[i]])
								* instance.A_csr_values[i];
						}

						D alphaI = instance.x[idx] + deltaAlpha[idx];
						D deltaAl = 0; // FINISH

//			D norm = cblas_l2_norm(
//					instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
//					&instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
						deltaAl = (1.0 - 0.5 * alphaI - instance.b[idx] * dotProduct) *instance.Li[idx];
						deltaAl = (deltaAl < -alphaI) ? -alphaI : deltaAl;

						deltaAlpha[idx] += deltaAl;
						for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {

							deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
								* instance.A_csr_values[i] * deltaAl * instance.b[idx];
						}
					}
					vall_reduce(world, deltaW, wBuffer);
					cblas_sum_of_vectors(w, wBuffer, gamma);
					cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);
				}
				double primalError;
				double dualError;

				finish = gettime_();
				elapsedTime += finish - start;

				this->computeObjectiveValue(instance, world, w, dualError, primalError);

				if (ctx.settings.verbose) {
					cout << "Iteration " << t << " elapsed time " << elapsedTime << "  error " << primalError << "    "
				    	 << dualError << "    " << primalError + dualError << endl;

					logFile << t << "," << elapsedTime << "," << primalError << "," << dualError << ","
				        << primalError + dualError << endl;
				}

			}
	}
};

#endif /* SQUAREHINGELOSSCD_H_ */
