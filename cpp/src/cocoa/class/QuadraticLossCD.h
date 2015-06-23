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
		instance.vi.resize(instance.n);

		for (L idx = 0; idx < instance.n; idx++) {
			D norm = cblas_l2_norm(
					instance.A_csr_row_ptr[idx + 1]
							- instance.A_csr_row_ptr[idx],
					&instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
			instance.Li[idx] = 1. / (norm * norm * instance.penalty * instance.oneOverLambdaN + 1);

			instance.vi[idx] = norm * norm;
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

//			D denom = norm * norm * instance.penalty * instance.oneOverLambdaN
//					+ 1;

			deltaAl = (instance.b[idx] - alphaI - instance.b[idx] * dotProduct) * instance.Li[idx];

			deltaAlpha[idx] += deltaAl;
			for (L i = instance.A_csr_row_ptr[idx];
					i < instance.A_csr_row_ptr[idx + 1]; i++) {

				deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
						* instance.A_csr_values[i] * deltaAl * instance.b[idx];

			}

		}
	}


	virtual void fast_SDCA(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW, std::vector<D> &zk, std::vector<D> &uk,
			std::vector<D> &yk, std::vector<D> &deltayk, std::vector<D> &Ayk, std::vector<D> &deltaAyk, D &theta, DistributedSettings & distributedSettings) {

		D tk = 0;
		D thetasquare = theta * theta;

		for (unsigned int it = 0; it < distributedSettings.iterationsPerThread;
				it++) {

			L idx = rand() / (0.0 + RAND_MAX) * instance.n;

			D dotProduct = 0;
			for (L i = instance.A_csr_row_ptr[idx];
					i < instance.A_csr_row_ptr[idx + 1]; i++) {
				dotProduct += (Ayk[instance.A_csr_col_idx[i]]
						+ instance.penalty * deltaAyk[instance.A_csr_col_idx[i]])
						* instance.A_csr_values[i];
			}

			tk = (-dotProduct * instance.b[idx] + instance.b[idx] - zk[idx] ) //- thetasquare * uk[idx])
					/ (instance.vi[idx] * instance.n /distributedSettings.iterationsPerThread
							* theta * instance.penalty * instance.oneOverLambdaN + 1.);

			zk[idx] += tk;
			uk[idx] -= (1.0 - theta * instance.n /distributedSettings.iterationsPerThread) / (thetasquare) * tk;

			D deltaAl = thetasquare * uk[idx] + zk[idx] - instance.x[idx] - deltaAlpha[idx];
			deltaAlpha[idx] += deltaAl;
			instance.x[idx] = theta * theta * uk[idx] + zk[idx];
			//cout <<idx<<"          "<< theta << "    "<<uk[idx] << "    " <<zk[idx] <<endl;

			for (L i = instance.A_csr_row_ptr[idx];
					i < instance.A_csr_row_ptr[idx + 1]; i++) {

				deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
						* instance.A_csr_values[i] * deltaAl* instance.b[idx];
			}

			D thetanext = 0;
			thetanext = 0.5 * sqrt(thetasquare * thetasquare + 4 * thetasquare) - 0.5 * thetasquare;
			D dyk = thetanext * thetanext * uk[idx] + zk[idx] - yk[idx] - deltayk[idx];
			deltayk[idx] += dyk;
			yk[idx] = thetanext * thetanext * uk[idx] + zk[idx];

			for (L i = instance.A_csr_row_ptr[idx];
								i < instance.A_csr_row_ptr[idx + 1]; i++) {

				deltaAyk[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
									* instance.A_csr_values[i] * dyk* instance.b[idx];
			}

		}
	}

	virtual void fulldescent(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &w,
			std::vector<D> &deltaW, D &dualobj, DistributedSettings & distributedSettings) {

		//std::vector<D> Aalpha(instance.m);
		//cblas_set_to_zero(Aalpha);
		//matrixvector(instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr ,instance.x, instance.m, Aalpha);

		std::vector<D> gradient(instance.n);
		cblas_set_to_zero(gradient);
		vectormatrix(w, instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr , instance.m, gradient);

		for (L i = 0; i < instance.n; i++){
			gradient[i] = 1.0 / instance.total_n * gradient[i]
					+ 1.0 / instance.total_n * (instance.x[i] - 0.5 * instance.b[i]);
		}

		D rho = 0.5;
		D c1ls = 0.8;
		D a = 10.0;
		L iter = 0;
		std::vector<D> potent(instance.n);


		while (1){
			if (iter > 1000){
				a = 0.01;
				break;
			}
			for (L i = 0; i < instance.n; i++)
				potent[i] = instance.x[i] - a * gradient[i];

			D obj = 0;
				for (unsigned int i = 0; i < instance.n; i++) {
					D tmp = 0.5 * potent[i] * potent[i] - potent[i] * instance.b[i];
					obj += tmp;
				}
			std::vector<D> Apotent(instance.m);
			cblas_set_to_zero(Apotent);
			matrixvector(instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr ,
						potent, instance.m, Apotent);
			D aQa = cblas_l2_norm(instance.m, &Apotent[0], 1);
			obj = 1.0 / instance.total_n * obj + 0.5 * instance.oneOverLambdaN / instance.total_n * aQa * aQa;
			D gg;
			gg = cblas_l2_norm(instance.n, &gradient[0], 1);

			if (obj <= dualobj - c1ls * a * gg * gg){
//cout<<a<<endl;
				for (L idx = 0; idx < instance.n; idx++){
					instance.x[idx] = potent[idx];
					//deltaAlpha[idx] = - a * gradient[idx];
					//for (L i = instance.A_csr_row_ptr[idx];	i < instance.A_csr_row_ptr[idx + 1]; i++)
						//deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
							//								* instance.A_csr_values[i] * deltaAlpha[idx];
				}
				matrixvector(instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr ,
							instance.x, instance.m, deltaW);
				for (L idx = 0; idx < instance.m; idx++)
					deltaW[idx] = instance.oneOverLambdaN * deltaW[idx];
				dualobj = obj;
				break;
			}
			a = rho * a;
			iter += 1;
		}

	}



};

#endif /* QUADRATICLOSSCD_H_ */
