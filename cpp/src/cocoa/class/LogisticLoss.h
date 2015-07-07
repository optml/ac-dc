/*
 * LogisticLoss.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef LOGISTICLOSS_H_
#define LOGISTICLOSS_H_


#include "LossFunction.h"

template<typename L, typename D>
class LogisticLoss : public LossFunction<L, D> {
public:

	LogisticLoss(){

	}

	virtual void computeObjectiveValue(ProblemData<L, D> & instance,
				mpi::communicator & world, std::vector<D> & w, double &finalDualError,
				double &finalPrimalError){
		D localError = 0;
			for (unsigned int i = 0; i < instance.n; i++) {
				D tmp = 0;
				//if(instance.b[i]*instance.x[i]<=0)cout<<instance.b[i]<<"  "<<instance.x[i]<<endl;
				if (instance.b[i] == -1.0){
				  //cout<<instance.x[i]<<endl;
					if (instance.x[i] < 0){
						tmp += -instance.x[i] * log(-instance.x[i]) ;
					}
					if (instance.x[i] > -1){
						tmp += (1 + instance.x[i]) * log(1 + instance.x[i]);
					}

				}
				if (instance.b[i] == 1.0){
				  //cout<<instance.x[i]<<"   ";
				        if (instance.x[i] > 0){
						tmp += instance.x[i] * log(instance.x[i]) ;
					}
					if (instance.x[i] < 1){
						tmp += (1 - instance.x[i]) * log(1 - instance.x[i]);
					}
				}
				//if (tmp>0) cout<<tmp<<endl;
				localError += tmp;
			}
			//cout<<localError<<" ";

			D localLogisticLoss = 0;
			for (unsigned int idx = 0; idx < instance.n; idx++) {
				D dotProduct = 0;
				for (L i = instance.A_csr_row_ptr[idx];
						i < instance.A_csr_row_ptr[idx + 1]; i++) {
					dotProduct += (w[instance.A_csr_col_idx[i]])
							* instance.A_csr_values[i];
				}

				D tmp = -1.0 * instance.b[idx] * instance.b[idx] * dotProduct;
				localLogisticLoss += log(1 + exp(tmp));

			}
			//cout<<localLogisticLoss<<"  ";
			finalPrimalError = 0;
			vall_reduce(world, &localLogisticLoss, &finalPrimalError, 1);

			finalDualError = 0;
			vall_reduce(world, &localError, &finalDualError, 1);

			D tmp2 = cblas_l2_norm(w.size(), &w[0], 1);
			finalDualError = 1.0 / instance.total_n * finalDualError
					+ 0.5 * instance.lambda * tmp2 * tmp2;
			finalPrimalError =  1.0 / instance.total_n * finalPrimalError
				+ 0.5 * instance.lambda * tmp2 * tmp2;
			//for(unsigned int i=0; i<w.size();i++){cout<<w[i]<<"  ";}
		}

};

#endif /* LOGISTICLOSS_H_ */
