/*
 * LogisticLossCD.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef LOGISTICLOSSCD_H_
#define LOGISTICLOSSCD_H_

#include "LogisticLoss.h"
template<typename L, typename D>
class LogisticLossCD :   public LogisticLoss<L, D>{
public:
	LogisticLossCD(){

	}

	virtual void solveLocalProblem(ProblemData<L, D> &instance,
				std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW,
				DistributedSettings & distributedSettings){
		for (unsigned int it = 0; it < distributedSettings.iterationsPerThread;
					it++) {

				L idx = rand() / (0.0 + RAND_MAX) * instance.n;

		// compute "delta alpha" = argmin

				D dotProduct = 0;
				for (L i = instance.A_csr_row_ptr[idx];
						i < instance.A_csr_row_ptr[idx + 1]; i++) {

					dotProduct += (w[instance.A_csr_col_idx[i]]
							+ deltaW[instance.A_csr_col_idx[i]])
							* instance.A_csr_values[i];

				}
				//cout<<deltaAlpha[idx];

				D alphaI = instance.x[idx] + deltaAlpha[idx];

				D norm = cblas_l2_norm(
						instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
						&instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);

				dotProduct = instance.b[idx] * dotProduct;

				D deltaAl = 0.0;
				D epsilon = 1e-5;
				if (instance.b[idx] == 1.0)
				{
				        if (alphaI == 0) {deltaAl = 0.5;}
				        D FirstDerivative = 1.0 * deltaAl * instance.oneOverLambdaN * norm * norm
						  + dotProduct - log(1.0 - alphaI - deltaAl) + log(alphaI + deltaAl);

					while (FirstDerivative > epsilon || FirstDerivative < -epsilon)
					{
						D SecondDerivative = 1.0 * norm * norm * instance.oneOverLambdaN
								+ 1.0 / (1.0 - alphaI - deltaAl) + 1.0 / (alphaI + deltaAl);
						deltaAl = 1.0 * deltaAl - FirstDerivative / SecondDerivative;
						deltaAl = (deltaAl > 1 - alphaI) ? 1 - alphaI - 1e-15 : (deltaAl < -alphaI ? -alphaI + 1e-15 : deltaAl);
						FirstDerivative = 1.0 * deltaAl * instance.oneOverLambdaN * norm * norm
							  + dotProduct - log(1.0 - alphaI - deltaAl) + log(alphaI + deltaAl);
					}
				        //cout<<deltaAl+alphaI<<"  ";
					//cout<<FirstDerivative<<"  ";
				}

				else if (instance.b[idx] == -1.0)
				    {
		   		        if(alphaI == 0) {deltaAl = -0.5;}
				        D FirstDerivative = 1.0 * deltaAl * instance.oneOverLambdaN * norm * norm
							+ dotProduct + log(1.0 + alphaI + deltaAl) - log(-1.0 * alphaI - deltaAl);

					while (FirstDerivative > epsilon || FirstDerivative < -epsilon)
					{
						D SecondDerivative = norm * norm * instance.oneOverLambdaN
								+ 1.0 / (1.0 + alphaI + deltaAl) - 1.0 / (alphaI + deltaAl);
						deltaAl = 1.0 * deltaAl - FirstDerivative / SecondDerivative;
						deltaAl = (deltaAl > -alphaI) ? -alphaI - 1e-15 : (deltaAl < -1.0 - alphaI ? -1.0 - alphaI + 1e-15: deltaAl);
						FirstDerivative = 1.0* deltaAl * instance.oneOverLambdaN * norm * norm
								+ dotProduct + log(1.0 + alphaI + deltaAl) - log(-1.0 * alphaI - deltaAl);
						//if(idx==52) cout<<idx<<"  1  "<<deltaAl<<"  2  "<<FirstDerivative<<"  3  "<<SecondDerivative<<"  5  "<<alphaI<<"  6  "<<log(1.0+alphaI+deltaAl)<<endl;
					}
					//cout<<deltaAl+alphaI<<"  ";
					//cout<<FirstDerivative<<"  ";
				}
				//if (isnan(deltaAl)) {cout<<deltaAl<<" 1 "<<alphaI<<"  2  "<<instance.b[idx]<<"  3  "<<idx<<endl;}
				//cout<<deltaAl<<"  ";
				//cout<<idx<<"  ";
				deltaAlpha[idx] += deltaAl;
				//L mm=1;
				for (L i = instance.A_csr_row_ptr[idx];
						i < instance.A_csr_row_ptr[idx + 1]; i++) {
				        //cout<<deltaW[instance.A_csr_col_idx[i]]<<"  ";

				        D tmd =  instance.oneOverLambdaN * instance.A_csr_values[i] * deltaAl * instance.b[idx];
					//if(fabs(tmd) <eps) tmd = 0;
					//D ctmd = deltaW[instance.A_csr_col_idx[i]];
					deltaW[instance.A_csr_col_idx[i]] += tmd;
		     			/*if (isnan(tmd) && mm==1)
					  {
					    mm=0;
					    cout<<tmd<<"  "<<deltaAl <<"      "<<idx <<"           "<<instance.A_csr_col_idx[i]<<endl;
					    }*/
					//cout <<instance.A_csr_col_idx[i]<<"  "<<idx<<endl;}
					//if (instance.A_csr_col_idx[i] == 5) cout<< tmd<<"  "<<deltaW[5]<<endl;
					//cout<<deltaW[instance.A_csr_col_idx[i]]<<"      ";
					//cout<<setprecision (16) <<tmd<<"  ";
				}
			}
	}

};

#endif /* LOGISTICLOSSCD_H_ */
