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
	virtual ~QuadraticLossCD() {}

	virtual void init(ProblemData<L, D> & instance) {

		instance.Li.resize(instance.n);
		instance.vi.resize(instance.n);

		for (L idx = 0; idx < instance.n; idx++) {
			D norm = cblas_l2_norm(
					instance.A_csr_row_ptr[idx + 1]
										   - instance.A_csr_row_ptr[idx],
										   &instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
			instance.Li[idx] = 1.0 / (norm * norm * instance.penalty * instance.oneOverLambdaN + 1.0);

			instance.vi[idx] = norm * norm;
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

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread; it++) {
					L idx = rand() / (0.0 + RAND_MAX) * instance.n;
					// compute "delta alpha" = argmin
					D dotProduct = 0;
					for (L i = instance.A_csr_row_ptr[idx];
							i < instance.A_csr_row_ptr[idx + 1]; i++) {
						dotProduct += (w[instance.A_csr_col_idx[i]]
										 + 1.0 * instance.penalty * deltaW[instance.A_csr_col_idx[i]])
																										* instance.A_csr_values[i];
					}
					D alphaI = instance.x[idx] + deltaAlpha[idx];
					D deltaAl = 0; // FINISH
					deltaAl = (1.0 * instance.b[idx] - alphaI - dotProduct * instance.b[idx]) * instance.Li[idx];
					deltaAlpha[idx] += deltaAl;
					for (L i = instance.A_csr_row_ptr[idx];
							i < instance.A_csr_row_ptr[idx + 1]; i++)
						deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * deltaAl * instance.A_csr_values[i] * instance.b[idx];

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
				cout << "Iteration " << t << " elapsed time " << elapsedTime
						<< "  error " << primalError << "    " << dualError
						<< "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
						<< dualError << "," << primalError + dualError << endl;

			}
		}

	}

	virtual void subproblem_solver_accelerated_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
			std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
			mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;

		double theta = 1.0 * distributedSettings.iterationsPerThread / instance.n;
		std::vector<double> zk(instance.n);
		std::vector<double> uk(instance.n);
		std::vector<double> Ayk(instance.m);
		std::vector<double> yk(instance.n);
		std::vector<double> deltayk(instance.n);
		std::vector<double> deltaAyk(instance.m);
		cblas_set_to_zero(uk);
		cblas_set_to_zero(yk);
		cblas_set_to_zero(Ayk);
		std::vector<double> AykBuffer(instance.m);

		for (unsigned int i = 0; i < instance.n; i++)
			zk[i] = instance.x[i];

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();
			for (int jj = 0; jj < 10*distributedSettings.iters_bulkIterations_count; jj++) {

				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);
				cblas_set_to_zero(deltayk);
				cblas_set_to_zero(deltaAyk);

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread; it++)
					this->accelerated_SDCA_oneIteration(instance, deltaAlpha, w, deltaW,
							zk, uk, yk, deltayk, Ayk, deltaAyk, theta, distributedSettings);

				double thetasq = theta * theta;
				theta = 0.5 * sqrt(thetasq * thetasq + 4 * thetasq) - 0.5 * thetasq;
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);
				cblas_sum_of_vectors(yk, deltayk, gamma);

				vectormatrix_b(instance.x, instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr,
						instance.b, instance.oneOverLambdaN, instance.n, deltaW);

				vectormatrix_b(yk, instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr,
						instance.b, instance.oneOverLambdaN, instance.n, deltaAyk);

				cblas_set_to_zero(w);
				cblas_set_to_zero(Ayk);

				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				vall_reduce(world, deltaAyk, AykBuffer);
				cblas_sum_of_vectors(Ayk, AykBuffer, gamma);

			}
			double primalError;
			double dualError;

			finish = gettime_();
			elapsedTime += finish - start;

			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime
						<< "  error " << primalError << "    " << dualError
						<< "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
						<< dualError << "," << primalError + dualError << endl;
			}


		}
	}


	virtual void accelerated_SDCA_oneIteration(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW, std::vector<D> &zk, std::vector<D> &uk,
			std::vector<D> &yk, std::vector<D> &deltayk, std::vector<D> &Ayk, std::vector<D> &deltaAyk,
			D &theta, DistributedSettings & distributedSettings) {

		D thetasquare = theta * theta;
		L idx = rand() / (0.0 + RAND_MAX) * instance.n;

		D dotProduct = 0;
		for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
			dotProduct += (1.0 * Ayk[instance.A_csr_col_idx[i]] + instance.penalty * deltaAyk[instance.A_csr_col_idx[i]])
																										* instance.A_csr_values[i];
		}
		//matrixvector(instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr ,
		//		yk, instance.m, Apotent);

		D tk = ( 1.0 * instance.b[idx] - zk[idx] - dotProduct * instance.b[idx] ) //- thetasquare * uk[idx])
													/ (instance.vi[idx] * instance.n /distributedSettings.iterationsPerThread
															* theta * instance.penalty * instance.oneOverLambdaN + 1.);
		zk[idx] += tk;
		uk[idx] -= (1.0 - theta * instance.n /distributedSettings.iterationsPerThread) / (thetasquare) * tk;

		D deltaAl = thetasquare * uk[idx] + zk[idx] - instance.x[idx] - deltaAlpha[idx];
		deltaAlpha[idx] += deltaAl;
		//cout<<idx<<"     "<<deltaAlpha[idx]<<endl;
		//instance.x[idx] = theta * theta * uk[idx] + zk[idx];
		//cout <<idx<<"          "<< theta << "    "<<uk[idx] << "    " <<zk[idx] <<endl;

		//			for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
		//				deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * instance.A_csr_values[i] * deltaAl* instance.b[idx];
		//			}

		D thetanext = theta;
		thetanext = 0.5 * sqrt(thetasquare * thetasquare + 4 * thetasquare) - 0.5 * thetasquare;
		D dyk = thetanext * thetanext * uk[idx] + zk[idx] - yk[idx] - deltayk[idx];
		deltayk[idx] += dyk;
		//yk[idx] = thetanext * thetanext * uk[idx] + zk[idx];

		//			for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
		//				deltaAyk[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN * instance.A_csr_values[i] * dyk* instance.b[idx];
		//			}

	}

	virtual void subproblem_solver_steepestdescent(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
			std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
			mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;

		double dualobj = 0;
		std::vector<D> gradient(instance.n);
		D rho = 0.8;
		D c1ls = 0.1;
		D a = 20.00;
		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();
			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {

				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);


				for (L line_search_iter = 0; line_search_iter < 2; line_search_iter++){

					this->compute_subproproblem_gradient(instance, gradient, deltaAlpha, w);
					this->backtrack_linesearch(instance, deltaAlpha, gradient, w, dualobj, rho, c1ls, a, distributedSettings);

				}
				for (unsigned int idx = 0; idx < instance.n; idx++){
					for (unsigned int i = instance.A_csr_row_ptr[idx];	i < instance.A_csr_row_ptr[idx + 1]; i++)
						deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
						* instance.A_csr_values[i] * deltaAlpha[idx]* instance.b[idx];
				}

				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				//cout << deltaW[0]<<"   "<<w[0]<<"   "<<wBuffer[0]<<endl;
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);

			}
			double primalError;
			double dualError;

			finish = gettime_();
			elapsedTime += finish - start;

			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime
						<< "  error " << primalError << "    " << dualError
						<< "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
						<< dualError << "," << primalError + dualError << endl;


			}
		}
	}


	virtual void subproblem_solver_LBFGS(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
			std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
			mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;

		double dualobj = 0;
		int limit_BFGS = 10;
		std::vector<double> old_grad(instance.n);
		std::vector<double> sk(instance.n * limit_BFGS);
		std::vector<double> rk(instance.n * limit_BFGS);
		cblas_set_to_zero(sk);
		cblas_set_to_zero(rk);

		std::vector<D> gradient(instance.n);
		std::vector<D> search_direction(instance.n);
		int flag_BFGS = 0;
		std::vector<D> oneoversy(limit_BFGS);

		D rho = 0.8;
		D c1ls = 0.1;
		D a = 20.0;

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();

			for (int jj = 0; jj < 5; jj++) {

				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);


				for (L iter_counter = 0; iter_counter < 10; iter_counter++){

					this->compute_subproproblem_gradient(instance, gradient, deltaAlpha, w);
					this->LBFGS_update(instance, search_direction, old_grad,
							sk, rk, gradient, oneoversy, iter_counter, limit_BFGS, flag_BFGS);
					this->backtrack_linesearch(instance, deltaAlpha, search_direction, w, dualobj, rho, c1ls, a, distributedSettings);

					for (L idx = 0; idx < instance.n; idx++)
						sk[instance.n * flag_BFGS + idx] = - a * search_direction[idx];

					flag_BFGS += 1;
					if (flag_BFGS == limit_BFGS)
						flag_BFGS = 0;

				}
				for (unsigned int idx = 0; idx < instance.n; idx++){
					for (unsigned int i = instance.A_csr_row_ptr[idx];	i < instance.A_csr_row_ptr[idx + 1]; i++)
						deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
						* instance.A_csr_values[i] * deltaAlpha[idx]* instance.b[idx];
				}

				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);
			}

			finish = gettime_();
			elapsedTime += finish - start;

			double primalError;
			double dualError;
			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime
						<< "  error " << primalError << "    " << dualError
						<< "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
						<< dualError << "," << primalError + dualError << endl;

			}
		}

	}
};

#endif /* QUADRATICLOSSCD_H_ */


