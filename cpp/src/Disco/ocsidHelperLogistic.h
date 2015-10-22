#ifndef OCSIDHELPERLOGISTIC_H_
#define OCSIDHELPERLOGISTIC_H_


#include <math.h>
#include <gsl/gsl_blas.h> // for BLAS
#include "../solver/distributed/distributed_essentials.h"
#include "QR_solver.h"


void computeObjectiveLogistic(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &obj,
                       boost::mpi::communicator &world) {

	obj = 0.0;
	std::vector<double> w_x(instance.n + 1);
	std::vector<double> w_x_world(instance.n + 1);

	w_x[instance.n] = cblas_l2_norm(w.size(), &w[0], 1) * cblas_l2_norm(w.size(), &w[0], 1);

	for (unsigned int idx = 0; idx < instance.n; idx++) {
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			w_x[idx] += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];
	}
	vall_reduce(world, w_x, w_x_world);

	for (unsigned int idx = 0; idx < instance.n; idx++)
		obj += log(1.0 + exp(-w_x_world[idx] * instance.b[idx]));

	obj = 1.0 / instance.n * obj + 0.5 * instance.lambda * w_x_world[instance.n];

}


void computePrimalAndDualObjectiveLogistic(ProblemData<unsigned int, double> &instance,
        std::vector<double> &alpha, std::vector<double> &w, double &rho, double &finalDualError,
        double &finalPrimalError) {

	double localError = 0;
	for (unsigned int i = 0; i < instance.n; i++) {
		double tmp = 0;
		if (instance.b[i] == -1.0) {
			if (alpha[i] < 0) {
				tmp += -alpha[i] * log(-alpha[i]) ;
			}
			if (alpha[i] > -1) {
				tmp += (1.0 + alpha[i]) * log(1.0 + alpha[i]);
			}

		}
		if (instance.b[i] == 1.0) {
			if (alpha[i] > 0) {
				tmp += alpha[i] * log(alpha[i]) ;
			}
			if (alpha[i] < 1) {
				tmp += (1.0 - alpha[i]) * log(1.0 - alpha[i]);
			}
		}
		localError += tmp;
	}

	double localLogisticLoss = 0;
	for (unsigned int idx = 0; idx < instance.n; idx++) {
		double dotProduct = 0;
		for (unsigned int i = instance.A_csr_row_ptr[idx];
		        i < instance.A_csr_row_ptr[idx + 1]; i++) {
			dotProduct += (w[instance.A_csr_col_idx[i]])
			              * instance.A_csr_values[i];
		}

		double tmp = -1.0 * instance.b[idx] * instance.b[idx] * dotProduct;
		localLogisticLoss += log(1.0 + exp(tmp));

	}
	finalPrimalError = 0;
	finalDualError = 0;

	double tmp2 = cblas_l2_norm(instance.m, &w[0], 1);
	finalDualError = 1.0 / instance.n * localError
	                 + 0.5 * rho * tmp2 * tmp2;
	finalPrimalError =  1.0 / instance.n * localLogisticLoss
	                    + 0.5 * rho * tmp2 * tmp2;

}

void computeGradientLogistic(std::vector<double> &w, std::vector<double> &Aw, std::vector<double> &grad,
                             ProblemData<unsigned int, double> &instance) {

	cblas_set_to_zero(grad);
	double temp = 0.0;
	double w_x = 0.0;

	for (unsigned int idx = 0; idx < instance.n; idx++) {
		temp = exp(-1.0 * Aw[idx]);
		temp = temp / (1.0 + temp) * (-instance.b[idx]) / instance.n;
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			grad[instance.A_csr_col_idx[i]] += temp * instance.A_csr_values[i];
	}

	cblas_daxpy(instance.m, instance.lambda, &w[0], 1, &grad[0], 1);

}


void computeLocalHessianTimesAULogistic(std::vector<double> &w, std::vector<double> &u, std::vector<double> &Au,
                                        std::vector<double> &Hu_local, ProblemData<unsigned int, double> &instance,
                                        boost::mpi::communicator &world) {

	cblas_set_to_zero(Hu_local);
	double temp;
	double scalar;

	std::vector<double> w_x(instance.n);
	std::vector<double> w_x_world(instance.n);

	for (unsigned int idx = 0; idx < instance.n; idx++) {
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			w_x[idx] += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];
	}
	vall_reduce(world, w_x, w_x_world);

	for (unsigned int idx = 0; idx < instance.n; idx++) {
		temp = exp(-w_x_world[idx] * instance.b[idx]);
		scalar = temp / (temp + 1) / (temp + 1) * Au[idx];
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			Hu_local[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * instance.b[idx] * scalar / instance.n;
	}

	for (unsigned int i = 0; i < instance.m; i++)
		Hu_local[i] += instance.lambda * u[i];


}



void distributed_PCGByD_Logistic(std::vector<double> &w, ProblemData<unsigned int, double> &instance,
                                 ProblemData<unsigned int, double> &preConData, double &mu,
                                 std::vector<double> &vk, double &deltak, boost::mpi::communicator &world, int nPartition, int rank,
                                 std::ofstream &logFile) {

	// Compute Matrix P
	// Broadcastw_k
	int flag;
	int innerflag;
	//std::vector<int> flag(10);
	//mpi::request reqs[1];

	double difference;
	double objPre;

	double start = 0;
	double finish = 0;
	double elapsedTime = 0;
	double grad_norm;

	double epsilon;
	double obj;
	double alpha = 0.0;
	double beta = 0.0;
	unsigned int batchSize = preConData.n;

	//std::vector<double> P(instance.m);
	std::vector<double> v(instance.m);
	std::vector<double> s(instance.m);
	std::vector<double> r(instance.m);
	std::vector<double> u(instance.m);
	std::vector<double> Au_local(instance.n);
	std::vector<double> Au(instance.n);
	std::vector<double> Aw_local(instance.n);
	std::vector<double> Aw(instance.n);
	std::vector<double> Hv_local(instance.m);
	std::vector<double> Hu_local(instance.m);
	std::vector<double> local_gradient(instance.m);
	std::vector<double> constantLocal(8);
	std::vector<double> constantSum(8);
	std::vector<unsigned int> randPick(batchSize);
	std::vector<double> woodburyU(instance.m * batchSize);

	flag = 1;

	computeObjectiveLogistic(w, instance, obj, world);
	computeDataMatrixATimesU(w, w, Aw_local, instance);

	vall_reduce(world, Aw_local, Aw);
	computeGradientLogistic(w, Aw, local_gradient, instance);
	grad_norm = cblas_l2_norm(instance.m, &local_gradient[0], 1);
	constantLocal[6] = grad_norm * grad_norm;
	vall_reduce(world, constantLocal, constantSum);
	constantSum[6] = sqrt(constantSum[6]);
	if (rank == 0) {
		difference = abs(obj - objPre) / obj;
		printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective gap is %E\n",
		       0, 0, constantSum[6], difference);
		logFile << 0 << "," << 0 << "," << 0 << "," << constantSum[6]  << "," << difference << endl;
	}
	objPre = obj;

	for (unsigned int iter = 1; iter <= 100; iter++) {
		// Compute local first derivative
		start = gettime_();
		innerflag = 1;
		constantLocal[5] = innerflag;
		constantSum[5] = innerflag * nPartition;

		cblas_set_to_zero(v);
		cblas_set_to_zero(Hv_local);
		computeDataMatrixATimesU(w, w, Aw_local, instance);

		vall_reduce(world, Aw_local, Aw);
		computeGradientLogistic(w, Aw, local_gradient, instance);
		grad_norm = cblas_l2_norm(instance.m, &local_gradient[0], 1);
		constantLocal[6] = grad_norm;
		vall_reduce(world, constantLocal, constantSum);
		epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
		//printf("In %ith iteration, node %i now has the norm of gradient: %E \n", iter, rank, grad_norm);

		cblas_dcopy(instance.m, &local_gradient[0], 1, &r[0], 1);

		double diag = instance.lambda + mu;
		// s= p^-1 r
		if (batchSize == 0)
			ifNoPreconditioning(instance.m, r, s);
		else
			WoodburySolverForDisco(instance, instance.m, batchSize, r, s, diag);

		cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);
		int inner_iter = 0;
		// can only use this type of iteration now. Any stop or break in one node will interrupt
		// the others, there has to be a reduceall operation somehow after that.
		while (1) { //		while (flag != 0)
			computeDataMatrixATimesU(w, u, Au_local, instance);
			vall_reduce(world, Au_local, Au);
			computeLocalHessianTimesAULogistic(w, u, Au, Hu_local, instance, world);

			double rsLocal = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
			double uHuLocal = cblas_ddot(instance.m, &u[0], 1, &Hu_local[0], 1);
			constantLocal[0] = rsLocal;
			constantLocal[1] = uHuLocal;
			vall_reduce(world, constantLocal, constantSum);

			alpha = constantSum[0] / constantSum[1];
			cblas_daxpy(instance.m, alpha, &u[0], 1, &v[0], 1);
			cblas_daxpy(instance.m, alpha, &Hu_local[0], 1, &Hv_local[0], 1);
			cblas_daxpy(instance.m, -alpha, &Hu_local[0], 1, &r[0], 1);


			//CGSolver(P, instance.m, r, s);
			if (batchSize == 0)
				ifNoPreconditioning(instance.m, r, s);
			else
				WoodburySolverForDisco(instance, instance.m, batchSize, r, s, diag);

			double rsNextLocal = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
			constantLocal[2] = rsNextLocal;
			vall_reduce(world, constantLocal, constantSum);
			beta = constantSum[2] / constantSum[0];

			cblas_dscal(instance.m, beta, &u[0], 1); // FIXME -> put as one for loop, will be faster :)
			cblas_daxpy(instance.m, 1.0, &s[0], 1, &u[0], 1);

			double r_normLocal = cblas_l2_norm(instance.m, &r[0], 1);
			inner_iter++;
			if (constantSum[5] == 0) {
				break; // stop if all the inner flag = 0.
			}

			if ( r_normLocal <= epsilon || inner_iter > 100) {			//	if (r_norm <= epsilon || inner_iter > 100)
				cblas_dcopy(instance.m, &v[0], 1, &vk[0], 1);
				double vHvLocal = cblas_ddot(instance.m, &vk[0], 1, &Hv_local[0], 1); //vHvT^(t) or vHvT^(t+1)
				double vHuLocal = cblas_ddot(instance.m, &vk[0], 1, &Hu_local[0], 1);
				constantLocal[3] = vHvLocal;
				constantLocal[4] = vHuLocal;
				innerflag = 0;
				constantLocal[5] = innerflag;
			}

		}
		vall_reduce(world, constantLocal, constantSum);
		deltak = sqrt(constantSum[3] + alpha * constantSum[4]);
		cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);

		finish = gettime_();
		elapsedTime += finish - start;

		computeObjectiveLogistic(w, instance, obj, world);

		constantLocal[6] = grad_norm * grad_norm;
		vall_reduce(world, constantLocal, constantSum);
		constantSum[6] = sqrt(constantSum[6]);
		if (rank == 0) {
			difference = abs(obj - objPre) / obj;
			printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective gap is %E\n",
			       iter, inner_iter * 2, constantSum[6], difference);
			logFile << iter << "," << inner_iter * 2 << "," << elapsedTime << "," << constantSum[6] << "," << difference << endl;
		}
		objPre = obj;

		if (constantSum[6] < 1e-8) {
			//cout << endl;
			//flag = 0;
			//constantLocal[6] = flag;
			break;
		}
	}

}






void computeInitialWLogistic(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &rho, int rank) {
	std::vector<double> deltaW(instance.m);
	std::vector<double> deltaAlpha(instance.n);
	std::vector<double> alpha(instance.n);
	std::vector<double> Li(instance.n);
	for (unsigned int idx = 0; idx < instance.n; idx++) {
		double norm = cblas_l2_norm(instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
		                            &instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
		Li[idx] = 1.0 / (norm * norm / rho / instance.n + 1.0);
	}

	for (unsigned int jj = 0; jj < 10; jj++) {
		cblas_set_to_zero(deltaW);
		cblas_set_to_zero(deltaAlpha);

		for (unsigned int it = 0; it < floor(instance.n / 10); it++) {
			unsigned int idx = rand() / (0.0 + RAND_MAX) * instance.n;

			double dotProduct = 0;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
				dotProduct += (w[instance.A_csr_col_idx[i]]
				               + 1.0 * deltaW[instance.A_csr_col_idx[i]])
				              * instance.A_csr_values[i];
			}
			double alphaI = alpha[idx] + deltaAlpha[idx];
			double deltaAl = 0;
			deltaAl = (1.0 * instance.b[idx] - alphaI - dotProduct * instance.b[idx]) * Li[idx];
			deltaAlpha[idx] += deltaAl;
			for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
				deltaW[instance.A_csr_col_idx[i]] += 1.0 / instance.n / rho * deltaAl
				                                     * instance.A_csr_values[i] * instance.b[idx];

		}
		cblas_daxpy(instance.m, 1.0, &deltaW[0], 1, &w[0], 1);
		cblas_daxpy(instance.n, 1.0, &deltaAlpha[0], 1, &alpha[0], 1);
	}
	double primalError;
	double dualError;

	computePrimalAndDualObjectiveLogistic(instance, alpha, w, rho, dualError, primalError);
	printf("No. %i node now has the duality gap %E \n", rank, primalError + dualError);

}



#endif
