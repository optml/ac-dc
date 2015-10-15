#ifndef OCSIDHELPER_H_
#define OCSIDHELPER_H_


#include <math.h>
#include <gsl/gsl_blas.h> // for BLAS
#include "../solver/distributed/distributed_essentials.h"
#include "QR_solver.h"


void compute_objective(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &obj, 
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

	for (unsigned int idx = 0; idx < instance.n; idx++) {
		obj += 0.5 * (w_x_world[idx] - instance.b[idx]) * (w_x_world[idx] - instance.b[idx]);
		//obj += log(1.0 + exp(-1.0 * instance.b[idx] * w_x));
	}

	obj = 1.0 / instance.n * obj + 0.5 * instance.lambda * w_x_world[instance.n];

}


void computePrimalAndDualObjectiveValue(ProblemData<unsigned int, double> &instance,
                                        std::vector<double> &alpha, std::vector<double> &w, double &rho, double &finalDualError,
                                        double &finalPrimalError) {

	double localError = 0.0;
	for (unsigned int i = 0; i < instance.n; i++) {
		double tmp = alpha[i] * alpha[i] * 0.5 - instance.b[i] * alpha[i];
		localError += tmp;
	}

	double localQuadLoss = 0.0;
	for (unsigned int idx = 0; idx < instance.n; idx++) {
		double dotProduct = 0.0;
		for (unsigned int i = instance.A_csr_row_ptr[idx];
		        i < instance.A_csr_row_ptr[idx + 1]; i++) {
			dotProduct += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];
		}
		double single = 1.0 * instance.b[idx] -  dotProduct * instance.b[idx];
		double tmp = 0.5 * single * single;

		localQuadLoss += tmp;
	}

	finalPrimalError = 0;
	finalDualError = 0;

	double tmp2 = cblas_l2_norm(instance.m, &w[0], 1);
	finalDualError = 1.0 / instance.n * localError
	                 + 0.5 * rho * tmp2 * tmp2;
	finalPrimalError =  1.0 / instance.n * localQuadLoss
	                    + 0.5 * rho * tmp2 * tmp2;

}

void compute_gradient(std::vector<double> &w, std::vector<double> &Aw, std::vector<double> &grad,
                      ProblemData<unsigned int, double> &instance) {

	cblas_set_to_zero(grad);
	double temp = 0.0;
	double w_x = 0.0;

	for (unsigned int idx = 0; idx < instance.n; idx++) {
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			grad[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] / instance.total_n
			                                   * (Aw[idx] - instance.b[idx]);

		//cout<<instance.A_csr_row_ptr[idx]<<"    ";
	}

	cblas_daxpy(instance.m, instance.lambda, &w[0], 1, &grad[0], 1);

}

void computeDataMatrixATimesU(std::vector<double> &w, std::vector<double> &u, std::vector<double> &Au,
                              ProblemData<unsigned int, double> &instance) {

//	cblas_set_to_zero(Au); <- not necesary as you are updating Au[idx] in a loop sequencially
	for (unsigned int idx = 0; idx < instance.n; idx++) {
		Au[idx] = 0;
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			Au[idx] += instance.A_csr_values[i] * u[instance.A_csr_col_idx[i]];
	}


}


void computeLocalHessianTimesAU(std::vector<double> &w, std::vector<double> &u, std::vector<double> &Au,
                                std::vector<double> &Hu_local, ProblemData<unsigned int, double> &instance) {

	cblas_set_to_zero(Hu_local);

	for (unsigned int idx = 0; idx < instance.n; idx++) {
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			Hu_local[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * Au[idx] / instance.n;
	}

	for (unsigned int i = 0; i < instance.m; i++)
		Hu_local[i] += instance.lambda * u[i];


}



void distributed_PCGByD_SparseP(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &mu,
                                std::vector<double> &vk, double &deltak, boost::mpi::communicator &world, int nPartition, int rank,
                                std::ofstream &logFile) {

	// Compute Matrix P
	// Broadcastw_k
	int flag;
	int innerflag;
	//std::vector<int> flag(10);
	//mpi::request reqs[1];
	double start = 0;
	double finish = 0;
	double elapsedTime = 0;
	double grad_norm;

	double epsilon;
	double obj;
	double alpha = 0.0;
	double beta = 0.0;
	unsigned int batchSize = 100;

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
	constantLocal[6] = flag;
	constantSum[6] = flag;	

	for (unsigned int iter = 0; iter < 500; iter++) {
		// Compute local first derivative
		start = gettime_();
		innerflag = 1;
		constantLocal[5] = innerflag;
		constantSum[5] = innerflag * nPartition;

		cblas_set_to_zero(v);
		cblas_set_to_zero(Hv_local);
//		cout<<iter<<endl;
		computeDataMatrixATimesU(w, w, Aw_local, instance);

		vall_reduce(world, Aw_local, Aw);
		compute_gradient(w, Aw, local_gradient, instance);
		grad_norm = cblas_l2_norm(instance.m, &local_gradient[0], 1);
		constantLocal[6] = grad_norm;
		vall_reduce(world, constantLocal, constantSum);
		epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
		//printf("In %ith iteration, node %i now has the norm of gradient: %E \n", iter, rank, grad_norm);

		//if (constantSum[6] == 0) 	break;
		

		cblas_dcopy(instance.m, &local_gradient[0], 1, &r[0], 1);


		for (unsigned int i = 0; i < batchSize; i++)
			randPick[i] = rand() / (0.0 + RAND_MAX) * instance.n;

		double diag = instance.lambda + mu;
		// s= p^-1 r
		WoodburySolver(instance, instance.m, batchSize, r, s, randPick, diag);

		cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);
		int inner_iter = 0;
		// can only use this type of iteration now. Any stop or break in one node will interrupt
		// the others, there has to be a reduceall operation somehow after that.
		while (1) { //		while (flag != 0)
			computeDataMatrixATimesU(w, u, Au_local, instance);
			vall_reduce(world, Au_local, Au);  
			computeLocalHessianTimesAU(w, u, Au, Hu_local, instance);

			double rsLocal = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
			double uHuLocal = cblas_ddot(instance.m, &u[0], 1, &Hu_local[0], 1);
			constantLocal[0] = rsLocal;
			constantLocal[1] = uHuLocal;
			vall_reduce(world, constantLocal, constantSum);
			//if (rank == 3) cout<<constantLocal[0]<<"    "<<constantSum[0]<<endl;
			alpha = constantSum[0] / constantSum[1];
			cblas_daxpy(instance.m, alpha, &u[0], 1, &v[0], 1);
			cblas_daxpy(instance.m, alpha, &Hu_local[0], 1, &Hv_local[0], 1);
			cblas_daxpy(instance.m, -alpha, &Hu_local[0], 1, &r[0], 1);

			// ? solve linear system to get new s
			//CGSolver(P, instance.m, r, s);
			WoodburySolver(instance, instance.m, batchSize, r, s, randPick, diag);

			double rsNextLocal = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
			constantLocal[2] = rsNextLocal;
			vall_reduce(world, constantLocal, constantSum);
			beta = constantSum[2] / constantSum[0];

			cblas_dscal(instance.m, beta, &u[0], 1); // FIXME -> put as one for loop, will be faster :)
			cblas_daxpy(instance.m, 1.0, &s[0], 1, &u[0], 1);

			double r_normLocal = cblas_l2_norm(instance.m, &r[0], 1);
			inner_iter++;
			if (constantSum[5] == 0){
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
				//vall_reduce(world, flag, flagWorld);
			}

		}
		vall_reduce(world, constantLocal, constantSum);
		deltak = sqrt(constantSum[3] + alpha * constantSum[4]);
		cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);

		finish = gettime_();
		elapsedTime += finish - start;

		compute_objective(w, instance, obj, world);
		//constantLocal[7] = obj; vall_reduce(world, constantLocal, constantSum);

		if (rank == 0) {
			printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective value is %E\n", 
									iter, inner_iter, constantSum[6], obj);
			logFile << iter << "," << elapsedTime << "," << constantSum[6] << ","<<obj<<endl;
		}
		if (constantSum[6] < 1e-8) {
			//cout << endl;
			//flag = 0;
			//constantLocal[6] = flag;
			break;
		}
	}

}




void distributed_PCGByD(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &mu,
                        std::vector<double> &vk, double &deltak, boost::mpi::communicator &world, int nPartition, int rank
                        , std::ofstream &logFile) {

	// Compute Matrix P
	// Broadcastw_k
	int flag;
	//std::vector<int> flag(10);
	//mpi::request reqs[1];


	double epsilon;
	double grad_norm;
	double alpha = 0.0;
	double beta = 0.0;

	std::vector<double> P(instance.m * instance.m);
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
	std::vector<double> constantLocal(6);
	std::vector<double> constantSum(6);

	for (unsigned int iter = 0; iter < 50; iter++) {
		// Compute local first derivative
		flag = 1;

		cblas_set_to_zero(P);
		cblas_set_to_zero(v);
		cblas_set_to_zero(Hv_local);
//		cout<<iter<<endl;
		computeDataMatrixATimesU(w, w, Aw_local, instance);

		vall_reduce(world, Aw_local, Aw);
		compute_gradient(w, Aw, local_gradient, instance);
		double grad_norm = cblas_l2_norm(instance.m, &local_gradient[0], 1);
		epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
		printf("In %ith iteration, node %i now has the norm of gradient: %E \n", iter, rank, grad_norm);
		if (grad_norm < 1e-8) {
			cout << endl;
			break;
		}

		cblas_dcopy(instance.m, &local_gradient[0], 1, &r[0], 1);

		for (unsigned int idx = 0; idx < instance.n; idx++) {
			for (unsigned int i = instance.A_csr_row_ptr[idx];	i < instance.A_csr_row_ptr[idx + 1]; i++) {
				for (unsigned int j = instance.A_csr_row_ptr[idx];	j < instance.A_csr_row_ptr[idx + 1]; j++) {
					P[instance.m * instance.A_csr_col_idx[i] + instance.A_csr_col_idx[j]] +=
					    instance.A_csr_values[i] * instance.A_csr_values[j] / instance.total_n;
				}
			}
		}
		for (unsigned int i = 0; i < instance.m; i++) {
			P[instance.m * i + i] += instance.lambda + mu;
		}
		// s= p^-1 r
		CGSolver(P, instance.m, r, s); // should be done differently if Square loss is used
		cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);


		int inner_iter = 0;

		// can only use this type of iteration now. Any stop or break in one node will interrupt
		// the others, there has to be a reduceall operation somehow after that.
		while (1) { //		while (flag != 0)
			computeDataMatrixATimesU(w, u, Au_local, instance);
			vall_reduce(world, Au_local, Au);  //BUG is here!
			computeLocalHessianTimesAU(w, u, Au, Hu_local, instance);

			double rsLocal = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
			double uHuLocal = cblas_ddot(instance.m, &u[0], 1, &Hu_local[0], 1);
			constantLocal[0] = rsLocal;
			constantLocal[1] = uHuLocal;
			vall_reduce(world, constantLocal, constantSum);
			//if (rank == 3) cout<<constantLocal[0]<<"    "<<constantSum[0]<<endl;
			alpha = constantSum[0] / constantSum[1];
			cblas_daxpy(instance.m, alpha, &u[0], 1, &v[0], 1);
			cblas_daxpy(instance.m, alpha, &Hu_local[0], 1, &Hv_local[0], 1);
			cblas_daxpy(instance.m, -alpha, &Hu_local[0], 1, &r[0], 1);

			// ? solve linear system to get new s
			CGSolver(P, instance.m, r, s);

			double rsNextLocal = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
			constantLocal[2] = rsNextLocal;
			vall_reduce(world, constantLocal, constantSum);
			beta = constantSum[2] / constantSum[0];

			cblas_dscal(instance.m, beta, &u[0], 1); // FIXME -> put as one for loop, will be faster :)
			cblas_daxpy(instance.m, 1.0, &s[0], 1, &u[0], 1);

			double r_normLocal = cblas_l2_norm(instance.m, &r[0], 1);
			constantLocal[5] = r_normLocal;
			vall_reduce(world, constantLocal, constantSum);
			inner_iter++;
			if ( inner_iter > 20) {			//	if (r_norm <= epsilon || inner_iter > 100)
				cblas_dcopy(instance.m, &v[0], 1, &vk[0], 1);
				double vHvLocal = cblas_ddot(instance.m, &vk[0], 1, &Hv_local[0], 1); //vHvT^(t) or vHvT^(t+1)
				double vHuLocal = cblas_ddot(instance.m, &vk[0], 1, &Hu_local[0], 1);
				constantLocal[3] = vHvLocal;
				constantLocal[4] = vHuLocal;
				flag = 0;
				break;
				//vall_reduce(world, flag, flagWorld);
			}

		}
		vall_reduce(world, constantLocal, constantSum);
		deltak = sqrt(constantSum[3] + alpha * constantSum[4]);
		cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);

	}


}


void compute_initial_w(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &rho, int rank) {
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

	computePrimalAndDualObjectiveValue(instance, alpha, w, rho, dualError, primalError);
	printf("No. %i node now has the duality gap %E \n", rank, primalError + dualError);

}



#endif
