#include <math.h>
#include <gsl/gsl_blas.h> // for BLAS
#include "../solver/distributed/distributed_essentials.h"
#include "QR_solver.h"

#ifndef DISCOHELPER_H_
#define DISCOHELPER_H_



void compute_objective(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &obj, int nPartition) {

	obj = 0.0;

	for (unsigned int idx = 0; idx < instance.n; idx++) {

		double w_x = 0.0;
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			w_x += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];

		obj += 0.5 * (w_x * instance.b[idx] - instance.b[idx]) * (w_x * instance.b[idx] - instance.b[idx]);
		//obj += log(1.0 + exp(-1.0 * instance.b[idx] * w_x));
	}

	obj = 1.0 / instance.total_n * obj + 0.5 * instance.lambda * cblas_l2_norm(w.size(), &w[0], 1)
	      * cblas_l2_norm(w.size(), &w[0], 1) / nPartition;

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


void compute_gradient(std::vector<double> &w, std::vector<double> &grad,
                      ProblemData<unsigned int, double> &instance) {

	cblas_set_to_zero(grad);
	double temp = 0.0;
	double w_x = 0.0;
	// for (unsigned int idx = 0; idx < instance.n; idx++) {
	// 	w_x = 0.0;
	// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
	// 		w_x += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];

	// 	temp = exp(-1.0 * instance.b[idx] * w_x);
	// 	temp = temp / (1.0 + temp) * (-instance.b[idx]) / instance.total_n;
	// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
	// 		grad[instance.A_csr_col_idx[i]] += temp * instance.A_csr_values[i];
	// }
	for (unsigned int idx = 0; idx < instance.n; idx++) {
		w_x = 0.0;
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			w_x += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];

		temp = (w_x * instance.b[idx] - instance.b[idx]);
		//temp = temp / (1.0 + temp) * (-instance.b[idx]) / instance.total_n;
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			grad[instance.A_csr_col_idx[i]] += temp * instance.A_csr_values[i] * instance.b[idx] / instance.n;
	}

	cblas_daxpy(instance.m, instance.lambda, &w[0], 1, &grad[0], 1);

}

void computeHessianTimesU(std::vector<double> &w, std::vector<double> &u, std::vector<double> &Hu,
                          ProblemData<unsigned int, double> &instance) {

	cblas_set_to_zero(Hu);
	double temp = 0.0;
	// std::vector<double> Hessian(instance.m * instance.m);
	// for (unsigned int idx = 0; idx < instance.n; idx++) {
	// 	double w_x = 0.0;
	// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
	// 		w_x += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];

	// 	temp = exp(-1.0 * instance.b[idx] * w_x);
	// 	temp = temp / (1.0 + temp) / (1.0 + temp);

	// 	for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
	// 		for (unsigned int j = instance.A_csr_row_ptr[idx]; j < instance.A_csr_row_ptr[idx + 1]; j++)
	// 			Hessian[instance.m * i + j] += temp * instance.A_csr_values[i] * instance.A_csr_values[j]
	// 																								   / instance.total_n;

	// 	for (unsigned int i = 0; i < instance.m; i++)
	// 		Hessian[instance.m * i + i] += instance.lambda;
	// }

	// cblas_dgemv(CblasRowMajor, CblasNoTrans, instance.m, instance.m, 1.0, &Hessian[0], instance.m, &u[0], 1, 1.0,
	// 		&Hu[0], 1);


	for (unsigned int idx = 0; idx < instance.n; idx++) {

		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
			for (unsigned int j = instance.A_csr_row_ptr[idx]; j < instance.A_csr_row_ptr[idx + 1]; j++) {

				Hu[instance.A_csr_col_idx[i]] += (instance.A_csr_values[i] * instance.A_csr_values[j])
				                                 * u[instance.A_csr_col_idx[j]] / instance.n;

			}
		}

	}

	for (unsigned int i = 0; i < instance.m; i++)
		Hu[i] += instance.lambda * u[i];

}

void computeDataMatrixATimesU(std::vector<double> &w, std::vector<double> &u, std::vector<double> &Au,
                              ProblemData<unsigned int, double> &instance) {

//	cblas_set_to_zero(Au); <- not necesary as you are updating Au[idx] in a loop sequencially
	for (unsigned int idx = 0; idx < instance.n; idx++) {
		Au[idx] = 0;
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			Au[idx] += instance.A_csr_values[i] * instance.b[idx] * u[instance.A_csr_col_idx[i]];
	}


}

void computeLocalHessianTimesAU(std::vector<double> &w, std::vector<double> &u, std::vector<double> &Au,
                                std::vector<double> &Hu_local, ProblemData<unsigned int, double> &instance) {

	cblas_set_to_zero(Hu_local);

	for (unsigned int idx = 0; idx < instance.n; idx++) {
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			Hu_local[instance.A_csr_col_idx[i]] += instance.A_csr_values[i] * instance.b[idx] * Au[idx] / instance.n;
	}

	for (unsigned int i = 0; i < instance.m; i++)
		Hu_local[i] += instance.lambda * u[i];


}


void distributed_PCG_SparseP(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &mu,
                             std::vector<double> &vk, double &deltak, boost::mpi::communicator &world, std::ofstream &logFile) {

	// Compute Matrix P
	// Broadcastw_k
	std::vector<int> flag(2);
	mpi::request reqs[1];
	double difference;
	double objPre;

	double start = 0;
	double finish = 0;
	double elapsedTime = 0;
	double grad_norm;

	double epsilon;
	double alpha = 0.0;
	double beta = 0.0;
	unsigned int batchSize = 100;

	std::vector<double> v(instance.m);
	std::vector<double> s(instance.m);
	std::vector<double> r(instance.m);
	std::vector<double> u(instance.m);
	std::vector<double> Au(instance.n);
	std::vector<double> Hu_local(instance.m);
	std::vector<double> Hu(instance.m);
	std::vector<double> Hv(instance.m);
	std::vector<double> gradient(instance.m);
	std::vector<double> local_gradient(instance.m);
	std::vector<unsigned int> randPick(batchSize);
	std::vector<double> woodburyU(instance.m * batchSize);
	std::vector<double> objective(2);
	std::vector<double> objective_world(2);
	double diag = instance.lambda + mu;

	compute_objective(w, instance, objective[0], world.size());
	//boost::mpi::reduce(world, objective, objective_world, plus<double>(), 1);
	vall_reduce(world, objective, objective_world);
	//objective_world /= world.size();
	//if (world.rank() == 1) 	cout  << objective_world << endl;
	compute_gradient(w, local_gradient, instance);
	vall_reduce(world, local_gradient, gradient);
	cblas_dscal(instance.m, 1.0 / world.size(), &gradient[0], 1);

	if (world.rank() == 0) {
		grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
		difference = abs(objective_world[0] - objPre) / objective_world[0];
		printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective gap is %E\n",
		       0, 0, grad_norm, difference);
		logFile << 0 << "," << 0 << "," << 0 << "," << grad_norm << "," << difference << endl;
	}
	objPre = objective_world[0];

	for (unsigned int iter = 1; iter <= 100; iter++) {
		// Compute local first derivative
		start = gettime_();

		flag[0] = 1;
		flag[1] = 1;

		cblas_set_to_zero(v);
		cblas_set_to_zero(Hv);
		vbroadcast(world, w, 0);
//		cout<<iter<<endl;
		compute_gradient(w, local_gradient, instance);
//		for (unsigned int i = 0; i < instance.m; i++)			local_gradient[i] = 0.1 * rand() / (RAND_MAX + 0.0);
		// Aggregates to form f'(w_k)
		vall_reduce(world, local_gradient, gradient);
		cblas_dscal(instance.m, 1.0 / world.size(), &gradient[0], 1);

		if (world.rank() == 0) {
			grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
			epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
			//printf("In %ith iteration, now has the norm of gradient: %E \n", iter, grad_norm);
			if (grad_norm < 1e-8) {
				flag[1] = 0;
			}

			cblas_dcopy(instance.m, &gradient[0], 1, &r[0], 1);

			// s= p^-1 r
			//CGSolver(P, instance.m, r, s);
			WoodburySolverForDisco(instance, instance.m, batchSize, r, s, diag);
			//ifNoPreconditioning(instance.m, r, s);
			cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);

		}

		int inner_iter = 0;
		while (flag[0] != 0) {
			vbroadcast(world, u, 0);
			computeDataMatrixATimesU(w, u, Au, instance);
			computeLocalHessianTimesAU(w, u, Au, Hu_local, instance);
			//computeHessianTimesU(w, u, Hu_local, instance); //cout<<world.rank()<<"    "<<Hu_local[0]<<endl;
			vall_reduce(world, Hu_local, Hu);
			cblas_dscal(instance.m, 1.0 / world.size(), &Hu[0], 1); //for (unsigned int i = 0; i < instance.m; i++)	cout<<i<<"    "<<Hu_local[i]<<"  "<<Hu[i]<<endl;

			if (world.rank() == 0) {
				//cout<<"I will do this induvidually!!!!!!!!!!"<<endl;
				double nom = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
				double denom = cblas_ddot(instance.m, &u[0], 1, &Hu[0], 1);
				alpha = nom / denom;

				cblas_daxpy(instance.m, alpha, &u[0], 1, &v[0], 1);
				cblas_daxpy(instance.m, alpha, &Hu[0], 1, &Hv[0], 1);
				cblas_daxpy(instance.m, -alpha, &Hu[0], 1, &r[0], 1);

				// ? solve linear system to get new s
				//CGSolver(P, instance.m, r, s);
				WoodburySolverForDisco(instance, instance.m, batchSize, r, s, diag);
				//ifNoPreconditioning(instance.m, r, s);

				double nom_new = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
				beta = nom_new / nom;
				cblas_dscal(instance.m, beta, &u[0], 1);
				cblas_daxpy(instance.m, 1.0, &s[0], 1, &u[0], 1);

				double r_norm = cblas_l2_norm(instance.m, &r[0], 1);
				inner_iter++;

				if (r_norm <= epsilon || inner_iter > 100) {
					cblas_dcopy(instance.m, &v[0], 1, &vk[0], 1);
					double vHv = cblas_ddot(instance.m, &vk[0], 1, &Hv[0], 1); //vHvT^(t) or vHvT^(t+1)
					double vHu = cblas_ddot(instance.m, &vk[0], 1, &Hu[0], 1);
					deltak = sqrt(vHv + alpha * vHu);
					flag[0] = 0;
				}
			}

			vbroadcast(world, flag, 0);

		}

		if (world.rank() == 0) {
			cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);
		}

		finish = gettime_();
		elapsedTime += finish - start;
		vbroadcast(world, w, 0);

		compute_objective(w, instance, objective[0], world.size());
		//boost::mpi::reduce(world, objective, objective_world, plus<double>(), 1);
		vall_reduce(world, objective, objective_world);
		//objective_world /= world.size();
		//if (world.rank() == 1) 	cout  << objective_world << endl;

		if (world.rank() == 0) {
			difference = abs(objective_world[0] - objPre) / objective_world[0];
			printf("%ith runs %i CG iterations, the norm of gradient is %E, the objective gap is %E\n",
			       iter, 2 * inner_iter + 2, grad_norm, difference);
			logFile << iter << "," << 2 * inner_iter + 2 << "," << elapsedTime << "," << grad_norm << "," << difference << endl;
		}
		objPre = objective_world[0];

		if (flag[1] == 0)
			break;


	}


}


void distributed_PCG(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &mu,
                     std::vector<double> &vk, double &deltak, boost::mpi::communicator &world, std::ofstream &logFile) {

	// Compute Matrix P
	// Broadcastw_k
	std::vector<int> flag(1);
	mpi::request reqs[1];


	double epsilon;
	double grad_norm;
	double alpha = 0.0;
	double beta = 0.0;

	std::vector<double> P(instance.m * instance.m);
	std::vector<double> v(instance.m);
	std::vector<double> s(instance.m);
	std::vector<double> r(instance.m);
	std::vector<double> u(instance.m);
	std::vector<double> Hu_local(instance.m);
	std::vector<double> Hu(instance.m);
	std::vector<double> Hv(instance.m);
	std::vector<double> gradient(instance.m);
	std::vector<double> local_gradient(instance.m);

	for (unsigned int iter = 0; iter < 100; iter++) {
		// Compute local first derivative
		flag[0] = 1;

		cblas_set_to_zero(P);
		cblas_set_to_zero(v);
		cblas_set_to_zero(Hv);
		vbroadcast(world, w, 0);
//		cout<<iter<<endl;
		compute_gradient(w, local_gradient, instance);
//		for (unsigned int i = 0; i < instance.m; i++)			local_gradient[i] = 0.1 * rand() / (RAND_MAX + 0.0);
		// Aggregates to form f'(w_k)
		vall_reduce(world, local_gradient, gradient);
		cblas_dscal(instance.m, 1.0 / world.size(), &gradient[0], 1);

		if (world.rank() == 0) {
			double grad_norm = cblas_l2_norm(instance.m, &gradient[0], 1);
			epsilon = 0.05 * grad_norm * sqrt(instance.lambda / 10.0);
			cout << iter << "   " << grad_norm << "    " << epsilon << "   ";
			if (grad_norm < 1e-8) {
				cout << endl;
				flag[0] = -1;
			}

			cblas_dcopy(instance.m, &gradient[0], 1, &r[0], 1);

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
			CGSolver(P, instance.m, r, s);
			cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);

		}

		int inner_iter = 0;
		while (flag[0] == 1) {
			vbroadcast(world, u, 0);
			computeHessianTimesU(w, u, Hu_local, instance); //cout<<world.rank()<<"    "<<Hu_local[0]<<endl;
			vall_reduce(world, Hu_local, Hu);
			cblas_dscal(instance.m, 1.0 / world.size(), &Hu[0], 1); //for (unsigned int i = 0; i < instance.m; i++)	cout<<i<<"    "<<Hu_local[i]<<"  "<<Hu[i]<<endl;

			if (world.rank() == 0) {
				//cout<<"I will do this induvidually!!!!!!!!!!"<<endl;
				double nom = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
				double denom = cblas_ddot(instance.m, &u[0], 1, &Hu[0], 1);
				alpha = nom / denom;

				cblas_daxpy(instance.m, alpha, &u[0], 1, &v[0], 1);
				cblas_daxpy(instance.m, alpha, &Hu[0], 1, &Hv[0], 1);
				cblas_daxpy(instance.m, -alpha, &Hu[0], 1, &r[0], 1);

				// ? solve linear system to get new s
				CGSolver(P, instance.m, r, s);

				double nom_new = cblas_ddot(instance.m, &r[0], 1, &s[0], 1);
				beta = nom_new / nom;
				cblas_dscal(instance.m, beta, &u[0], 1);
				cblas_daxpy(instance.m, 1.0, &s[0], 1, &u[0], 1);

				double r_norm = cblas_l2_norm(instance.m, &r[0], 1);
				inner_iter++;

				if (r_norm <= epsilon || inner_iter > 100) {
					cblas_dcopy(instance.m, &v[0], 1, &vk[0], 1);
					double vHv = cblas_ddot(instance.m, &vk[0], 1, &Hv[0], 1); //vHvT^(t) or vHvT^(t+1)
					double vHu = cblas_ddot(instance.m, &vk[0], 1, &Hu[0], 1);
					deltak = sqrt(vHv + alpha * vHu);
					flag[0] = 0;
				}
			}

			vbroadcast(world, flag, 0);

		}

		vbroadcast(world, flag, 0);
		if (flag[0] == -1)
			break;

		if (world.rank() == 0) {
			cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);
		}
		vbroadcast(world, w, 0);

		double objective = 0.0;
		double objective_world = 0.0;
		compute_objective(w, instance, objective, world.size());
		boost::mpi::reduce(world, objective, objective_world, plus<double>(), 1);
		objective_world /= world.size();
		if (world.rank() == 1) 	cout  << objective_world << endl;

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

#endif /* DISCOHELPER_H_ */
