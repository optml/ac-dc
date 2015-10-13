#include <math.h>
#include <gsl/gsl_blas.h> // for BLAS
#include "../solver/distributed/distributed_essentials.h"
#include "QR_solver.h"

#ifndef DISCOHELPER_H_
#define DISCOHELPER_H_

void compute_initial_w(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &rho) {



}

void update_w(std::vector<double> &w, std::vector<double> &vk, double &deltak) {

}

void compute_objective(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &obj) {

	obj = 0.0;

	for (unsigned int idx = 0; idx < instance.n; idx++) {

		double w_x = 0.0;
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			w_x += w[instance.A_csr_col_idx[i]] * instance.A_csr_values[i];

		obj += 0.5 * (w_x - instance.b[idx]) * (w_x - instance.b[idx]);
		//obj += log(1.0 + exp(-1.0 * instance.b[idx] * w_x));
	}

	obj = 1.0 / instance.n * obj + 0.5 * instance.lambda * cblas_l2_norm(w.size(), &w[0], 1);

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

		temp = (w_x - instance.b[idx]) / instance.n;
		//temp = temp / (1.0 + temp) * (-instance.b[idx]) / instance.total_n;
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			grad[instance.A_csr_col_idx[i]] += temp * instance.A_csr_values[i];
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



void distributed_PCG_SparseP(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &mu,
                     std::vector<double> &vk, double &deltak, boost::mpi::communicator &world) {

	// Compute Matrix P
	// Broadcastw_k
	std::vector<int> flag(1);
	mpi::request reqs[1];


	double epsilon;
	double grad_norm;
	double alpha = 0.0;
	double beta = 0.0;
	unsigned int batchSize = 100;

	std::vector<double> v(instance.m);
	std::vector<double> s(instance.m);
	std::vector<double> r(instance.m);
	std::vector<double> u(instance.m);
	std::vector<double> Hu_local(instance.m);
	std::vector<double> Hu(instance.m);
	std::vector<double> Hv(instance.m);
	std::vector<double> gradient(instance.m);
	std::vector<double> local_gradient(instance.m);
	std::vector<unsigned int> randPick(batchSize);
	std::vector<double> woodburyU(instance.m * batchSize);
	double diag = instance.lambda + mu;

	for (unsigned int iter = 0; iter < 100; iter++) {
		// Compute local first derivative
		flag[0] = 1;

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
			printf("In %ith iteration, now has the norm of gradient: %E \n", iter, grad_norm);
			if (grad_norm < 1e-8) {
				cout << endl;
				break;
			}

			cblas_dcopy(instance.m, &gradient[0], 1, &r[0], 1);


			for (unsigned int i = 0; i < batchSize; i++)
				randPick[i] = rand() / (0.0 + RAND_MAX) * instance.n;

			// s= p^-1 r
			WoodburySolver(instance, instance.m, batchSize, r, s, randPick, diag);
			cblas_dcopy(instance.m, &s[0], 1, &u[0], 1);

		}

		int inner_iter = 0;
		while (flag[0] != 0) {
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
				//CGSolver(P, instance.m, r, s);
				WoodburySolver(instance, instance.m, batchSize, r, s, randPick, diag);

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
		vbroadcast(world, w, 0);

		double objective = 0.0;
		double objective_world = 0.0;
		compute_objective(w, instance, objective);
		boost::mpi::reduce(world, objective, objective_world, plus<double>(), 1);
		objective_world /= world.size();
		if (world.rank() == 1) 	cout  << objective_world << endl;

	}


}


void distributed_PCG(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &mu,
                     std::vector<double> &vk, double &deltak, boost::mpi::communicator &world) {

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
				break;
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
		while (flag[0] != 0) {
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

		if (world.rank() == 0) {
			cblas_daxpy(instance.m, -1.0 / (1.0 + deltak), &vk[0], 1, &w[0], 1);
		}
		vbroadcast(world, w, 0);

		double objective = 0.0;
		double objective_world = 0.0;
		compute_objective(w, instance, objective);
		boost::mpi::reduce(world, objective, objective_world, plus<double>(), 1);
		objective_world /= world.size();
		if (world.rank() == 1) 	cout  << objective_world << endl;

	}


}


#endif /* DISCOHELPER_H_ */
