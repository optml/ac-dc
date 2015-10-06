#ifndef OCSIDHELPER_H_
#define OCSIDHELPER_H_


#include <math.h>
#include <gsl/gsl_blas.h> // for BLAS
#include "../solver/distributed/distributed_essentials.h"
#include "QR_solver.h"


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

void compute_gradient(std::vector<double> &w, std::vector<double> &Aw, std::vector<double> &grad,
                      ProblemData<unsigned int, double> &instance) {

	cblas_set_to_zero(grad);
	double temp = 0.0;
	double w_x = 0.0;

	for (unsigned int idx = 0; idx < instance.n; idx++) {
		for (unsigned int i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++)
			grad[instance.A_csr_col_idx[i]] += 2.0 * instance.A_csr_values[i] / instance.total_n 
					* (Aw[idx] - instance.b[idx]);

	//cout<<instance.A_csr_row_ptr[idx]<<"    ";
	}

	for (unsigned int i = 0; i < instance.m; i++){
		grad[i] += instance.lambda * w[i];
	}
}

void computeDataMatrixATimesU(std::vector<double> &w, std::vector<double> &u, std::vector<double> &Au,
                              ProblemData<unsigned int, double> &instance) {

	cblas_set_to_zero(Au);

	for (unsigned int idx = 0; idx < instance.n; idx++) {
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


void distributed_PCGByD(std::vector<double> &w, ProblemData<unsigned int, double> &instance, double &mu,
                        std::vector<double> &vk, double &deltak, boost::mpi::communicator &world, int nPartition, int rank) {

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

	for (unsigned int iter = 0; iter < 20; iter++) {
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
		cout << iter << "   " << grad_norm << "    " << epsilon << "   "<<endl;
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
		CGSolver(P, instance.m, r, s);
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
			cblas_dscal(instance.m, beta, &u[0], 1);
			cblas_daxpy(instance.m, 1.0, &s[0], 1, &u[0], 1);

			double r_normLocal = cblas_l2_norm(instance.m, &r[0], 1);
			constantLocal[5] = r_normLocal;
			vall_reduce(world, constantLocal, constantSum);
			inner_iter++;
//cout<<constantSum[5]<<endl;
			if ( inner_iter > 50) {			//	if (r_norm <= epsilon || inner_iter > 100)
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
		// double objective = 0.0;
		// double objective_world = 0.0;
		// compute_objective(w, instance, objective);
		// //boost::mpi::reduce(world, objective, objective_world, plus<double>(), 1);
		// if (world.rank() == 0) 	cout  << objective << endl;

	}


}




#endif