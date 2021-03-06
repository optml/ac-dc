/*
 * QuadraticLoss.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef QUADRATICLOSS_H_
#define QUADRATICLOSS_H_


#include "LossFunction.h"

template<typename L, typename D>
class QuadraticLoss: public LossFunction<L, D>  {
public:
	QuadraticLoss() {

	}

	virtual ~QuadraticLoss() {}

	virtual void computeObjectiveValue(ProblemData<L, D> & instance,
	                                   mpi::communicator & world, std::vector<D> & w, double &finalDualError,
	                                   double &finalPrimalError) {

		D localError = 0.0;
		for (unsigned int i = 0; i < instance.n; i++) {
			D tmp = instance.x[i] * instance.x[i] * 0.5 - instance.b[i] * instance.x[i];
			localError += tmp;
		}

		D localQuadLoss = 0.0;
		for (unsigned int idx = 0; idx < instance.n; idx++) {
			D dotProduct = 0.0;
			for (L i = instance.A_csr_row_ptr[idx];
			        i < instance.A_csr_row_ptr[idx + 1]; i++) {
				dotProduct += (w[instance.A_csr_col_idx[i]])
				              * instance.A_csr_values[i];
			}
			D tmp = 0.5 * (1.0 * instance.b[idx] -  dotProduct * instance.b[idx]) * (1.0 * instance.b[idx] - dotProduct * instance.b[idx]);

			localQuadLoss += tmp;

		}

		finalPrimalError = 0;
		vall_reduce(world, &localQuadLoss, &finalPrimalError, 1);

		finalDualError = 0;
		//cout<<localError<<endl;
		vall_reduce(world, &localError, &finalDualError, 1);
		//cout<<finalDualError<<endl;

		D tmp2 = cblas_l2_norm(w.size(), &w[0], 1);
		finalDualError = 1.0 / (0.0 + instance.total_n) * finalDualError
		                 + 0.5 * instance.lambda * tmp2 * tmp2;
		finalPrimalError =  1.0 / (0.0 + instance.total_n) * finalPrimalError
		                    + 0.5 * instance.lambda * tmp2 * tmp2;
	}


	virtual void compute_subproproblem_obj(ProblemData<L, D> &instance,
	                                       std::vector<D> &deltaAlpha, std::vector<D> &w, D &potent_obj) {
		D obj_temp1 = 0.0;
		D obj_temp2 = 0.0;
		D obj_temp3 = 0.0;
		D temp;
		for (unsigned int i = 0; i < instance.n; i++) {
			temp = instance.x[i] + deltaAlpha[i];
			obj_temp1 += 0.5 * temp * temp - temp * instance.b[i];
		}

		std::vector<D> Apotent(instance.m);
		cblas_set_to_zero(Apotent);
		vectormatrix_b(deltaAlpha, instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
		               instance.b, 1.0, instance.n, Apotent);

		D g_norm = cblas_l2_norm(instance.m, &Apotent[0], 1);
		obj_temp3 = g_norm * g_norm;

		for (L i = 0; i < instance.m; i++)
			obj_temp2 += w[i] * Apotent[i];
		
		D con = 1.0 / instance.total_n / instance.lambda;

		potent_obj = 1.0 / instance.total_n * (obj_temp1 + obj_temp2 
			+ 0.5 * instance.penalty * con * obj_temp3);

	}

	virtual void compute_subproproblem_gradient(ProblemData<L, D> &instance,
	        std::vector<D> &gradient, std::vector<D> &deltaAlpha, std::vector<D> &w) {

		std::vector<D> gradient_temp1(instance.n);
		std::vector<D> gradient_temp2(instance.m);
		std::vector<D> gradient_temp3(instance.n);

		matrixvector(instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr, w, instance.n, gradient_temp1);
		vectormatrix_b(deltaAlpha, instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
		               instance.b, 1.0, instance.n, gradient_temp2);
		matrixvector(instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
		             gradient_temp2, instance.n, gradient_temp3);
		D temp = 1.0 / instance.total_n / instance.lambda;

		for (L i = 0; i < instance.n; i++) {
			gradient[i] = 1.0 / instance.total_n * ( gradient_temp1[i] * instance.b[i] + instance.x[i] - instance.b[i]
			              + 1.0 * instance.penalty * temp * gradient_temp3[i] * instance.b[i] + deltaAlpha[i] );
		}

	}

	virtual void compute_subproproblem_obj_LineSearch(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha, 
						std::vector<D> &Apotent, std::vector<D> &w, D &potent_obj) {
		D obj_temp1 = 0.0;
		D obj_temp2 = 0.0;
		D obj_temp3 = 0.0;
		D temp;
		for (unsigned int i = 0; i < instance.n; i++) {
			temp = instance.x[i] + deltaAlpha[i];
			obj_temp1 += 0.5 * temp * temp - temp * instance.b[i];
		}

		D g_norm = cblas_l2_norm(instance.m, &Apotent[0], 1);
		obj_temp3 = g_norm * g_norm;

		for (L i = 0; i < instance.m; i++)
			obj_temp2 += w[i] * Apotent[i];
		
		D con = 1.0 / instance.total_n / instance.lambda;

		potent_obj = 1.0 / instance.total_n * (obj_temp1 + obj_temp2 
			+ 0.5 * instance.penalty * con * obj_temp3);

	}

	virtual void compute_subproproblem_gradient_LineSearch(ProblemData<L, D> &instance, std::vector<D> &gradient, 
		std::vector<D> &gradient_temp1, std::vector<D> &gradient_temp3, std::vector<D> &deltaAlpha, std::vector<D> &w) {

		D temp = 1.0 / instance.total_n / instance.lambda;

		for (L i = 0; i < instance.n; i++) {
			gradient[i] = 1.0 / instance.total_n * ( gradient_temp1[i] * instance.b[i] + instance.x[i] - instance.b[i]
			              + 1.0 * instance.penalty * temp * gradient_temp3[i] * instance.b[i] + deltaAlpha[i] );
		}

	}

	virtual void backtrack_linesearch(ProblemData<L, D> &instance,
	                                  std::vector<D> &deltaAlpha, std::vector<D> &search_direction, std::vector<D> &w, 
	                                  D &dualobj, D &a) {

		D rho = 0.8;
		D c1ls = 0.01;
		//a = 10000.0;
		int iter = 0;
		std::vector<D> potent(instance.n);
		D gg;
		gg = cblas_l2_norm(instance.n, &search_direction[0], 1);
		std::vector<D> AdeltaAlpha(instance.m);
		std::vector<D> Ad(instance.m);
		std::vector<D> Apotent(instance.m);
		vectormatrix_b(deltaAlpha, instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
		               instance.b, 1.0, instance.n, AdeltaAlpha);
		vectormatrix_b(search_direction, instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
		               instance.b, 1.0, instance.n, Ad);

		while (1) {

			if (iter > 20) {
				//cout<<a<<endl;
				//a = 1.0;
				cblas_dcopy(instance.n, &potent[0], 1, &deltaAlpha[0], 1);
				break;
			}

			for (L i = 0; i < instance.n; i++) {
				potent[i] = deltaAlpha[i] - a * search_direction[i];
			}			
			for (L i = 0; i < instance.m; i++) {
				Apotent[i] = AdeltaAlpha[i] - a * Ad[i];
			}

			D obj;

			this->compute_subproproblem_obj_LineSearch(instance, potent, Apotent, w, obj);

			//cout<<a<<"    "<<obj<<"    "<<dualobj - c1ls * a * gg * gg<<endl;
			if (obj <= dualobj - c1ls * a * gg * gg) {
				//cout<<a<<"    "<<obj<<"  "<<dualobj - c1ls * a * gg * gg<<"    "<<a<<endl;
				cblas_dcopy(instance.n, &potent[0], 1, &deltaAlpha[0], 1);
				// cout << a << "  " << iter << endl;
				dualobj = obj;
				break;
			}
			a = rho * a;
			iter += 1;
		}
	}

	virtual void wolfe_linesearch(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha, 
			std::vector<D> &search_direction, std::vector<D> &gradient_temp1, 
			std::vector<D> &AdeltaAlpha, std::vector<D> &Ad, std::vector<D> &AAdeltaAlpha, std::vector<D> &AAd, 
			std::vector<D> &w, D &dualobj, D &a) {

		D a0 = 0;
		D a1 = 1.0;
		D amax = 10.0;
		D c1ls = 0.01;    // parameter for sufficient decrease condition
		D c2ls = 0.2;     // parameter for curvature condition
		int iter = 0;
		
		D objCurrent, objPrevious, objNext, gradD, gradDNext;
		std::vector<D> potent(instance.n);
		std::vector<D> grad(instance.n);
		std::vector<D> gradNext(instance.n);
		
		std::vector<D> Apotent(instance.m);
		std::vector<D> AApotent(instance.n);


		if (c1ls > c2ls)
			printf("the parameter c1ls should be less that c2ls;");

		this->compute_subproproblem_obj_LineSearch(instance, deltaAlpha, AdeltaAlpha, w, objCurrent);
		objPrevious = objCurrent;
		this->compute_subproproblem_gradient_LineSearch(instance, grad, gradient_temp1, AAdeltaAlpha, deltaAlpha ,w);
		gradD = cblas_ddot(instance.n, &grad[0], 1, &search_direction[0], 1);

		while (1){
			for (L i = 0; i < instance.n; i++) {
				potent[i] = deltaAlpha[i] - a1 * search_direction[i];
				AApotent[i] = AAdeltaAlpha[i] - a1 * AAd[i];
			}
			for (L i = 0; i < instance.m; i++) {
				Apotent[i] = AdeltaAlpha[i] - a1 * Ad[i];
			}
			this->compute_subproproblem_obj_LineSearch(instance, potent, Apotent, w, objNext);

			if (objNext > objCurrent - c1ls * a1 * gradD || (objNext >= objPrevious && iter > 1)){
				zoom(instance, w, a0, a1, deltaAlpha, search_direction, objCurrent, gradD, c1ls, c2ls, a, 
						AdeltaAlpha, Ad, AAdeltaAlpha, AAd, gradient_temp1);
				for (L i = 0; i < instance.n; i++) 
					deltaAlpha[i] -= a * search_direction[i];
				break;
			}
			this->compute_subproproblem_gradient_LineSearch(instance, gradNext, gradient_temp1, AApotent, potent, w);
			gradDNext = cblas_ddot(instance.n, &gradNext[0], 1, &search_direction[0], 1);

			if (abs(gradDNext) <= c2ls * gradD){
				a = a1;
				cblas_dcopy(instance.n, &potent[0], 1, &deltaAlpha[0], 1);
				break;
			}

			if (gradDNext <= 0){
				zoom(instance, w, a1, a0, deltaAlpha, search_direction, objCurrent, gradD, c1ls, c2ls, a,
						AdeltaAlpha, Ad, AAdeltaAlpha, AAd, gradient_temp1);
				for (L i = 0; i < instance.n; i++) 
					deltaAlpha[i] -= a * search_direction[i];
				break;
			}

			if (a1 >= (1 - 1e-1) * amax){
				a = 1.0;
				for (L i = 0; i < instance.n; i++) 
					deltaAlpha[i] -= a * search_direction[i];
				break;
			}
			a0 = a1;
			a1 = 0.5 * (a1 + amax);
			iter++;
		}
	}


	virtual void zoom(ProblemData<L, D> &instance, std::vector<D> &w, 
		D &aLow, D &aHigh, std::vector<D> &x,
		std::vector<D> &d, D &obj, D &gradD, D &c1, D &c2, D &a,
		std::vector<D> &Ax, std::vector<D> &Ad, std::vector<D> &AAx, std::vector<D> &AAd, std::vector<D> &gradient_temp1){

		std::vector<D> potentMid(instance.n);
		std::vector<D> potentLow(instance.n);
		std::vector<D> ApotentMid(instance.m);
		std::vector<D> ApotentLow(instance.m);
		std::vector<D> AApotentMid(instance.n);
		std::vector<D> AApotentLow(instance.n);
		std::vector<D> gradMid(instance.n);
		D objLow, objMid, gradDMid;
		int iter = 0;
		
		while (1){
			iter++;

			D aMid = 0.5 * (aLow + aHigh);
			for (L i = 0; i < instance.n; i++) {
				potentMid[i] = x[i] - aMid * d[i];
				AApotentMid[i] = AAx[i] - aMid * AAd[i];
				potentLow[i] = x[i] - aLow * d[i];				
				AApotentLow[i] = AAx[i] - aLow * AAd[i];
			}
			for (L i = 0; i < instance.m; i++) {
				ApotentMid[i] = Ax[i] - aMid * Ad[i];
				ApotentLow[i] = Ax[i] - aLow * Ad[i];
			}
			this->compute_subproproblem_obj_LineSearch(instance, potentMid, ApotentMid, w, objMid);
			this->compute_subproproblem_obj_LineSearch(instance, potentLow, ApotentLow, w, objLow);

			if (objLow > obj - c1 * aMid * gradD || objMid >= objLow)
				aHigh = aMid;
			else{

				this->compute_subproproblem_gradient_LineSearch(instance, gradMid, gradient_temp1, AApotentMid, potentMid, w);
				gradDMid = cblas_ddot(instance.n, &gradMid[0], 1, &d[0], 1);
				if (abs(gradDMid) <= c2 * gradD){
					a = aMid;
					break;
				}
				if (gradDMid * (aHigh - aLow) <= 0)
					aHigh = aLow;
				
				aLow = aMid;
			}

			if (abs(aLow - aHigh) < 1e-2) {
				a = 0.5 * (aLow + aHigh);	
				break;
			}

		}

	}


	virtual void LBFGS_update(ProblemData<L, D> &instance, std::vector<D> &search_direction, std::vector<D> old_grad,
	                          std::vector<D> &sk, std::vector<D> &rk, std::vector<D> gradient, std::vector<D> &oneoversy,
	                          L iter_counter, int limit_BFGS, int flag_BFGS) {

		L startPoint;
		int flag_old;

		cblas_dcopy(instance.n, &gradient[0], 1, &search_direction[0], 1);

		if (iter_counter > 0) {

			if (flag_BFGS > 0)
				flag_old = flag_BFGS - 1;
			else
				flag_old = limit_BFGS - 1;

			startPoint = instance.n * flag_old;
			for (L i = 0; i < instance.n; i++)
				rk[startPoint + i] = gradient[i] - old_grad[i];
			// for (int i=0; i<10; i++)
			// 	cout<<rk[instance.n * i + 11] <<"  ";
			// cout<<flag_old<<endl;
			oneoversy[flag_old] = 1.0 / cblas_ddot(instance.n, &rk[startPoint], 1, &sk[startPoint], 1);
//cout<<1./oneoversy[flag_old]<<endl;
			std::vector<D> aa(limit_BFGS);

			int kai = flag_BFGS;
			int wan = kai + limit_BFGS - 1;
			if (iter_counter < limit_BFGS) {
				kai = 0;
				wan = flag_BFGS - 1;
			}
			//double n1, n2;
			D bb = 0.0;
			for (L i = kai; i <= wan; i++) {
				L ii = wan - (i - kai);
				if (ii >= limit_BFGS)
					ii = ii - limit_BFGS;
				aa[ii] = oneoversy[ii] * cblas_ddot(instance.n, &sk[instance.n * ii], 1, &search_direction[0], 1);
				//n1 = cblas_dnrm2(instance.n, &search_direction[0], 1);
				cblas_daxpy(instance.n, -aa[ii], &rk[instance.n * ii], 1, &search_direction[0], 1);
				//n2 = cblas_dnrm2(instance.n, &search_direction[0], 1);	
			}

			D Hk_zero = 0.0;
			//D t1 = cblas_ddot(instance.n, &sk[startPoint], 1, &rk[startPoint], 1);
			D t2 = cblas_dnrm2(instance.n, &rk[startPoint], 1);
			Hk_zero = 1.0 / oneoversy[flag_old] / t2 / t2;
			if (Hk_zero >1000.0)
				Hk_zero = 1000.0;
			else if (Hk_zero < 0.001)
				Hk_zero = 0.001;
			//cout<<oneoversy[0] << "   "<<t2<<endl;

			cblas_dscal(instance.n, Hk_zero, &search_direction[0], 1);
			//double n3 = cblas_dnrm2(instance.n, &search_direction[0], 1);   
			for (L i = kai; i <= wan; i++) {
				L ii = i;
				if (i >= limit_BFGS)
					ii = i - limit_BFGS;

				bb = oneoversy[ii] * cblas_ddot(instance.n, &rk[instance.n * ii], 1, &search_direction[0], 1);
				cblas_daxpy(instance.n, (aa[ii] - bb), &sk[instance.n * ii], 1, &search_direction[0], 1);
			}
			//double n4 = cblas_dnrm2(instance.n, &search_direction[0], 1);   
			//cout<<oneoversy[flag_old]<<"  "<<sk[startPoint+11]<<"  "<< rk[startPoint+11]<<"  "<< gradient[11]<<"  "<< old_grad[11]<<endl;
			//cout<<n1<<"  "<<n2<<"  "<<n3<<"  "<<n4<<"  "<<aa[0]<<"  "<<bb<<"       "<<Hk_zero<<endl;
			//cout<<sk[1]<<"   "<<sk[2]<<endl;
		}
		// for (L i = 0; i < instance.n; i++){

		// 	if (iter_counter > 0){
		// 		if (flag_BFGS > 0)
		// 			rk[instance.n * (flag_BFGS - 1)  + i] = gradient[i] - old_grad[i];
		// 		else
		// 			rk[instance.n * (limit_BFGS - 1)  + i] = gradient[i] - old_grad[i];
		// 	}
		// 	search_direction[i] = gradient[i];
		// }

// 		if (iter_counter > 0) {

// 			int flag_old = flag_BFGS - 1;
// 			if (flag_BFGS == 0)
// 				flag_old = limit_BFGS - 1;

// 			oneoversy[flag_old] = 0.0;
// 			for (L idx = 0; idx < instance.n; idx++){
// 				oneoversy[flag_old] += rk[instance.n * flag_old + idx]* sk[instance.n * flag_old + idx];
// 			}
// //cout<<rk[instance.n * flag_old + 11]<<"     ";
// 			oneoversy[flag_old] = 1.0 / oneoversy[flag_old];

// 			L kai = flag_BFGS;
// 			L wan = kai + limit_BFGS - 1;// min(kai + 9, kai + iter_counter - 1);
// 			std::vector<D> aa(limit_BFGS);
// 			cblas_set_to_zero(aa);
// 			D bb = 0.0;
// 			//cout<<kai<<"    "<<wan<<"    "<<iter_counter<<endl;

// 			if (iter_counter < limit_BFGS) {
// 				kai = 0;
// 				wan = flag_BFGS - 1;
// 			}

// 			for (L i = kai; i <= wan; i++){
// 				L ii = wan - (i-kai);
// 				if (ii >= limit_BFGS)
// 					ii = ii - 10;

// 				for (L j = 0; j < instance.n; j++)
// 					aa[ii] += sk[instance.n * ii + j] * search_direction[j];
// 				aa[ii] *= oneoversy[ii];
// 				for (L idx = 0; idx < instance.n; idx++)
// 					search_direction[idx] -= aa[ii] * rk[instance.n * ii + idx];
// 			}
// 			D Hk_zero = 0.0;
// 			D t1 = 0.0;
// 			D t2 = 0.0;
// 			for (L idx = instance.n * flag_old; idx < instance.n * (flag_old + 1); idx++){
// 				t1 += sk[idx] * rk[idx];
// 				t2 += rk[idx] * rk[idx];
// 			}

// 			Hk_zero = t1 / t2;
// 			//cout<< Hk_zero << "   "<<t2<<endl;
// 			for (L idx = 0; idx < instance.n; idx++)
// 				search_direction[idx] = Hk_zero * search_direction[idx];
// 			for (L i = kai; i <= wan; i++){
// 				L ii = i;
// 				if (i >= limit_BFGS)
// 					ii = i - limit_BFGS;

// 				bb = 0.0;
// 				for (L j = 0; j < instance.n; j++)
// 					bb += rk[instance.n * ii + j] * search_direction[j];
// 				bb *= oneoversy[ii];
// 				for (L j = 0; j < instance.n; j++){
// 					search_direction[j] += sk[instance.n * ii + j] * (aa[ii] - bb);
// 				}
// 			}

	}


};

#endif /* QUADRATICLOSS_H_ */
