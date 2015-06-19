/*
 * cdn_common.h
 *
 *  Created on: Jan 30, 2015
 *      Author: taki
 */

#ifndef CDN_COMMON_H_
#define CDN_COMMON_H_

#include "../helpers/c_libs_headers.h"
#include "../utils/randomNumbersUtil.h"
#include <gsl/gsl_linalg.h>

#ifdef MKL
#include "mkl_lapacke.h"
#else
#include "lapacke.h"
#endif

#include <gsl/gsl_cblas.h>
#include "../utils/my_cblas_wrapper.h"
#include "../helpers/utils.h"

#include "../utils/distributed_instances_loader.h"

template<typename L, typename D>
D computeDualityGapSparse(L m, L n, ProblemData<int, double>& part,
		std::vector<D> &b, std::vector<D> &x, std::vector<D> &w, D& lambda,
		D& primal, D&dual, D& lonezeroloss) {

	primal = 0;
	dual = 0;
	lonezeroloss = 0;

	L correct = 0;
	for (int i = 0; i < part.n; i++) {
		D tmp = 0;
		for (int j = part.A_csr_row_ptr[i]; j < part.A_csr_row_ptr[i + 1];
				j++) {
			tmp += part.A_csr_values[j] * w[part.A_csr_col_idx[j]];
		}
		if (tmp * b[i] > 0) {
			correct++;
		}

		primal += 0.5 * (tmp - b[i]) * (tmp - b[i]);

		dual += 0.5 * (x[i] * x[i]) - x[i] * b[i];
	}
	D norm = cblas_l2_norm(m, &w[0], 1);
	lonezeroloss = correct/(part.n+0.0);
	primal = 1 / (0.0 + part.n) * primal + lambda * 0.5 * norm * norm;
	dual = 1 / (0.0 + part.n) * dual + lambda * 0.5 * norm * norm;
	return primal + dual;

}
template<typename L, typename D>
D computeDualityGapSparse(L m, L n, ProblemData<int, double>& part,
		std::vector<D> &b, std::vector<D> &x, std::vector<D> &w, D& lambda,
		D& primal, D&dual) {

	primal = 0;
	dual = 0;

	for (int i = 0; i < n; i++) {
		D tmp = 0;
		for (int j = part.A_csr_row_ptr[i]; j < part.A_csr_row_ptr[i + 1];
				j++) {
			tmp += part.A_csr_values[j] * w[part.A_csr_col_idx[j]];
		}
		primal += 0.5 * (tmp - b[i]) * (tmp - b[i]);

		dual += 0.5 * (x[i] * x[i]) - x[i] * b[i];
	}
	D norm = cblas_l2_norm(m, &w[0], 1);

	primal = 1 / (0.0 + n) * primal + lambda * 0.5 * norm * norm;
	dual = 1 / (0.0 + n) * dual + lambda * 0.5 * norm * norm;
	return primal + dual;

}
template<typename L, typename D>
D computeDualityGap(L m, L n, std::vector<D> &A, std::vector<D> &b,
		std::vector<D> &x, std::vector<D> &w, D& lambda, D& primal, D&dual) {

	primal = 0;
	dual = 0;

	for (int i = 0; i < n; i++) {
		D tmp = 0;
		for (int j = 0; j < m; j++) {
			tmp += A[i * m + j] * w[j];
		}
		primal += 0.5 * (tmp - b[i]) * (tmp - b[i]);

		dual += 0.5 * (x[i] * x[i]) - x[i] * b[i];
	}
	D norm = cblas_l2_norm(m, &w[0], 1);

	primal = 1 / (0.0 + n) * primal + lambda * 0.5 * norm * norm;
	dual = 1 / (0.0 + n) * dual + lambda * 0.5 * norm * norm;
	return primal + dual;

}

template<typename L, typename D>
void runPCDMExperiment(L m, L n, std::vector<D>& A, std::vector<D>& b,
		D & lambda, int tau, ofstream& logFile, std::vector<D> & Li, D& sigma,
		D maxTime) {

	std::vector < D > x(n, 0);
	for (int i = 0; i < x.size(); i++) {
		x[i] = 0;
	}
	std::vector < D > w(m, 0);
	for (int i = 0; i < w.size(); i++) {
		w[i] = 0;
	}

	D primal;
	D dual;
	D gap;
	gap = computeDualityGap(m, n, A, b, x, w, lambda, primal, dual);
	cout << "Duality Gap: " << gap << "   " << primal << "   " << dual << endl;

	logFile << setprecision(16) << "0," << tau << "," << m << "," << n << ","
			<< lambda << "," << primal << "," << dual << "," << 0 << endl;

	std::vector<D> deltaAlpha(tau);
	std::vector<int> S(tau);

	double scaling = 1 / (lambda * n);
	double scaling2 = sigma / (lambda * n);
	double elapsedTime = 0;
	double start;

	long long it = 0;
	for (;;) {
		it = it + tau;
		for (int i = 0; i < tau; i++) {
			bool done = true;
			do {
				done = true;
				S[i] = gsl_rng_uniform_int(gsl_rng_r, n);
				for (int j = 0; j < i; j++) {

					if (S[i] == S[j]) {
						done = false;
						break;
					}
				}

			} while (!done);
		}
		start = gettime_();

//#pragma omp parallel for
		for (int i = 0; i < tau; i++) {
			deltaAlpha[i] = 0;
			D w_xi = cblas_ddot(m, &A[S[i] * m + 0], 1, &w[0], 1);

//			for (int j = 0; j < m; j++) {
//				w_xi += A[S[i] * m + j] * w[j];
//			}

// minimize
//			   h_i     =    (y_i - w_xi - x_i) / (1+   1/( lambda n Li)     );
			deltaAlpha[i] = (b[S[i]] - w_xi - x[S[i]])
					/ (1 + Li[S[i]] * scaling2);

		}

//#pragma omp parallel for
		for (int i = 0; i < tau; i++) {
			x[S[i]] = x[S[i]] + deltaAlpha[i];

			cblas_daxpy(m, deltaAlpha[i] * scaling, &A[S[i] * m], 1, &w[0], 1);

		}

		elapsedTime += (gettime_() - start);

		if ((it + tau) % n < tau) {
			gap = computeDualityGap(m, n, A, b, x, w, lambda, primal, dual);
			cout << "Duality Gap: " << gap << "   " << primal << "   " << dual
					<< "  " << elapsedTime << endl;
			logFile << setprecision(16) << "0," << tau << "," << m << "," << n
					<< "," << lambda << "," << primal << "," << dual << ","
					<< elapsedTime << endl;
		}

		if (elapsedTime > maxTime || gap < 0.000000000000001) {
			break;
		}
	}

}

template<typename L, typename D>
void runPCDMExperimentSparse(L m, L n, ProblemData<int, double>& part,
		std::vector<D>& b, D & lambda, int tau, ofstream& logFile,
		std::vector<D> & Li, D& sigma, D maxTime) {

	cout << "Solver starts " << m << " x " << n << endl;

	std::vector < D > x(n, 0);
	for (int i = 0; i < x.size(); i++) {
		x[i] = 0;
	}
	std::vector < D > w(m, 0);
	for (int i = 0; i < w.size(); i++) {
		w[i] = 0;
	}

	D primal;
	D dual;
	D gap;
	gap = computeDualityGapSparse(m, n, part, b, x, w, lambda, primal, dual);
	cout << "Duality Gap: " << gap << "   " << primal << "   " << dual << endl;

	logFile << setprecision(16) << "0," << tau << "," << m << "," << n << ","
			<< lambda << "," << primal << "," << dual << "," << 0 << endl;

	std::vector<D> deltaAlpha(tau);
	std::vector<int> S(tau);

	double scaling = 1 / (lambda * n);
	double scaling2 = sigma / (lambda * n);
	double elapsedTime = 0;
	double start;

	long long it = 0;
	for (;;) {
		it = it + tau;

		for (int i = 0; i < tau; i++) {
			bool done = true;
			do {
				done = true;
				S[i] = gsl_rng_uniform_int(gsl_rng_r, n);
				for (int j = 0; j < i; j++) {

					if (S[i] == S[j]) {
						done = false;
						break;
					}
				}

			} while (!done);
		}

		start = gettime_();

//#pragma omp parallel for
		for (int i = 0; i < tau; i++) {
			deltaAlpha[i] = 0;

			D w_xi = 0; //cblas_ddot(m, &A[S[i] * m + 0], 1, &w[0], 1);
			for (int j = part.A_csr_row_ptr[S[i]];
					j < part.A_csr_row_ptr[S[i] + 1]; j++) {
				w_xi += part.A_csr_values[j] * w[part.A_csr_col_idx[j]];
			}

// minimize
//			   h_i     =    (y_i - w_xi - x_i) / (1+   1/( lambda n Li)     );
			deltaAlpha[i] = (b[S[i]] - w_xi - x[S[i]])
					/ (1 + Li[S[i]] * scaling2);

		}

//#pragma omp parallel for
		for (int i = 0; i < tau; i++) {
			x[S[i]] = x[S[i]] + deltaAlpha[i];

			for (int j = part.A_csr_row_ptr[S[i]];
					j < part.A_csr_row_ptr[S[i] + 1]; j++) {
				w[part.A_csr_col_idx[j]] += scaling * deltaAlpha[i]
						* part.A_csr_values[j];
			}

		}

		elapsedTime += (gettime_() - start);

		if ((it + tau) % n < tau) {
			gap = computeDualityGapSparse(m, n, part, b, x, w, lambda, primal,
					dual);
			cout << "Duality Gap: " << it << "  " << gap << "   " << primal
					<< "   " << dual << "  " << elapsedTime << endl;
			logFile << setprecision(16) << "0," << tau << "," << m << "," << n
					<< "," << lambda << "," << primal << "," << dual << ","
					<< elapsedTime << endl;
		}

		if (elapsedTime > maxTime || gap < 0.000000000000001) {
			break;
		}
	}

}

template<typename L, typename D>
void runCDNExperiment(L m, L n, std::vector<D>& A, std::vector<D>& b,
		D & lambda, int tau, ofstream& logFile, int type, D maxTime,
		std::vector<D>& Hessian) {

	bool hessianPrecomputed = (Hessian.size() > 0);

	std::vector < D > x(n, 0);
	for (int i = 0; i < x.size(); i++) {
		x[i] = 0;
	}
	std::vector < D > w(m, 0);
	for (int i = 0; i < w.size(); i++) {
		w[i] = 0;
	}

	D primal;
	D dual;
	D gap;
	gap = computeDualityGap(m, n, A, b, x, w, lambda, primal, dual);
	cout << "Duality Gap: " << gap << "   " << primal << "   " << dual << endl;

	logFile << setprecision(16) << type << "," << tau << "," << m << "," << n
			<< "," << lambda << "," << primal << "," << dual << "," << 0
			<< endl;

	std::vector<D> Qdata(tau * tau, 0);
//	std::vector<D> Qb(tau);
	std::vector<D> bS(tau);
	std::vector<int> S(tau);
	gsl_vector *T = gsl_vector_alloc(tau);
	gsl_permutation * p = gsl_permutation_alloc(tau);
	std::vector < D > AS(m * tau);
	gsl_matrix_view mm = gsl_matrix_view_array(&Qdata[0], tau, tau);
	gsl_vector_view bb = gsl_vector_view_array(&bS[0], tau);

	double scaling = 1 / (lambda * n);

	int info;
	double elapsedTime = 0;
	double start;

	long long it = 0;
	for (;;) {
		it = it + tau;
		for (int i = 0; i < tau; i++) {
			bool done = true;
			do {
				done = true;
				S[i] = gsl_rng_uniform_int(gsl_rng_r, n);
				for (int j = 0; j < i; j++) {

					if (S[i] == S[j]) {
						done = false;
						break;
					}
				}

			} while (!done);
		}

		if (hessianPrecomputed) {
			for (int row = 0; row < tau; row++) {
				for (int col = row; col < tau; col++) {
					D tmp = scaling * Hessian[S[row] * n + S[col]];
					Qdata[row * tau + col] = tmp;
					Qdata[col * tau + row] = tmp;
				}
			}
		} else {
			for (int i = 0; i < tau; i++) {
				cblas_dcopy(m, &A[S[i] * m], 1, &AS[i * m], 1);
			}
			cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, tau, tau, m,
					1 / (lambda * n + 0.0), &AS[0], m, &AS[0], m, 0.0,
					&Qdata[0], tau);

		}

		start = gettime_();

//
//#pragma omp parallel for
//

		// bS = -A_{S} * w    .....		y := alpha*A*x + beta*y,
//		cblas_dgemv(CblasColMajor, CblasTrans, m, tau, -1.0, &AS[0], m, &w[0],
//				1, 0.0, &bS[0], 1);

		for (int i = 0; i < tau; i++) {
			bS[i] = -cblas_ddot(m, &A[S[i] * m + 0], 1, &w[0], 1);

		}
//			for (int j = 0; j < m; j++) {
//				w_xi += A[S[i] * m + j] * w[j];
//			}

// minimize
//			   h_i     =    (y_i - w_xi - x_i) / (1+   1/( lambda n Li)     );
//			deltaAlpha[i] = (b[S[i]] - w_xi - x[S[i]])
//					/ (1 + Li[S[i]] * scaling2);
//
//		}
//
//#pragma omp parallel for
//		for (int i = 0; i < tau; i++) {
//			x[S[i]] = x[S[i]] + deltaAlpha[i];
//
//			cblas_daxpy(m, deltaAlpha[i] * scaling, &A[S[i] * m], 1, &w[0], 1);
//
//		}

		//  1/lambda n   h^T Ms H

		for (int i = 0; i < tau; i++) {
			//bS = bS +  x_i    - y_i
			bS[i] += -x[S[i]] + b[S[i]];
			Qdata[i + i * tau] += 1;  //+  h_i
		}

		if (type == 1) {

			int s;
			gsl_linalg_LU_decomp(&mm.matrix, p, &s);
			gsl_linalg_LU_solve(&mm.matrix, p, &bb.vector, T);

			for (int i = 0; i < tau; i++) {
				x[S[i]] = x[S[i]] + T->data[i];
				cblas_daxpy(m, T->data[i] * scaling, &A[S[i] * m], 1, &w[0], 1);
			}
//			cblas_dgemv(CblasColMajor, CblasNoTrans, m, tau, scaling, &AS[0], m,
//					&(T->data[0]), 1, 1, &w[0], 1);

		} else {

			//	DGETRS
			//	DPOTRS
			//  DPPTRS
			//  DPTTRS
			//	dgels
			info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', tau, tau, 1, &Qdata[0],
					tau, &bS[0], tau);

			if (info > 0) {
				printf("The diagonal element %i of the triangular factor ",
						info);
				printf("of A is zero, so that A does not have full rank;\n");
				printf("the least squares solution could not be computed.\n");
				exit(1);
			}

			for (int i = 0; i < tau; i++) {
				x[S[i]] = x[S[i]] + bS[i];
				cblas_daxpy(m, bS[i] * scaling, &A[S[i] * m], 1, &w[0], 1);

			}

//			cblas_dgemv(CblasColMajor, CblasNoTrans, m, tau, scaling, &AS[0], m,
//					&bS[0], 1, 1, &w[0], 1);

//
//				for (int j = 0; j < m; j++) {
//					w[j] += scaling * AS[i * m + j] * bS[i];
//				}

		}
		elapsedTime += (gettime_() - start);
		if ((it + tau) % n < tau) {

			gap = computeDualityGap(m, n, A, b, x, w, lambda, primal, dual);
			cout << "Duality Gap: " << it << " , " << gap << "   " << primal
					<< "   " << dual << "  " << elapsedTime << endl;
			logFile << setprecision(16) << type << "," << tau << "," << m << ","
					<< n << "," << lambda << "," << primal << "," << dual << ","
					<< elapsedTime << endl;
		}

		if (elapsedTime > maxTime || gap < 0.000000000000001) {
			break;
		}
	}

	gsl_permutation_free(p);
	gsl_vector_free(T);

}

template<typename L, typename D>
void runCDNExperimentSparse(L m, L n, ProblemData<int, double>& part,
		std::vector<D>& b, D & lambda, int tau, ofstream& logFile, int type,
		D maxTime, std::vector<D>& Hessian) {
	bool hessianPrecomputed = (Hessian.size() > 0);
	std::vector < D > x(n, 0);
	for (int i = 0; i < x.size(); i++) {
		x[i] = 0;
	}
	std::vector < D > w(m, 0);
	for (int i = 0; i < w.size(); i++) {
		w[i] = 0;
	}

	D primal;
	D dual;
	D gap;
	gap = computeDualityGapSparse(m, n, part, b, x, w, lambda, primal, dual);
	cout << "Duality Gap: " << gap << "   " << primal << "   " << dual << endl;

	logFile << setprecision(16) << type << "," << tau << "," << m << "," << n
			<< "," << lambda << "," << primal << "," << dual << "," << 0
			<< endl;

	std::vector<D> Qdata(tau * tau, 0);
	std::vector<D> bS(tau);
	std::vector<int> S(tau);
	gsl_vector *T = gsl_vector_alloc(tau);
	gsl_permutation * p = gsl_permutation_alloc(tau);
	std::vector < D > AS(m * tau);
	gsl_matrix_view mm = gsl_matrix_view_array(&Qdata[0], tau, tau);
	gsl_vector_view bb = gsl_vector_view_array(&bS[0], tau);

	double scaling = 1 / (lambda * n);

	int info;
	double elapsedTime = 0;
	double start;

	long long it = 0;
	for (;;) {
		it = it + tau;
		for (int i = 0; i < tau; i++) {
			bool done = true;
			do {
				done = true;
				S[i] = gsl_rng_uniform_int(gsl_rng_r, n);
				for (int j = 0; j < i; j++) {

					if (S[i] == S[j]) {
						done = false;
						break;
					}
				}

			} while (!done);
		}

		if (hessianPrecomputed) {

			for (int row = 0; row < tau; row++) {
				for (int col = row; col < tau; col++) {

					D tmp = Hessian[S[row] * n + S[col]] * scaling;

					Qdata[row * tau + col] = tmp;
					Qdata[col * tau + row] = tmp;
				}
			}
		} else {

			std::vector<double>& vals = part.A_csr_values;
			std::vector<int> &rowPtr = part.A_csr_row_ptr;
			std::vector<int> &colIdx = part.A_csr_col_idx;

			for (int row = 0; row < tau; row++) {
				for (int col = row; col < tau; col++) {

					double tmp = 0;

					int id1 = rowPtr[S[row]];
					int id2 = rowPtr[S[col]];

					while (id1 < rowPtr[S[row] + 1] && id2 < rowPtr[S[col] + 1]) {

						if (colIdx[id1] == colIdx[id2]) {
							tmp += vals[id1] * vals[id2];
							id1++;
							id2++;
						} else if (colIdx[id1] < colIdx[id2]) {
							id1++;
						} else {
							id2++;
						}

					}

					Qdata[row * tau + col] = tmp * scaling;
					Qdata[col * tau + row] = tmp * scaling;

				}
			}

		}

		start = gettime_();
		// bS = -A_{S} * w    .....		y := alpha*A*x + beta*y,
		for (int i = 0; i < tau; i++) {
			bS[i] = 0;
			for (int j = part.A_csr_row_ptr[S[i]];
					j < part.A_csr_row_ptr[S[i] + 1]; j++) {
				bS[i] -= part.A_csr_values[j] * w[part.A_csr_col_idx[j]];
			}
		}

		//  1/lambda n   h^T Ms H
//		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, tau, tau, m,
//				1 / (lambda * n + 0.0), &AS[0], m, &AS[0], m, 0.0, &Qdata[0],
//				tau);

		for (int i = 0; i < tau; i++) {
			//bS = bS +  x_i    - y_i
			bS[i] += -x[S[i]] + b[S[i]];
			Qdata[i + i * tau] += 1;  //+  h_i
		}

		if (type == 1) {

			int s;
			gsl_linalg_LU_decomp(&mm.matrix, p, &s);
			gsl_linalg_LU_solve(&mm.matrix, p, &bb.vector, T);

			for (int i = 0; i < tau; i++) {
				x[S[i]] = x[S[i]] + T->data[i];
			}

			for (int i = 0; i < tau; i++) {
				for (int j = part.A_csr_row_ptr[S[i]];
						j < part.A_csr_row_ptr[S[i] + 1]; j++) {
					w[part.A_csr_col_idx[j]] += scaling * part.A_csr_values[j]
							* T->data[i];
				}
			}

		} else {
			info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', tau, tau, 1, &Qdata[0],
					tau, &bS[0], tau);

			if (info > 0) {
				printf("The diagonal element %i of the triangular factor ",
						info);
				printf("of A is zero, so that A does not have full rank;\n");
				printf("the least squares solution could not be computed.\n");
				exit(1);
			}

			for (int i = 0; i < tau; i++) {
				x[S[i]] = x[S[i]] + bS[i];
			}

			for (int i = 0; i < tau; i++) {
				for (int j = part.A_csr_row_ptr[S[i]];
						j < part.A_csr_row_ptr[S[i] + 1]; j++) {
					w[part.A_csr_col_idx[j]] += scaling * part.A_csr_values[j]
							* bS[i];
				}
			}

//			cblas_dgemv(CblasColMajor, CblasNoTrans, m, tau, scaling, &AS[0], m,
//					&bS[0], 1, 1, &w[0], 1);

		}
		elapsedTime += (gettime_() - start);
		if ((it + tau) % n < tau) {

			gap = computeDualityGapSparse(m, n, part, b, x, w, lambda, primal,
					dual);
			cout << "Duality Gap: " << it << " , " << gap << "   " << primal
					<< "   " << dual << "  " << elapsedTime << endl;
			logFile << setprecision(16) << type << "," << tau << "," << m << ","
					<< n << "," << lambda << "," << primal << "," << dual << ","
					<< elapsedTime << endl;
		}

		if (elapsedTime > maxTime || gap < 0.000000000000001) {
			break;
		}
	}

	gsl_permutation_free(p);
	gsl_vector_free(T);

}

#include "../helpers/matrix_conversions.h"

#include "mkl_spblas.h"

#endif /* CDN_COMMON_H_ */
