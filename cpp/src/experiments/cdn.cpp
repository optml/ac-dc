/*
 * This file contains an experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */

#include "cdn_common.h"

void generateOrtogonalMatrix(std::vector<double> &V, int n) {
	V.resize(n * n);
	for (int i = 0; i < n; i++) {
		double norm = 0;
		for (int j = 0; j < n; j++) {
			V[i * n + j] = rand() / (0.0 + RAND_MAX);
			norm += V[i * n + j] * V[i * n + j];
		}
		cblas_vector_scale(n, &V[i * n], 1 / sqrt(norm));
		for (int k = 0; k < i; k++) {
			double projection = 0;
			for (int l = 0; l < n; l++) {
				projection += V[k * n + l] * V[i * n + l];
			}
			for (int l = 0; l < n; l++) {
				V[i * n + l] -= projection * V[k * n + l];
			}
		}
		norm = cblas_l2_norm(n, &V[i * n], 1);
		cblas_vector_scale(n, &V[i * n], 1/norm);
	}
}

int main(int argc, char * argv[]) {

//	omp_set_num_threads(24);
//	double sum = 0;
//	double add = 1;
//
//	double sst=gettime_();
//#pragma omp parallel for
//	for (long i = 0; i < 1000000; i++) {
//		parallel::atomic_add(sum, add);
//	}
//	cout << "total is "<<sum<<"  took "<< gettime_()-sst<<endl;
//
//
//	sum=0;
//	sst=gettime_();
//	#pragma omp parallel for
//		for (long i = 0; i < 1000000; i++) {
//			 sum+= add;
//		}
//		cout << "total is "<<sum<<"  took "<< gettime_()-sst<<endl;
//
//
//		sum=0;
//		sst=gettime_();
// 			for (long i = 0; i < 1000000; i++) {
//				 sum+= add;
//			}
//			cout << "total is "<<sum<<"  took "<< gettime_()-sst<<endl;
//
//
//
//return 0;
	std::string file;

	int RDN = 0;

	char c;
	while ((c = getopt(argc, argv, "A:R:")) != -1) {
		switch (c) {
		case 'A':
			file = optarg;
			break;
		case 'R':
			RDN = atoi(optarg);
			break;
		}
	}

//	file= "../../SVM/epsilon_normalized";
//	file = "data/dense.svm";
//	file = "data/a1a";
//	file = "../../SVM/random_dense";

	stringstream ss("");
	ss << file << "_logNEW2";

	ofstream logFile;
	logFile.open(ss.str().c_str());

	omp_set_num_threads(1);

	const int MAXIMUM_THREADS = 64;
	std::vector<gsl_rng *> rs = randomNumberUtil::inittializeRandomSeeds(
			MAXIMUM_THREADS);

	int n = 1000; // this value was used for experiments
	int m = n * 2;

	randomNumberUtil::init_random_seeds(rs);
//--------------------- run experiment - one can change precission here

//	string inputFile, int file, int totalFiles,
//			ProblemData<L, D> & part, bool zeroBased

	ProblemData<int, double> part;
	std::vector<double> A;
	std::vector<double> b;
	double lambda = 0;
	if (RDN == 0) {
		loadDistributedSparseSVMRowData(file, -1, -1, part, false);
		m = part.m;
		n = part.n;
//		std::vector<double> &A = part.A_csr_values;
//		std::vector<double> &b = part.b;
		lambda = 1 / (n + 0.0);
	} else {

		switch (RDN) {
		case 1:

			n = 2048;
			m = n / 2;

			break;

		case 2:

			n = 2048;
			m = n * 2;

			break;

		case 3:

			n = 2048;
			m = n;

			break;
		case 4:

			n = 2048 * 4;
			m = n / 2;

			break;

		case 5:

			n = 2048 * 4;
			m = n * 2;

			break;

		case 6:

			n = 2048 * 4;
			m = n;

			break;

		default:
			break;
		}

		A.resize(m * n, 0);
		b.resize(n);

		std::vector<double> V(n * n, 0);
		std::vector<double> U(m * m, 0);

		for (int i = 0; i < n; i++) {
			b[i] = -1 + 2 * round(rand() / (0.0 + RAND_MAX));
			for (int j = 0; j < m; j++) {
				A[i * m + j] = rand() / (0.0 + RAND_MAX);
//				if (i == 0) {
//					A[i * m + j] = -1 + 2 * rand() / (0.0 + RAND_MAX);
//				} else {
//
//					A[i * m + j] = A[j]
//							+ b[i] * 0.001 * rand() / (0.0 + RAND_MAX);
//
//				}
			}
		}

		generateOrtogonalMatrix(V, n);
		generateOrtogonalMatrix(U, m);

		int k = m;
		if (n < k) {
			k = n;
		}

		std::vector<double> S(k);
		double a = 1;
		int pos = 0;
		for (double l = 0.000000001; l < a; l = l + a / k) {
			S[pos] = exp(exp(l));
//			S[pos]=S[pos]*S[pos];
			pos++;
		}
		for (int i = 0; i++; i < 0) {
			for (int j = 0; j < m; j++) {
				A[i * m + j] = 0;
				for (int l = 0; l < k; l++) {
					A[i * m + j] += U[j * m + l] * V[i * n + l] * S[l];
				}
			}
		}


//		double maxNorm = 0;
//		for (int i = 0; i < n; i++) {
//			double norm = cblas_l2_norm(m, &A[m * i], 1);
//			if (norm > maxNorm) {
//				maxNorm = norm;
//			}
//		}
//		for (int i = 0; i < n; i++) {
////			cblas_vector_scale(m, &A[m * i], 1 / maxNorm);
//		}

		lambda = 1 / (n + 0.0);

	}
	std::vector<double> Li(n, 0);
	std::vector<double> LiSqInv(n, 0);

	bool dense = (m * n == A.size());

	if (dense) {
		cout << "Input data is dense!!!" << endl;
		for (int i = 0; i < n; i++) {
			Li[i] = 0;
			LiSqInv[i] = 0;
			for (int j = 0; j < m; j++) {
				Li[i] += A[i * m + j] * A[i * m + j];
			}
			if (Li[i] > 0) {
				LiSqInv[i] = 1 / sqrt(Li[i]);
			}
		}

	} else {
		cout << "Input data is sparse!!!  " << part.A_csr_row_ptr.size()
				<< endl;

		for (int i = 0; i < n; i++) {
			Li[i] = 0;
			LiSqInv[i] = 0;
			for (int j = part.A_csr_row_ptr[i]; j < part.A_csr_row_ptr[i + 1];
					j++) {
				Li[i] += part.A_csr_values[j] * part.A_csr_values[j];
			}
			if (Li[i] > 0) {
				LiSqInv[i] = 1 / sqrt(Li[i]);
			}
		}

	}
	cout << "Stage 2" << endl;

	std::vector<double> x(n);
	std::vector<double> y(m);

	for (int i = 0; i < n; i++) {
		x[i] = rand() / (0.0 + RAND_MAX);
	}
	double norm = cblas_l2_norm(n, &x[0], 1);
	cblas_vector_scale(n, &x[0], 1 / norm);

	double maxEig = 1;

	if (!dense) {
		ofstream logFileR;
		logFileR.open("/tmp/rows.txt");

		ofstream logFileC;
		logFileC.open("/tmp/cols.txt");

		ofstream logFileV;
		logFileV.open("/tmp/vals.txt");

		for (int row = 0; row < n; row++) {
			for (int tmo = part.A_csr_row_ptr[row];
					tmo < part.A_csr_row_ptr[row + 1]; tmo++) {
				logFileR << row + 1 << endl;
				logFileC << part.A_csr_col_idx[tmo] + 1 << endl;
				logFileV << part.A_csr_values[tmo] << endl;
			}
		}

		logFileR.close();

		logFileC.close();

		logFileV.close();
	}
	for (int PM = 0; PM < 20; PM++) {

		if (dense) {
			for (int j = 0; j < m; j++) {
				y[j] = 0;

				for (int i = 0; i < n; i++) {
					y[j] += A[i * m + j] * LiSqInv[i] * x[i];
				}
			}

		} else {
			for (int j = 0; j < m; j++) {
				y[j] = 0;
			}
			for (int i = 0; i < n; i++) {
				for (int j = part.A_csr_row_ptr[i];
						j < part.A_csr_row_ptr[i + 1]; j++) {
					y[part.A_csr_col_idx[j]] += part.A_csr_values[j]
							* LiSqInv[i] * x[i];
				}

			}

		}

		for (int i = 0; i < n; i++) {
			x[i] = 0;
			if (dense) {
				for (int j = 0; j < m; j++) {
					x[i] += A[i * m + j] * LiSqInv[i] * y[j];
				}
			} else {
				for (int j = part.A_csr_row_ptr[i];
						j < part.A_csr_row_ptr[i + 1]; j++) {
					x[i] += part.A_csr_values[j] * LiSqInv[i]
							* y[part.A_csr_col_idx[j]];
				}

			}

		}

		maxEig = cblas_l2_norm(n, &x[0], 1);
		cout << maxEig << endl;
		cblas_vector_scale(n, &x[0], 1 / maxEig);

	}

	cout << "Max eigenvalue estimated " << endl;

	x.resize(0);
	y.resize(0);
	LiSqInv.resize(0);

	double maxTime = 10000;

	std::vector<double> Hessian;
	if (n < 10000) {
		Hessian.resize(n * n);

		if (dense) {

			cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, m, 1.0,
					&A[0], m, &A[0], m, 0, &Hessian[0], n);

		} else {

			std::vector<double>& vals = part.A_csr_values;
			std::vector<int> &rowPtr = part.A_csr_row_ptr;
			std::vector<int> &colIdx = part.A_csr_col_idx;

			for (int row = 0; row < n; row++) {
				for (int col = row; col < n; col++) {

					double tmp = 0;

					int id1 = rowPtr[row];
					int id2 = rowPtr[col];

					while (id1 < rowPtr[row + 1] && id2 < rowPtr[col + 1]) {

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

					Hessian[row * n + col] = tmp;
					Hessian[col * n + row] = tmp;

				}
			}
		}
	}

	int MAXTAU = n / 2;
	if (MAXTAU > 1024 * 8) {
		MAXTAU = 1024 * 8;
	}

	for (int tau =1; tau <= MAXTAU; tau = tau * 2) {
//		int tau = n;
//		omp_set_num_threads(tau);

		double sigma = 1 + (tau - 1) * (maxEig - 1) / (n - 1.0);

		if (dense) {

			omp_set_num_threads(1);
			randomNumberUtil::init_random_seeds(rs);
			runPCDMExperiment(m, n, A, b, lambda, tau, logFile, Li, sigma,
					maxTime);
//
//			randomNumberUtil::init_random_seeds(rs);
//			runCDNExperiment(m, n, A, b, lambda, tau, logFile, 1, maxTime,
//					Hessian);

			for (int jj = 1; jj <= 32; jj = jj * 2) {
				omp_set_num_threads(jj);
				randomNumberUtil::init_random_seeds(rs);
				runCDNExperiment(m, n, A, b, lambda, tau, logFile, 2, maxTime,
						Hessian);

			}
		} else {

//			randomNumberUtil::init_random_seeds(rs);
//			runCDNExperimentSparse(m, n, part, part.b, lambda, tau, logFile, 1,
//					maxTime, Hessian);

			omp_set_num_threads(1);
			randomNumberUtil::init_random_seeds(rs);
			runCDNExperimentSparse(m, n, part, part.b, lambda, tau, logFile, 2,
					maxTime, Hessian);
			omp_set_num_threads(2);
			randomNumberUtil::init_random_seeds(rs);
			runCDNExperimentSparse(m, n, part, part.b, lambda, tau, logFile, 2,
					maxTime, Hessian);
			omp_set_num_threads(4);
			randomNumberUtil::init_random_seeds(rs);
			runCDNExperimentSparse(m, n, part, part.b, lambda, tau, logFile, 2,
					maxTime, Hessian);
			omp_set_num_threads(8);
			randomNumberUtil::init_random_seeds(rs);
			runCDNExperimentSparse(m, n, part, part.b, lambda, tau, logFile, 2,
					maxTime, Hessian);
			omp_set_num_threads(16);
			randomNumberUtil::init_random_seeds(rs);
			runCDNExperimentSparse(m, n, part, part.b, lambda, tau, logFile, 2,
					maxTime, Hessian);
			omp_set_num_threads(32);
			randomNumberUtil::init_random_seeds(rs);
			runCDNExperimentSparse(m, n, part, part.b, lambda, tau, logFile, 2,
					maxTime, Hessian);

			omp_set_num_threads(1);
			randomNumberUtil::init_random_seeds(rs);
			runPCDMExperimentSparse(m, n, part, part.b, lambda, tau, logFile,
					Li, sigma, maxTime);
		}
	}

	logFile.close();

//	histogramLogFile.close();
//	experimentLogFile.close();
	return 0;
}
