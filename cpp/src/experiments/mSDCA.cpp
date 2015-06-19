/*
 * This file contains an experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */

#include "cdn_common.h"

template<typename L, typename D>
void runPCDMExperimentSparseANDTEST(L m, L n, ProblemData<int, double>& part,
		std::vector<D>& b, D & lambda, int tau, ofstream& logFile,
		std::vector<D> & Li, D& sigma, D maxTime,
		ProblemData<int, double>& partTest, int sampling) {

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
	D lonezeroloss;

	D Tprimal;
	D Tdual;
	D Tgap;
	D Tlonezeroloss;

	gap = computeDualityGapSparse(m, n, part, b, x, w, lambda, primal, dual,
			lonezeroloss);
	computeDualityGapSparse(m, n, partTest, b, x, w, lambda, Tprimal, Tdual,
			Tlonezeroloss);
	cout << "Duality Gap: " << gap << "   " << primal << "   " << dual << endl;

	logFile << setprecision(16) << "0," << sampling << "," << tau << "," << m
			<< "," << n << "," << lambda << "," << primal << "," << dual << ","
			<< lonezeroloss << "," << Tprimal << "," << Tdual << ","
			<< Tlonezeroloss << endl;

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

		if ((it + tau) % (n / sampling) < tau) {
			gap = computeDualityGapSparse(m, n, part, b, x, w, lambda, primal,
					dual, lonezeroloss);
			computeDualityGapSparse(m, n, partTest, b, x, w, lambda, Tprimal,
					Tdual, Tlonezeroloss);
			cout << "Duality Gap: " << gap << "   " << primal << "   " << dual
					<< endl;

			logFile << setprecision(16) << it + tau << "," << sampling << ","
					<< tau << "," << m << "," << n << "," << lambda << ","
					<< primal << "," << dual << "," << lonezeroloss << ","
					<< Tprimal << "," << Tdual << "," << Tlonezeroloss << endl;
		}

		if (elapsedTime > maxTime || gap < 0.000000000000001) {
			break;
		}
	}

}

int main(int argc, char * argv[]) {

	std::string file;
	std::string testfile;

	int RDN = 0;
	int sampling = 1;
	char c;
	while ((c = getopt(argc, argv, "A:R:T:S:")) != -1) {
		switch (c) {
		case 'A':
			file = optarg;
			break;
		case 'T':
			testfile = optarg;
			break;
		case 'R':
			RDN = atoi(optarg);
			break;
		case 'S':
			sampling = atoi(optarg);
			break;
		}
	}

	stringstream ss("");
	ss << file << "_mSDCA_log";

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
	ProblemData<int, double> partTest;
	std::vector<double> A;
	std::vector<double> b;

	loadDistributedSparseSVMRowData(file, -1, -1, part, false);
	loadDistributedSparseSVMRowData(testfile, -1, -1, partTest, false);
	m = part.m;
	n = part.n;
//		std::vector<double> &A = part.A_csr_values;
//		std::vector<double> &b = part.b;

	std::vector<double> Li(n, 0);
	std::vector<double> LiSqInv(n, 0);

	bool dense = (m * n == A.size());

	if (dense) {

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

	for (int PM = 0; PM < 20; PM++) {

		if (dense) {

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

	double lambda = 1 / (n + 0.0);

	double maxTime = 1000;

	std::vector<double> Hessian;

	int MAXTAU = n / 2;
	if (MAXTAU > 1024 * 8) {
		MAXTAU = 1024 * 8;
	}
//	MAXTAU = 2;
	for (int tau = 1; tau <= MAXTAU; tau = tau * 2) {

		double sigma = 1 + (tau - 1) * (maxEig - 1) / (n - 1.0);

		randomNumberUtil::init_random_seeds(rs);
		runPCDMExperimentSparseANDTEST(m, n, part, part.b, lambda, tau, logFile,
				Li, sigma, maxTime, partTest, sampling);

	}

	logFile.close();

//	histogramLogFile.close();
//	experimentLogFile.close();
	return 0;
}
