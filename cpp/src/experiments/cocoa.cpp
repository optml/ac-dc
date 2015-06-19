/*
 * This file contains an experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */

#include "../helpers/c_libs_headers.h"
#include "../utils/randomNumbersUtil.h"

#include <gsl/gsl_cblas.h>
#include "../utils/my_cblas_wrapper.h"
#include "../helpers/utils.h"

#include "../utils/distributed_instances_loader.h"

#include <cmath>

int main(int argc, char * argv[]) {

	std::string file;

	int K = 0;

	char c;
	while ((c = getopt(argc, argv, "A:K:")) != -1) {
		switch (c) {
		case 'A':
			file = optarg;
			break;
		case 'K':
			K = atoi(optarg);
			break;
		}
	}

	stringstream ss("");
	ss << file << "_log";

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

	loadDistributedSparseSVMRowData(file, -1, -1, part, false);
	m = part.m;
	n = part.n;

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

	for (int kk = 0; kk < 6; kk++)
	{

	K = pow(2, kk+8);


	double sigma = 0;

	for (int k = 0; k < K; k++) {

		int nK = part.n / K;
		int fi = nK * (k);

		std::vector<double> x(n);
			std::vector<double> y(m);

			for (int i = 0; i < n; i++) {
				x[i] = rand() / (0.0 + RAND_MAX);
			}
			double norm = cblas_l2_norm(n, &x[0], 1);
			cblas_vector_scale(n, &x[0], 1 / norm);

			double maxEig = 1;




		if (k == K - 1) {
			nK += part.n - nK * K;
		}
		//cout << "Partiiton " <<  k <<"  "<<fi << "  "<< nK <<endl;
		for (int PM = 0; PM < 6; PM++) {

			for (int j = 0; j < m; j++) {
				y[j] = 0;
			}
			for (int i = fi; i < fi + nK; i++) {
				for (int j = part.A_csr_row_ptr[i];
						j < part.A_csr_row_ptr[i + 1]; j++) {
					y[part.A_csr_col_idx[j]] += part.A_csr_values[j]
							* LiSqInv[i] * x[i];
				}

			}

			for (int i = fi; i < fi + nK; i++) {
				x[i] = 0;

				for (int j = part.A_csr_row_ptr[i];
						j < part.A_csr_row_ptr[i + 1]; j++) {

					x[i] += part.A_csr_values[j] * LiSqInv[i]
							* y[part.A_csr_col_idx[j]];

				}

			}

			maxEig = cblas_l2_norm(n, &x[0], 1);
			//cout << maxEig << endl;
			cblas_vector_scale(n, &x[0], 1 / maxEig);

		}

		double sigmaK = maxEig;
		sigma += sigmaK * nK;
	}

	cout << "sigma is " << sigma << "n is "<< part.n << "  after norm "
			<< 1.0* sigma / part.n / part.n << endl;
	;

	cout << "Max eigenvalue estimated " << endl;

	}
	logFile.close();

//	histogramLogFile.close();
//	experimentLogFile.close();
	return 0;
}
