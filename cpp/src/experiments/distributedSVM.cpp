#include "../helpers/c_libs_headers.h"
#include "../utils/randomNumbersUtil.h"
#include "../problem_generator/distributed/generator_nesterov_to_file.h"

int main(int argc, char * argv[]) {
	const int MAXIMUM_THREADS = 64;
	std::vector<gsl_rng *> rs = randomNumberUtil::inittializeRandomSeeds(
			MAXIMUM_THREADS);
	//---------------------- Set output files

//	std::string logFileName = "/work/tmp/svm";
	std::string logFileName="/work/d11/d11-takac/taki/SVM/large";
	unsigned long long n = 1000000000;
	n = 1000000000;
	unsigned long long m = n / 1000;
	unsigned long long p = 60;

	unsigned long long files = 192; //192;
	files = 16;

//	n = 10000;

	double val = pow(1/(p+0.0),0.5);

#pragma omp parallel for schedule(static, 1)
	for (int f = 0; f < files; f++) {

		unsigned long long samples = n / files;
		stringstream ss;
		ss << logFileName << "." << files << "." << f;

		ofstream of;
		of.open(ss.str().c_str(), ios::out | ios::binary);

		std::vector<unsigned long long> idxV(p);

		for (unsigned long long l = 0; l < samples; l++) {
			double lab = (double) (rand_r(&myseed) / (RAND_MAX + 1.0));
			if (lab > 0.5) {
				of << "1";
			} else {
				of << "-1";
			}

			for (int j = 0; j < p; j++) {
				bool notfinished = true;
				unsigned long long idx;
				while (notfinished) {
					notfinished = 0;
					idx = gsl_rng_uniform_int(rs[f], m);
					for (int k = 0; k < j; k++) {
						if (idxV[k] == idx) {
							notfinished = 1;
						}
					}

				}
				idxV[j] = idx;
				of << " " << idx << ":" << val;

			}

			of << endl;
		}

		of.close();

	}

	return 0;
}
