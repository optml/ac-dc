#include "../helpers/c_libs_headers.h"
#include "../utils/randomNumbersUtil.h"
#include "../problem_generator/distributed/generator_nesterov_to_file.h"

int main(int argc, char * argv[]) {
	const int MAXIMUM_THREADS = 64;
	std::vector<gsl_rng *> rs = randomNumberUtil::inittializeRandomSeeds(
			MAXIMUM_THREADS);
	//---------------------- Set output files

	std::string logFileName = "/tmp/test_problem";
//	std::string logFileName="/work/d11/d11-takac/taki/data/5T_128_problem";
	unsigned long long n = 50000000000;
//	n = 50000000000;
//	n = 10000;
	unsigned long long m = n / 100;
	unsigned long long p = 128;
	p = 10;
	n=1000;
//	p=800;
	m=n*2;
//	logFileName="/work/tmp/data";

	unsigned long long files = 192; //192;
	files = 128;
	files=4;
	double localBlocking = 0.995;

	nesterov_generator_to_file(logFileName, n, m, p, files, rs, localBlocking);

	return 0;
}
