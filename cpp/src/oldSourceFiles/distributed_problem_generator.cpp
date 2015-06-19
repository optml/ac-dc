/*
 * generate_distributed_problem.cpp
 *
 *  Created on: Jan 17, 2013
 *      Author: taki
 */

#include "../helpers/c_libs_headers.h"
#include "../helpers/utils.h"
#include "../helpers/gsl_random_helper.h"

#include "../problem_generator/distributed/generator_nesterov_to_file.h"
int main(int argc, char * argv[]) {
	cout << "Generation starts" << endl;

	gsl_rng_env_setup();
	const gsl_rng_type * T;
	gsl_rng * rr;
	T = gsl_rng_default;
	rr = gsl_rng_alloc(T);
	const int MAXIMUM_THREADS = 32;
	std::vector<gsl_rng *> rs(MAXIMUM_THREADS);
	for (int i = 0; i < MAXIMUM_THREADS; i++) {
		rs[i] = gsl_rng_alloc(T);
		unsigned long seed = i + MAXIMUM_THREADS;
		gsl_rng_set(rs[i], seed);
	}

//	nesterov_generator_to_file("/tmp/experiment",
//			1000, 2000, 40,
//			4,rs,0.9);

	long n = 10000;
	long m = n * 2;
	long p = 10;
	double fraction = 0.5;
	long nodes = 4;

	fraction = 0.9;

	nesterov_generator_to_file("/tmp/experimentNEW", n, m, p,
			nodes, rs, fraction);
//
//	n = n * 100000;
//	m = n * 2;
//	nodes = 4 * 8;
//	nesterov_generator_to_file("/home/e248/e248/takac/work/experiment8b", n, m,
//			p, nodes, rs, fraction);

//	n = 500000000;
//	m = n * 2;
//	nodes = 512;
//	p = 350;
//	fraction = 0.9995;
//	cout << "Problem size is " << m << " x " << n << endl;
//	nesterov_generator_to_file("/home/d11/d11/takacn/work/data/experiment512", n,
//			m, p, nodes, rs, fraction);

	cout << "Generation finished" << endl;
}

