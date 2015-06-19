#ifndef GENERATOR_H_
#define GENERATOR_H_


#include "generator_blocked.h"
#include "generator_kdiagonal.h"
// #include "problem_generator/generator_nesterov.h"
#include "generator_random.h"

enum GeneratorType {
	Random = 0, RandomColumnNNZ, Blocked, kDiagonal, kDiagonalPlusMDense,
};

class generator_data {
public:
	GeneratorType type;
	int m; int n; int k;
	generator_data(int type_id, int cols, int rows,  int param)
	: type((GeneratorType)type_id), n(cols), m(rows), k(param) {};
	bool check() {
		if (type < 0 || type > kDiagonalPlusMDense) return false;
		if (n < 1) return false;
		if (m < 1) return false;
		if (k < 1) return false;
		return true;
	}
};

template<typename T>
T& operator<<(T& stream, generator_data& gen) {
	switch (gen.type) {
	case Random:
		stream << "Random-n" << gen.n << "m" << gen.m << "p" << gen.k;
		break;
	case RandomColumnNNZ:
		stream << "RandomColumnNNZ-n" << gen.n << "m" << gen.m << "p" << gen.k;
		break;
	case Blocked:
		stream << "Blocked"  << gen.n << "m" << gen.m << "k" << gen.k;
		break;
	case kDiagonal:
		stream << "kDiagonal-n" << gen.n << "k" << gen.k;
		break;
	case kDiagonalPlusMDense:
		stream << "kkDiagonalPlusDense-n" << gen.n << "p" << gen.m << "k" << gen.k;
		break;
	// case Nesterov: stream << "Nesterov-n" << gen.n << "m" << gen.m << "p" << gen.k; break;
	}
	return stream;
}

template<typename L, typename D>
void generate_instance(ProblemData<L, D> &inst, generator_data& gen) {
  int temp1, temp2;

	switch (gen.type) {
	case Random:
		generate_random_problem(inst, gen.n, gen.m, gen.k);
		break;

	case RandomColumnNNZ:
		// FIXME: Add
		break;

	case Blocked:
		generate_block_problem(inst, gen.n, gen.m, gen.k, 0.2);
		break;

	case kDiagonal:
		generate_k_diagonal(inst, temp1, temp2, gen.n, gen.k);
		break;

	case kDiagonalPlusMDense:
		generate_k_diagonal_with_few_full_columns(inst, temp1, temp2, gen.n, gen.k, gen.m);
		break;

	//case Nesterov: //nesterov_generator(inst, gen.n, gen.m, gen.k);	break;
	}
}

#endif // GENERATOR_H_
