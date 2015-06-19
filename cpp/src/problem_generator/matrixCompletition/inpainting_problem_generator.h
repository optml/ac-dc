/*
 * inpainting_problem_generator.h
 *
 *  Created on: Feb 21, 2012
 *      Author: taki
 */

#ifndef INPAINTING_PROBLEM_GENERATOR_H_
#define INPAINTING_PROBLEM_GENERATOR_H_

#include "mc_problem_generation.h"
#include "../solver/structures.h"

template<typename T, typename I>
void generate_random_mc_data_from_image(problem_mc_data<I, T> &ProblemData_instance,std::vector<T> & imageInRowFormat, const char* filename, const char* filename_sample
		, T sparsity ) {
	// Create random sample with given sparsity
	I m = ProblemData_instance.m;
	I n = ProblemData_instance.n;

	generate_random_matrix_completion_problem(m, n, sparsity, ProblemData_instance.A_coo_values,
			ProblemData_instance.A_coo_row_idx, ProblemData_instance.A_coo_col_idx);

	cout << "START READING DATA FROM FILE\n";
	loadDataFromCSVFile(m, n, filename, imageInRowFormat);

	ProblemData_instance.A_coo_operator.resize(ProblemData_instance.A_coo_values.size());
	cout << "DATA LOADED TOTAL LENGTH " << ProblemData_instance.A_coo_values.size() << "\n";
	for (int i = 0; i < ProblemData_instance.A_coo_values.size(); i++) {

		int tmp = (float) 3 * rand() / RAND_MAX;
		if (tmp == 3)
			tmp--;

		T val = imageInRowFormat[ProblemData_instance.A_coo_row_idx[i] * n
				+ ProblemData_instance.A_coo_col_idx[i]];

		if (tmp == 0)
			ProblemData_instance.A_coo_operator[i] = 0;
		else if (tmp == 1) {
			val = val + (1 - val) / 2;
			ProblemData_instance.A_coo_operator[i] = 1;
		} else {
			val = val / 2;
			ProblemData_instance.A_coo_operator[i] = -1;
		}
		ProblemData_instance.A_coo_values[i] = val;
	}
	cout << "DATA SAMPLED\n";

	std::vector<float> initialSampledData(n * m, -1);
	for (int i = 0; i < ProblemData_instance.A_coo_row_idx.size(); i++) {
		initialSampledData[ProblemData_instance.A_coo_row_idx[i] * n + ProblemData_instance.A_coo_col_idx[i]]
				= ProblemData_instance.A_coo_values[i];
	}
	saveDataToCSVFile(m, n, filename_sample, initialSampledData);
}

#endif /* INPAINTING_PROBLEM_GENERATOR_H_ */
