#ifndef GENERATOR_NESTEROV_FILE_H_
#define GENERATOR_NESTEROV_FILE_H_
#include <vector>
//#ifndef _OPENMP
//#include "../solver/structures.h"
//#else
//#include "../distributed/distributed_structures.h"
//#endif
#include <iostream>
#include <iomanip>
#include "../../parallel/parallel_essentials.h"
#include "../../utils/distributed_instances_loader.h"
#include <exception>
using namespace std;

class MyTopKElements {
	int bufferSize;
	int currentBufferSize;
	double minimum;

public:
	std::vector<double> values;

	void setBufferSize(int _bufferSize) {
		bufferSize = _bufferSize;
		currentBufferSize = 0;
		values.resize(bufferSize);
	}

	void addElement(double val) {
		if (currentBufferSize < bufferSize) {
			values[currentBufferSize] = val;
			if (currentBufferSize == 0) {
				minimum = val;
			} else if (val < minimum) {
				minimum = val;
			}
			currentBufferSize++;
		} else {
			if (val >= minimum) {

				for (int j = 0; j < bufferSize; j++) {
					if (values[j] == minimum) {
						values[j] = val;
					}
					break;
				}
				minimum = values[0];
				for (int j = 1; j < bufferSize; j++) {
					if (values[j] < minimum) {
						minimum = values[j];
					}
					break;
				}

			} else {
				//ignore, we have already our representive maximas

			}

		}
	}

};

class st_sortingByAbsWithIndex {
public:
	double value;
	int idx;
};

class st_sortingByAbs {
public:
	double value;
};

template<typename D>
int struct_cmp_by_abs_value(const void *a, const void *b) {
	return (abs(*(D*) b) - abs(*(D*) a));
}

int struct_cmp_by_value(const void *a, const void *b) {
	st_sortingByAbsWithIndex *ia = (st_sortingByAbsWithIndex *) a;
	st_sortingByAbsWithIndex *ib = (st_sortingByAbsWithIndex *) b;
	double aa = ia->value;
	double bb = ib->value;
	if (aa * aa > bb * bb)
		return -1;
	else if (aa * aa < bb * bb)
		return 1;
	else
		return 0;

}

string getStreamToFile(const char* base, string extra, int i = 0) {
	stringstream ss;
	string finalFileName = base;
	ss << finalFileName;
	ss << "_" << extra;
	ss << "_" << i;
	finalFileName = ss.str();
	return finalFileName;
}

#include "../../solver/structures.h"
#include "../../utils/my_cblas_wrapper.h"

template<typename D, typename L>
double nesterov_generator_to_file(string logFileName, L n, L m, L p,
		L filesCount, std::vector<gsl_rng *> &rs, D localBlocking) {
	double lambda = 1;
	double rho = 1;
	L n_nonzero = 1000000;
	double sqrtofnonzeros = sqrt(0.0 + n_nonzero);
	cout << "Nonzeros " << n_nonzero << " " << sqrtofnonzeros << endl;
	if (n_nonzero > n / 2) {
		n_nonzero = n / 1000;
		sqrtofnonzeros = sqrt((double) n_nonzero);
	}

	randomNumberUtil::init_random_seeds(rs, 0);

	L local_n = n / filesCount;
	n = local_n * filesCount;
	cout << "going to allocate " << filesCount * n_nonzero << endl;
	std::vector < D > dataToSort(filesCount * n_nonzero);
	std::vector < MyTopKElements > myTopKElements(filesCount);
	for (int i = 0; i < filesCount; i++) {
		myTopKElements[i].setBufferSize(n_nonzero);
	}
	cout << "going to allocate " << m << endl;

	std::vector < D > b(m);
	cout << "done" << endl;
	D tmp = 0;
#pragma omp parallel for schedule(static,1) reduction(+ : tmp)
	for (L j = 0; j < m; j++) {
		b[j] = (D) (rand_r(&myseed) / (RAND_MAX + 1.0));
		b[j] = b[j] * 2 - 1;
		tmp += b[j] * b[j];
	}

#pragma omp parallel for schedule(static,1)
	for (L j = 0; j < m; j++)
		b[j] = b[j] / tmp;

	L local_m = (m * localBlocking) / filesCount;
	L global_m = m - local_m * filesCount;

	cout << "local and global " << local_m << " " << global_m << endl;

#pragma omp parallel for schedule(static,1)
	for (int file = 0; file < filesCount; file++) {

		L global_m_local_part = file * local_m;
		L global_m_global_part = filesCount * local_m;

		L nPtr = local_n * file;

		ofstream A_csc_local_row_id;
		ofstream A_csc_local_col_ptr;
		ofstream A_csc_local_vals;
		ofstream A_csc_global_row_id;
		ofstream A_csc_global_col_ptr;
		ofstream A_csc_global_vals;
		A_csc_local_row_id.open(
				getFileName(logFileName, "rowid", file, true).c_str(),
				ios::out | ios::binary);

		ofstream local_X;
		local_X.open(getFileName(logFileName, "x", file, true).c_str(),
				ios::out | ios::binary);

		A_csc_local_col_ptr.open(
				getFileName(logFileName, "colptr", file, true).c_str(),
				ios::out | ios::binary);
		A_csc_local_vals.open(
				getFileName(logFileName, "valuesTMP", file, true).c_str(),
				ios::out | ios::binary);
		A_csc_global_row_id.open(
				getFileName(logFileName, "rowid", file, false).c_str(),
				ios::out | ios::binary);
		A_csc_global_col_ptr.open(
				getFileName(logFileName, "colptr", file, false).c_str(),
				ios::out | ios::binary);
		A_csc_global_vals.open(
				getFileName(logFileName, "valuesTMP", file, false).c_str(),
				ios::out | ios::binary);

		L local_col_ptr = 0;
		L global_col_ptr = 0;
		A_csc_local_col_ptr << local_col_ptr << " ";
		A_csc_global_col_ptr << global_col_ptr << " ";
		L localNNZ = p * localBlocking;
		L globalNNZ = p - localNNZ;
		std::vector < L > local_IDX(localNNZ);
		std::vector < L > global_IDX(globalNNZ);
		for (L nl = 0; nl < local_n; nl++) { //local column indexes
			L globalN = nPtr + nl; // global column Index

			D xValue = 0;
//			dataToSort[globalN] = 0;

			for (L j = 0; j < localNNZ; j++) {
				L idx;
				int notfinished = 1;
				D val = (D) (rand_r(&myseed) / (RAND_MAX + 1.0));
				val = 2 * val - 1;
				while (notfinished) {
					notfinished = 0;
					idx = gsl_rng_uniform_int(gsl_rng_r, local_m);
					for (L k = 0; k < j; k++) {
						if (local_IDX[k] == idx) {
							notfinished = 1;
						}
					}
				}
//				A_csc_local_row_id << idx << " ";
				A_csc_local_row_id.write((char*) &idx, sizeof(L));
//		nesterov_generator_to_file		A_csc_local_vals << setprecision(32) << val << " ";
				A_csc_local_vals.write((char*) &val, sizeof(D));
				local_IDX[j] = idx;
//				parallel::atomic_add(inst.b[idx], 1.0);
				xValue += b[idx + global_m_local_part] * val;
				local_col_ptr++;
			}
			A_csc_local_col_ptr << local_col_ptr << " ";

			for (L j = 0; j < globalNNZ; j++) {
				L idx;
				int notfinished = 1;
				D val = (D) (rand_r(&myseed) / (RAND_MAX + 1.0));
				val = 2 * val - 1;
				while (notfinished) {
					notfinished = 0;
					idx = gsl_rng_uniform_int(gsl_rng_r, global_m);
					for (L k = 0; k < j; k++) {
						if (global_IDX[k] == idx) {
							notfinished = 1;
						}
					}
				}
//				A_csc_global_row_id << idx << " ";
				A_csc_global_row_id.write((char*) &idx, sizeof(L));
//				A_csc_global_vals << setprecision(32) << val << " ";
				A_csc_global_vals.write((char*) &val, sizeof(D));
				global_IDX[j] = idx;
				//				parallel::atomic_add(inst.b[idx], 1.0);

				xValue += b[idx + global_m_global_part] * val;

				myTopKElements[file].addElement(abs(xValue));
				local_X.write((char*) &xValue, sizeof(D));

				global_col_ptr++;
			}
			A_csc_global_col_ptr << global_col_ptr << " ";
		}

		A_csc_local_row_id.close();
		A_csc_local_col_ptr.close();
		A_csc_local_vals.close();
		A_csc_global_row_id.close();
		A_csc_global_col_ptr.close();
		A_csc_global_vals.close();
		local_X.close();

	}
	cout << "part done" << endl;
	for (int file = 0; file < filesCount; file++) {
		for (int i = 0; i < n_nonzero; i++) {
			dataToSort[file * n_nonzero + i] = myTopKElements[file].values[i];
		}
	}
	cout << "part 2 done" << endl;
	D optimalValue = 0;
	std::vector<D> x(0);
//	for (L i = 0; i < dataToSort.size(); i++) {
//		x[i] = dataToSort[i];
//		dataToSort[i] = abs(dataToSort[i]);
//	}
	cout << "SORTING START" << endl;
	std::sort(dataToSort.begin(), dataToSort.end(), greater<D>());
#include <exception>
	cout << "sorted" << endl;
	D treshHoldValue = dataToSort[n_nonzero];
	cout << dataToSort[0] << "   -  " << dataToSort[1] << endl;

#pragma omp parallel for schedule(static,1) reduction(+ : optimalValue )
	for (L i = 0; i < b.size(); i++) {
		optimalValue += b[i] * b[i];
	}
	optimalValue = optimalValue * 0.5;

	cout << " Optimal value is " << optimalValue << endl;

	cout << "going to process files " << endl;

//return 0.1;

	D sum_of_x = 0;
#pragma omp parallel for schedule(static,1) reduction(+ : sum_of_x )
	for (int file = 0; file < filesCount; file++) {
		L global_m_local_part = file * local_m;
		L global_m_global_part = filesCount * local_m;
		L nPtr = local_n * file;
		cout << "Processing file " << file << " " << global_m_local_part << " "
				<< global_m_global_part << endl;

		ifstream A_csc_local_col_ptr;
		ifstream A_csc_local_vals_INPUT;
		ifstream X_INPUT;
		ifstream A_csc_global_col_ptr;
		ifstream A_csc_global_vals_INPUT;
		ifstream A_csc_local_row_id;
		ifstream A_csc_global_row_id;
		ofstream A_csc_local_vals_OUTPUT;
		ofstream A_csc_global_vals_OUTPUT;
		ofstream X_OUTPUT;

		try {
			X_INPUT.open(getFileName(logFileName, "x", file, true).c_str(),
					ios::in | ios::binary);

			A_csc_local_row_id.open(
					getFileName(logFileName, "rowid", file, true).c_str(),
					ios::in | ios::binary);
			A_csc_global_row_id.open(
					getFileName(logFileName, "rowid", file, false).c_str(),
					ios::in | ios::binary);
			A_csc_local_col_ptr.open(
					getFileName(logFileName, "colptr", file, true).c_str(),
					ios::in | ios::binary);
			A_csc_local_vals_OUTPUT.open(
					getFileName(logFileName, "values", file, true).c_str(),
					ios::out | ios::binary);
			A_csc_global_col_ptr.open(
					getFileName(logFileName, "colptr", file, false).c_str(),
					ios::in | ios::binary);
			A_csc_local_vals_INPUT.open(
					getFileName(logFileName, "valuesTMP", file, true).c_str(),
					ios::in | ios::binary);

			A_csc_global_vals_INPUT.open(
					getFileName(logFileName, "valuesTMP", file, false).c_str(),
					ios::in | ios::binary);
			A_csc_global_vals_OUTPUT.open(
					getFileName(logFileName, "values", file, false).c_str(),
					ios::out | ios::binary);
			X_OUTPUT.open(getFileName(logFileName, "xx", file, true).c_str(),
					ios::out | ios::binary);

		} catch (exception& e) {
			cout << "PART1 " << file << " " << e.what() << endl;
		}

		L localPtr;
		L globalPtr;
		A_csc_local_col_ptr >> localPtr;
		A_csc_global_col_ptr >> globalPtr;
		for (L nl = 0; nl < local_n; nl++) { //local column indexes
			L globalN = nPtr + nl; // global column Index

			double alpha = 1;
			D oldVal;
			D newX = 0;

			try {
				X_INPUT.read((char*) &oldVal, sizeof(D));
//				oldVal = x[globalN];
				if (abs(oldVal) > treshHoldValue) {
					alpha = (double) abs(1 / oldVal);
					//			printf("alpha = %f \n", alpha);
					newX = ((D) (rand_r(&myseed) / (RAND_MAX + 1.0))) * rho
							/ (sqrtofnonzeros);
					if (oldVal < 0) {
						newX = -newX;
					}
				} else if (abs(oldVal) > 0.1) {
					alpha = (double) abs(1 / oldVal)
							* ((D) (rand_r(&myseed) / (RAND_MAX + 1.0)));
				}

				X_OUTPUT.write((char*) &newX, sizeof(D));
				if (newX > 0)
					sum_of_x += newX;
				else
					sum_of_x -= newX;

			} catch (exception& e) {
				cout << "Part2: " << file << " " << e.what() << endl;
			}

			L newLcalPtr;
			L newGlobalPtr;
			A_csc_local_col_ptr >> newLcalPtr;
			A_csc_global_col_ptr >> newGlobalPtr;
			D value;
			L IDX;
			for (int i = 0; i < newLcalPtr - localPtr; i++) {
				try {
					A_csc_local_row_id.read((char*) &IDX, sizeof(L));
					A_csc_local_vals_INPUT.read((char*) &value, sizeof(D));
					value = value * alpha;
					A_csc_local_vals_OUTPUT.write((char*) &value, sizeof(D));
					parallel::atomic_add(b[global_m_local_part + IDX],
							newX * value);
				} catch (exception& e) {
					cout << "Part3: " << file << " " << e.what() << endl;
				}

			}
			localPtr = newLcalPtr;
			for (int i = 0; i < newGlobalPtr - globalPtr; i++) {
				try {
					A_csc_global_row_id.read((char*) &IDX, sizeof(L));
//				A_csc_global_row_id >> IDX;
//				A_csc_global_vals_INPUT >> value;
					A_csc_global_vals_INPUT.read((char*) &value, sizeof(D));
					value = value * alpha;
//				A_csc_global_vals_OUTPUT << setprecision(32) << value << " ";
					A_csc_global_vals_OUTPUT.write((char*) &value, sizeof(D));
					parallel::atomic_add(b[global_m_global_part + IDX],
							newX * value);
				} catch (exception& e) {
					cout << "Part4: " << file << " " << e.what() << endl;
				}

			}
			globalPtr = newGlobalPtr;
		}

		try {

			A_csc_local_col_ptr.close();
			A_csc_local_vals_INPUT.close();
			A_csc_global_vals_INPUT.close();
			X_INPUT.close();
			ofstream A_csc_local_vals_INPUTOUT;
			ofstream A_csc_global_vals_INPUTOUT;

			A_csc_local_vals_INPUTOUT.open(
					getFileName(logFileName, "valuesTMP", file, true).c_str(),
					ios::out | ios::binary);

			A_csc_global_vals_INPUTOUT.open(
					getFileName(logFileName, "valuesTMP", file, false).c_str(),
					ios::out | ios::binary);

			A_csc_local_vals_INPUTOUT.close();
			A_csc_global_vals_INPUTOUT.close();

			A_csc_global_col_ptr.close();
			A_csc_local_vals_OUTPUT.close();
			A_csc_global_vals_OUTPUT.close();
			A_csc_local_row_id.close();
			A_csc_global_row_id.close();
		} catch (exception& e) {
			cout << "Part6: " << file << " " << e.what() << endl;
		}

		cout << "Finish file " << file << endl;
	}

	cout << "all files done " << endl;

//	D sum_of_x = 0;
//#pragma omp parallel for schedule(static,1) reduction(+ : sum_of_x )
//	for (L i = 0; i < x.size(); i++) {
//		if (x[i] > 0)
//			sum_of_x += x[i];
//		else
//			sum_of_x -= x[i];
//	}
	optimalValue += lambda * sum_of_x;
	printf("optval %1.16f   (|x|=%f)\n", optimalValue, sum_of_x);

	ofstream dataInfo;
	dataInfo.open(logFileName.c_str());
	dataInfo << local_n << " " << local_m << " " << global_m << " "
			<< setprecision(32) << optimalValue << endl;
	dataInfo.close();
	cout << "Data file done " << endl;
	for (int file = 0; file < filesCount; file++) {
		L global_m_local_part = file * local_m;
		L global_m_global_part = filesCount * local_m;
		L nPtr = local_n * file;

		ofstream b_local;
		ofstream b_global;

		b_local.open(getFileName(logFileName, "b", file, true).c_str());
		b_global.open(getFileName(logFileName, "b", file, false).c_str());

		for (L i = global_m_local_part; i < global_m_local_part + local_m;
				i++) {
			b_local << setprecision(32) << b[i] << " ";
		}
		for (L i = global_m_global_part; i < global_m_global_part + global_m;
				i++) {
			b_global << setprecision(32) << b[i] << " ";
		}

		b_local.close();
		b_global.close();

	}
	if (false) {

		ofstream b_total;
		b_total.open(getFileName(logFileName, "b", -1, false).c_str());
		b_total << "b = [ " << endl;
		for (L i = 0; i < b.size(); i++) {
			b_total << setprecision(32) << b[i] << endl;
		}
		b_total << "]; " << endl;
		b_total.close();

		ofstream A_ALL;
		A_ALL.open(getFileName(logFileName, "A_ALL", -1, false).c_str());
		A_ALL << " S = [" << endl;
		for (int file = 0; file < filesCount; file++) {

			L global_m_local_part = file * local_m;
			L global_m_global_part = filesCount * local_m;
			L nPtr = local_n * file;

			ifstream A_csc_local_col_ptr;
			ifstream A_csc_local_vals_INPUT;
			ifstream A_csc_global_col_ptr;
			ifstream A_csc_global_vals_INPUT;
			ifstream A_csc_local_row_id;
			ifstream A_csc_global_row_id;

			A_csc_local_row_id.open(
					getFileName(logFileName, "rowid", file, true).c_str());
			A_csc_global_row_id.open(
					getFileName(logFileName, "rowid", file, false).c_str());
			A_csc_local_col_ptr.open(
					getFileName(logFileName, "colptr", file, true).c_str());
			A_csc_local_vals_INPUT.open(
					getFileName(logFileName, "values", file, true).c_str());
			A_csc_global_col_ptr.open(
					getFileName(logFileName, "colptr", file, false).c_str());
			A_csc_global_vals_INPUT.open(
					getFileName(logFileName, "values", file, false).c_str());

			L localPtr;
			L globalPtr;
			A_csc_local_col_ptr >> localPtr;
			A_csc_global_col_ptr >> globalPtr;

			for (L nl = 0; nl < local_n; nl++) { //local column indexes
				L globalN = nPtr + nl; // global column Index

				L newLcalPtr;
				L newGlobalPtr;
				A_csc_local_col_ptr >> newLcalPtr;
				A_csc_global_col_ptr >> newGlobalPtr;
				D value;
				D IDX;
				for (int i = 0; i < newLcalPtr - localPtr; i++) {
					A_csc_local_row_id >> IDX;
					A_csc_local_vals_INPUT >> value;
					A_ALL << (IDX + global_m_local_part) << " " << globalN
							<< " " << setprecision(32) << value << endl;
				}
				localPtr = newLcalPtr;
				for (int i = 0; i < newGlobalPtr - globalPtr; i++) {
					A_csc_global_row_id >> IDX;
					A_csc_global_vals_INPUT >> value;
					A_ALL << (IDX + global_m_global_part) << " " << globalN
							<< " " << setprecision(32) << value << endl;
				}
				globalPtr = newGlobalPtr;
			}
			A_csc_local_col_ptr.close();
			A_csc_local_vals_INPUT.close();
			A_csc_global_col_ptr.close();
			A_csc_global_vals_INPUT.close();
			A_csc_local_row_id.close();
			A_csc_global_row_id.close();
		}
		A_ALL << " ];" << endl;
		A_ALL.close();
	}

	return optimalValue;
	/*
	 //DEBUG
	 inst.x.resize(n);
	 std::vector<D> residuals(m, 0);
	 for (int i = 0; i < n; i++)
	 inst.x[i] = x_optimal[i];
	 Losses<L, D, square_loss_traits>::bulkIterations(inst, residuals);
	 D sum_of_residuals = 0;
	 for (L i = 0; i < m; i++) {
	 sum_of_residuals += residuals[i] * residuals[i];
	 }
	 D objective = Losses<L, D, square_loss_traits>::compute_fast_objective(inst, residuals);
	 printf("At termination:\nObjective %f, \t |residuals|^2 %f\n", objective, sum_of_residuals);
	 for (int i = 0; i < n; i++)
	 inst.x[i] = 0;
	 */

}

#endif // GENERATOR_NESTEROV_FILE_H_
