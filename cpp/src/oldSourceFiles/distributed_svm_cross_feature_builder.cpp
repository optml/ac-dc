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
#include "../svm/svm_parser.h"
#include "../helpers/matrix_conversions.h"
using namespace std;

#include "../problem_generator/distributed/generator_nesterov_to_file.h"
#include "../utils/distributed_instances_loader.h"

template<typename D, typename L>
void extendByRandomCrossFeatures(ProblemData<L, D>& part, long maxnnz) {
	if (part.A_csc_row_idx.size() < maxnnz) {
		cout << "GOING TO EXTEND DATA" << endl;

		L pointA = 0;
		L pointB = 1;
		while (part.A_csc_row_idx.size() < maxnnz) {

			bool initialized = false;
			bool doneFirst = false;
			bool doneSecond = false;
			L positionFirst = part.A_csc_col_ptr[pointA];
			L positionSecond = part.A_csc_col_ptr[pointB];
			L maxPositionFirst = part.A_csc_col_ptr[pointA + 1];
			L maxPositionSecond = part.A_csc_col_ptr[pointB + 1];
			L colFirst;
			L colSecond;

			while (!(doneFirst && doneSecond)) {
				if (!doneFirst) {
					colFirst = part.A_csc_row_idx[positionFirst];
				} else {
					colFirst = -1;
				}
				if (!doneSecond) {
					colSecond = part.A_csc_row_idx[positionSecond];
				} else {
					colSecond = -2;
				}

				//			cout << colFirst <<" "<<colSecond <<" "<<positionFirst<<"/"<<maxPositionFirst<<"  "<<positionSecond<<
				//					"/"<<maxPositionSecond<<" "
				//					;

				if ((!doneFirst && !doneSecond) && colFirst == colSecond) {

					part.A_csc_values.push_back(1);
					part.A_csc_row_idx.push_back(positionFirst);
					initialized = true;

//					F tmp = this->tf_idf[colFirst]
//							* A_A_csr_values[positionFirst] / l1NormQuery
//
//					* B_A_csr_values[positionSecond] / l1NormHistogram;
//					score += tmp;

					positionFirst++;
					positionSecond++;

				} else if (((!doneFirst && !doneSecond)
						&& (colFirst < colSecond))
						|| (!doneFirst && doneSecond)) {

					positionFirst++;
					part.A_csc_values.push_back(1);
					part.A_csc_row_idx.push_back(positionFirst);
					initialized = true;

				} else if (((!doneFirst && !doneSecond)
						&& (colFirst > colSecond))
						|| (doneFirst && !doneSecond)) {

					positionSecond++;
					part.A_csc_values.push_back(1);
					part.A_csc_row_idx.push_back(positionSecond);
					initialized = true;
				} else {
					cout << "ERROR!!!!" << colFirst << " " << colSecond << " "
							<< doneFirst << " " << doneSecond << endl;
					positionFirst++;
					positionSecond++;

				}

				if (positionFirst >= maxPositionFirst) {
					doneFirst = true;
				}

				if (positionSecond >= maxPositionSecond) {
					doneSecond = true;
				}

			}

			if (initialized) {
				if (part.n % 100000 == 0) {
					cout << part.n << " " << part.A_csc_values.size() << " / "
							<< maxnnz << endl;
				}
				part.n++;
				part.A_csc_col_ptr.push_back(part.A_csc_values.size());
			} else {
			}

			if (pointB == part.n - 1) {
				pointA++;
				pointB = pointA + 1;
			}
			if (pointA == part.n - 2) {
				cout << "XXXXXXX" << endl;
				break;
			}
		}
	}
}

template<typename D, typename L>
int run(string inputFile, int files) {
	cout << "XXXXXX" << endl;
	// load problem data description
	ifstream problemDescription;
	problemDescription.open(
			getFileName(inputFile, "problem", -1, files, true).c_str());
	cout << "YYYYYY" << endl;
	L nsamples;
	L nfeatures;

	problemDescription >> nsamples;
	problemDescription >> nfeatures;

	std::vector<L> localN(files);
	std::vector<L> localNNZ(files);
	for (int f = 0; f < files; f++) {
		L tmpN;
		L tmpNNZ;
		problemDescription >> tmpN;
		problemDescription >> tmpNNZ;
		localN[f] = tmpN;
		localNNZ[f] = tmpNNZ;
	}
	problemDescription.close();
	long maxnnz = 268435456;

	cout << "sizes loaded " << endl;

	for (int file = 0; file < 1; file++) {
		if (localNNZ[file] < maxnnz) {
			ProblemData<L, D> part;
			loadDistributedSVMData(inputFile, file, files, part);
			extendByRandomCrossFeatures(part, maxnnz);
			storeCOOMatrixData(inputFile, file, files, part);

			nfeatures+=part.n-localN[file];
			localN[file]=part.n;
			localNNZ[file]=part.A_csc_values.size();
		}
	}
	ofstream problemDescriptionOut;
	problemDescriptionOut.open(
			getFileName(inputFile, "problem", -1, files, true).c_str());
	problemDescriptionOut << nsamples << " ";
	problemDescriptionOut << nfeatures << " ";
	for (int f = 0; f < files; f++) {
		problemDescriptionOut << localN[f] << " ";
		problemDescriptionOut << localNNZ[f] << " ";
	}
	problemDescriptionOut.close();
}

int main(int argc, char * argv[]) {
//	run<double, int>("./data/a1a", 4);

	run<double, int>("/home/d11/d11/takacn/work/data/wikipedia.svn", 16);
//	run<double, int>("/home/e248/e248/takac/work/data/kdda.t", 4);
//
//	run<double, int>("/home/e248/e248/takac/work/data/kddb", 4);
//	run<double, int>("/home/e248/e248/takac/work/data/kddb.t", 4);
//
//	run<double, int>(
//			"/home/e248/e248/takac/work/data/webspam_wc_normalized_unigram.svm",
//			4);
//	run<double, int>(
//			"/home/e248/e248/takac/work/data/webspam_wc_normalized_trigram.svm",
//			4);

}
