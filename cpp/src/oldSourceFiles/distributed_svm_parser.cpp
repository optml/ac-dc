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

template<typename D, typename L>
int run(string inputFile, int files) {

	ofstream y_vector;
	y_vector.open(getFileName(inputFile, "y", -1, -1, true).c_str(),
			ios::out | ios::binary);
	ofstream y_vector2;
	y_vector2.open(getFileName(inputFile, "y_anscii", -1, -1, true).c_str());

	cout << "YYY" << endl;

	std::vector<ofstream*> dataTmp(files);

	cout << "XXX " << endl;

	for (int f = 0; f < files; f++) {
		dataTmp[f] = new ofstream();
		dataTmp[f]->open(
				getFileName(inputFile, "A_COO", f, files, true).c_str(),
				ios::out | ios::binary);
	}

	cout << "Create pointers for final datas " << endl;

	int nclasses;
	int nsamples;
	int nfeatures;
	long long nonzero_elements_of_input_data;
	parse_LIB_SVM_data_get_size(inputFile.c_str(), nsamples, nfeatures,
			nonzero_elements_of_input_data);
	cout << "Data file contains " << nfeatures << " features, " << nsamples
			<< " samples " << "and total " << nonzero_elements_of_input_data
			<< " nnz elements" << endl;
	int nPart = nfeatures / files;
	nPart++;

	FILE* file = fopen(inputFile.c_str(), "r");
	if (file == 0) {
		printf("File '%s' not found\n", inputFile.c_str());
		return 0;
	}

	cout <<"going to process data"<<endl;

	std::vector<L> nnzPerPart(files, 0);

	char* stringBuffer = (char*) malloc(65536);
	for (L i = 0; i < nsamples; i++) {
		char c;
		L pos = 0;
		char* bufferPointer = stringBuffer;
		do {
			c = fgetc(file);
//
			if ((c == ' ') || (c == '\n')) {
				if (pos == 0) {
					//Label found
					*(bufferPointer) = 0;
					int value;
					sscanf(stringBuffer, "%i", &value);

					D ddval = value;
					if (value < 100) {
						if (nclasses == 2 && value == 0) {
							ddval = (float) -1;
						} else {
						}

						y_vector.write((char*) &ddval, sizeof(D));
						y_vector2 << ddval << endl;
						pos++;
					}
				} else {
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);

					int part = pos / nPart;
					pos = pos - part * nPart;

					nnzPerPart[part]++;
					dataTmp[part]->write((char*) &pos, sizeof(L));
					dataTmp[part]->write((char*) &i, sizeof(L));
					D val = value;
					dataTmp[part]->write((char*) &val, sizeof(D));

					pos = 10;

//						h_data_values[nnz] = value;
//						h_data_row_index[nnz] = i + 1;
//						h_data_col_index[nnz] = pos;
//						colCount[pos - 1]++;
//					pos = 0;

//					printf("Feautre Found: X[%d , %d]=%f \n",
//							h_data_row_index[nnz], h_data_col_index[nnz],
//							h_data_values[nnz]);
//					nnz++;

				}
				bufferPointer = stringBuffer;
			} else if (c == ':') {
				//Position found
				*(bufferPointer) = 0;
				int value;
				sscanf(stringBuffer, "%i", &value);
				pos = value;
				bufferPointer = stringBuffer;
			} else {
				*(bufferPointer) = c;
				bufferPointer++;
			}

		} while (c != '\n');
	}
	free(stringBuffer);
	fclose(file);
	y_vector.close();
	y_vector2.close();
	for (int f = 0; f < files; f++) {
		dataTmp[f]->close();
	}

	ofstream problemDescription;
	problemDescription.open(
			getFileName(inputFile, "problem", -1, files, true).c_str());
	problemDescription << nsamples << " " << nfeatures << " ";
	for (int f = 0; f < files; f++) {
		ifstream CooData;
		CooData.open(getFileName(inputFile, "A_COO", f, files, true).c_str(),
				ios::in | ios::binary);
		std::vector<D> A_COO_Values(nnzPerPart[f]);
		std::vector<L> A_COO_RIDX(nnzPerPart[f]);
		std::vector<L> A_COO_CIDX(nnzPerPart[f]);
		L localN = 0;
		for (int s = 0; s < nnzPerPart[f]; s++) {
			L colIdx;
			L rowIdx;
			D val;
			CooData.read((char*) &colIdx, sizeof(L));
			CooData.read((char*) &rowIdx, sizeof(L));
			CooData.read((char*) &val, sizeof(D));
			if (colIdx > localN)
				localN = colIdx;
			A_COO_Values[s] = val;
			A_COO_RIDX[s] = rowIdx;
			A_COO_CIDX[s] = colIdx;
		}
		localN++;
		std::vector<D> Z_csc_val;
		std::vector<L> Z_csc_rowIdx;
		std::vector<L> Z_csc_ColPtr;
		getCSC_from_COO(A_COO_Values, A_COO_RIDX, A_COO_CIDX, Z_csc_val,
				Z_csc_rowIdx, Z_csc_ColPtr, nsamples, localN);
		problemDescription << localN << " " << nnzPerPart[f] << " ";

		ofstream A_csc_vals;
		ofstream A_csc_row_idx;
		ofstream A_csc_col_ptr;
		A_csc_row_idx.open(
				getFileName(inputFile, "rowid", f, files, true).c_str(),
				ios::out | ios::binary);
		A_csc_col_ptr.open(
				getFileName(inputFile, "colptr", f, files, true).c_str(),
				ios::out | ios::binary);
		A_csc_vals.open(
				getFileName(inputFile, "values", f, files, true).c_str(),
				ios::out | ios::binary);

		for (int i = 0; i < Z_csc_val.size(); i++) {
			A_csc_vals.write((char*) &Z_csc_val[i], sizeof(D));
		}
		for (int i = 0; i < Z_csc_rowIdx.size(); i++)
			A_csc_row_idx.write((char*) &Z_csc_rowIdx[i], sizeof(L));
		for (int i = 0; i < Z_csc_ColPtr.size(); i++) {
			A_csc_col_ptr.write((char*) &Z_csc_ColPtr[i], sizeof(L));
		}
		A_csc_vals.close();
		A_csc_row_idx.close();
		A_csc_col_ptr.close();

	}
	problemDescription.close();

	cout << "SVM Parser finished" << endl;
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
