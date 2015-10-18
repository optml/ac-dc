#ifndef READWHOLEDATA_H
#define READWHOLEDATA_H

template<typename D, typename L>
void readWholeData(string inputFile, ProblemData<L, D> & part, bool zeroBased) {

	int nclasses;
	int nsamples;
	int nfeatures;
	long long nonzero_elements_of_input_data;

	stringstream ss;
	ss << inputFile;
//	cout << "Going to read data" << endl;
	parse_LIB_SVM_data_get_size(ss.str().c_str(), nsamples, nfeatures, nonzero_elements_of_input_data);

//	cout << "Data file contains " << nfeatures << " features, "
//			<< nsamples << " samples " << "and total "
//			<< nonzero_elements_of_input_data << " nnz elements" << endl;

	FILE* filePtr = fopen(ss.str().c_str(), "r");
	if (filePtr == 0) {
		printf("File   '%s' not found\n", ss.str().c_str());
	}

	part.m = nfeatures;
	part.n = nsamples;

//	cout << "resize nsamples+1 " << nsamples + 1 << endl;
	part.A_csr_row_ptr.resize(nsamples + 1);
	part.A_csr_col_idx.resize(nonzero_elements_of_input_data);
//	cout << "resize nnz " << nonzero_elements_of_input_data << endl;
	part.A_csr_values.resize(nonzero_elements_of_input_data);
//	cout << "resize nnz " << nonzero_elements_of_input_data << endl;
	part.b.resize(nsamples);
//	cout << "resize nsamples " << nsamples << endl;
	L nnzPossition = 0;
	L processedSamples = -1;

	bool foundData = false;
	char* stringBuffer = (char*) malloc(65536);
	for (L i = 0; i < nsamples; i++) {

		char c;
		L pos = 0;
		char* bufferPointer = stringBuffer;

		do {
			c = fgetc(filePtr);

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

						processedSamples++;
						part.b[processedSamples] = ddval; // used for a1a data
						//part.b[processedSamples] = (-1.5 + ddval) * 2.0; // used for covtype data
						part.A_csr_row_ptr[processedSamples] = nnzPossition;

						pos++;
					}
				} else {
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);

					if (pos > 0) {

						if (!zeroBased)
							pos--;

						if (nnzPossition < nonzero_elements_of_input_data && foundData) {
							part.A_csr_col_idx[nnzPossition] = pos;
							part.A_csr_values[nnzPossition] = value;

							foundData = false;
							nnzPossition++;
						}

						pos = -1;
					}

				}
				bufferPointer = stringBuffer;
			} else if (c == ':') {
				//Position found
				*(bufferPointer) = 0;
				int value;
				sscanf(stringBuffer, "%i", &value);
				foundData = true;
				pos = value;
				bufferPointer = stringBuffer;
			} else {
				*(bufferPointer) = c;
				bufferPointer++;
			}

		} while (c != '\n');
	}

	processedSamples++;
	part.A_csr_row_ptr[processedSamples] = nnzPossition;
	free(stringBuffer);
	fclose(filePtr);
}

template<typename D, typename L>
int loadDistributedByFeaturesSVMRowData(string inputFile, int file, int totalFiles, ProblemData<L, D> & part,
		bool zeroBased) {

	int nclasses=2;
	int nsamples;
	int nfeatures;
	long long nonzero_elements_of_input_data;

	stringstream ss;
	ss << inputFile;
	if (totalFiles > 0) {
		ss << "_" << totalFiles << "_" << file;
	}
	cout << "Going to parse SVM data" << endl;
	parse_LIB_SVM_data_get_size(ss.str().c_str(), nsamples, nfeatures, nonzero_elements_of_input_data);
	cout << "Data file " << file << " contains " << nfeatures << " features, " << nsamples << " samples "
			<< "and total " << nonzero_elements_of_input_data << " nnz elements" << endl;

	FILE* filePtr = fopen(ss.str().c_str(), "r");
	if (filePtr == 0) {
		printf("File   '%s' not found\n", ss.str().c_str());
		return 0;
	}

	//cout << "Going to process data" << endl;

	part.m = nfeatures;
	part.n = nsamples;

	//cout << "resize nsamples+1 " << nsamples + 1 << endl;
	part.A_csr_row_ptr.resize(nsamples + 1);
	part.A_csr_col_idx.resize(nonzero_elements_of_input_data);
	//cout << "resize nnz " << nonzero_elements_of_input_data << endl;
	part.A_csr_values.resize(nonzero_elements_of_input_data);
	//cout << "resize nnz " << nonzero_elements_of_input_data << endl;
	part.b.resize(nsamples);
	//cout << "resize nsamples " << nsamples << endl;
	L nnzPossition = 0;
	L processedSamples = -1;

	bool foundData = false;
	char* stringBuffer = (char*) malloc(65536);
	for (L i = 0; i < nsamples; i++) {

		char c;
		L pos = 0;
		char* bufferPointer = stringBuffer;

		do {
			c = fgetc(filePtr);

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

						processedSamples++;
						part.b[processedSamples] = ddval; // used for a1a data
						if (ddval != -1 && ddval != 1)
							cout << part.b[processedSamples] << endl;
						//part.b[processedSamples] = (-1.5 + ddval) * 2.0; // used for covtype data
						part.A_csr_row_ptr[processedSamples] = nnzPossition;

						pos++;
					}
				} else {
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);

					if (pos > 0) {

						if (!zeroBased)
							pos--;

						if (nnzPossition < nonzero_elements_of_input_data && foundData) {
							part.A_csr_col_idx[nnzPossition] = pos;
							part.A_csr_values[nnzPossition] = value;

							foundData = false;
							nnzPossition++;
						}

						pos = -1;
					}

				}
				bufferPointer = stringBuffer;
			} else if (c == ':') {
				//Position found
				*(bufferPointer) = 0;
				int value;
				sscanf(stringBuffer, "%i", &value);
				foundData = true;
				pos = value;
				bufferPointer = stringBuffer;
			} else {
				*(bufferPointer) = c;
				bufferPointer++;
			}

		} while (c != '\n');
	}

	processedSamples++;
	part.A_csr_row_ptr[processedSamples] = nnzPossition;
	free(stringBuffer);
	fclose(filePtr);
	return 1;
}

template<typename D, typename L>
void partitionByFeature(ProblemData<L, D> & part, ProblemData<L, D> & newpart, int nPartition, int rank) {

	L nsamples = part.n;
	L nfeatures = part.m;

	L lowerRange;
	L upperRange;
	newpart.n = part.n;
	newpart.theta = part.theta;
	// part.m = floor(nfeatures / nPartition);
	// lowerRange = rank * part.m;
	// upperRange = (rank + 1) * part.m - 1;
	// if (rank == nPartition - 1){
	// 	part.m = nfeatures - (nPartition - 1) * floor(nfeatures / nPartition);
	// 	upperRange = nfeatures - 1;
	// }

	// for a1a data
	//lowerRange = 0; upperRange = nfeatures-1;
	// if (rank == 0)	{lowerRange = 0; upperRange = 20; }
	// if (rank == 1)	{lowerRange = 21; upperRange = 40;}
	// if (rank == 2)	{lowerRange = 41; upperRange = 65;}
	// if (rank == 3)	{lowerRange = 66; upperRange = nfeatures - 1;}	
	//for covtype data
	if (rank == 0) {
		lowerRange = 0;
		upperRange = 2;
	}
	if (rank == 1) {
		lowerRange = 3;
		upperRange = 5;
	}
	if (rank == 2) {
		lowerRange = 7;
		upperRange = 9;
	}
	if (rank == 3) {
		lowerRange = 10;
		upperRange = nfeatures - 1;
	}
	newpart.m = upperRange - lowerRange + 1;

	L nnElements = part.A_csr_col_idx.size(); //cout <<nnElements<<endl;
	newpart.A_csr_values.resize(nnElements);
	newpart.A_csr_col_idx.resize(nnElements);
	newpart.A_csr_row_ptr.resize(nnElements);
	newpart.b.resize(nnElements);

	L nnz = 0;
	L rowInd = 0;

	newpart.A_csr_row_ptr[0] = 0;
	for (L i = 0; i < nnElements; i++) {
		if (part.A_csr_col_idx[i] >= lowerRange && part.A_csr_col_idx[i] <= upperRange) {
			newpart.A_csr_values[nnz] = part.A_csr_values[i];
			newpart.A_csr_col_idx[nnz] = part.A_csr_col_idx[i] - lowerRange;
			newpart.b[nnz] = part.b[i];				//cout<<rank<<"   "<<part.A_csr_col_idx[i]<<endl;
			nnz++;

			if (part.A_csr_col_idx[i + 1] > upperRange || part.A_csr_col_idx[i + 1] < lowerRange) {
				newpart.A_csr_row_ptr[rowInd + 1] = nnz;
				rowInd++;
			}
		}
	}
	if (nPartition == 1) {
		rowInd = part.A_csr_row_ptr.size() - 1;
		for (L i = 0; i <= rowInd; i++)
			newpart.A_csr_row_ptr[i] = part.A_csr_row_ptr[i];
	}

	cout << rank << "   " << nnz << "   " << part.n << "   " << lowerRange << "   " << upperRange << "   " << newpart.m
			<< endl;

	newpart.A_csr_values.resize(nnz);
	newpart.A_csr_col_idx.resize(nnz);
	newpart.A_csr_row_ptr.resize(rowInd + 1);
	newpart.b.resize(nnz);

//	cout << "Data file contains features from " << lowerRange << " to "
//			<< upperRange << ", with "<< newpart.n <<" samples " << "and total "
//			<< nnz << " nnz elements" << endl;

}

template<typename D, typename L>
void readPartDataForPreCondi(string inputFile, ProblemData<L, D> & part, L Needed, bool zeroBased) {

	int nclasses;
	int nsamples;
	int nfeatures;
	long long nonzero_elements_of_input_data;

	stringstream ss;
	ss << inputFile;
	//ss << "_reorder";

//	cout << "Going to read data" << endl;
	parse_LIB_SVM_data_get_size(ss.str().c_str(), nsamples, nfeatures, nonzero_elements_of_input_data);

	//cout << "Data file contains " << nfeatures << " features, "
	//		<< nsamples << " samples " << "and total "
	//		<< nonzero_elements_of_input_data << " nnz elements" << endl;

	FILE* filePtr = fopen(ss.str().c_str(), "r");
	if (filePtr == 0) {
		printf("File   '%s' not found\n", ss.str().c_str());
	}
	nsamples = Needed;
	part.m = nfeatures;
	part.n = nsamples;

//	cout << "resize nsamples+1 " << nsamples + 1 << endl;
	part.A_csr_row_ptr.resize(nsamples + 1);
	part.A_csr_col_idx.resize(nonzero_elements_of_input_data);
//	cout << "resize nnz " << nonzero_elements_of_input_data << endl;
	part.A_csr_values.resize(nonzero_elements_of_input_data);
//	cout << "resize nnz " << nonzero_elements_of_input_data << endl;
	part.b.resize(nsamples);
//	cout << "resize nsamples " << nsamples << endl;
	L nnzPossition = 0;
	L processedSamples = -1;

	bool foundData = false;
	char* stringBuffer = (char*) malloc(65536);
	for (L i = 0; i < nsamples; i++) {

		char c;
		L pos = 0;
		char* bufferPointer = stringBuffer;

		do {
			c = fgetc(filePtr);

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

						processedSamples++;
						part.b[processedSamples] = ddval; // used for a1a data
						//part.b[processedSamples] = (-1.5 + ddval) * 2.0; // used for covtype data
						part.A_csr_row_ptr[processedSamples] = nnzPossition;

						pos++;
					}
				} else {
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);

					if (pos > 0) {

						if (!zeroBased)
							pos--;

						if (nnzPossition < nonzero_elements_of_input_data && foundData) {
							part.A_csr_col_idx[nnzPossition] = pos;
							part.A_csr_values[nnzPossition] = value;

							foundData = false;
							nnzPossition++;
						}

						pos = -1;
					}

				}
				bufferPointer = stringBuffer;
			} else if (c == ':') {
				//Position found
				*(bufferPointer) = 0;
				int value;
				sscanf(stringBuffer, "%i", &value);
				foundData = true;
				pos = value;
				bufferPointer = stringBuffer;
			} else {
				*(bufferPointer) = c;
				bufferPointer++;
			}

		} while (c != '\n');
	}

	processedSamples++;
	part.A_csr_row_ptr[processedSamples] = nnzPossition;
	free(stringBuffer);
	fclose(filePtr);
}
#endif
