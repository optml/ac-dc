#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "engine.h"
#define  BUFSIZE 256

#include <vector>

//int main(int argc, char *argv[]) {
//
//
//
//}

int main()

{
	Engine *ep;
	mxArray *T = NULL, *result = NULL;
	char buffer[BUFSIZE + 1];

	if (!(ep = engOpen("\0"))) {
		fprintf(stderr, "\nCan't start MATLAB engine\n");
		return EXIT_FAILURE;
	}

	int nnz = 10;

	int n = nnz;
	int d = nnz;

	std::vector<double> COL(nnz);
	std::vector<double> ROW(nnz);
	std::vector<double> VAL(nnz);
	for (int i = 0; i < nnz; i++) {
		COL[i] = i + 1;
		ROW[i] = i + 1;
		VAL[i] = i + 1;
	}

	mxArray *mRow = NULL, *mCol = NULL, *mVal = NULL, *mAlpha = NULL,
			*mW = NULL;
	mRow = mxCreateDoubleMatrix(1, nnz, mxREAL);
	memcpy((void *) mxGetPr(mRow), (void *) &COL[0], sizeof(double) * nnz);
	mCol = mxCreateDoubleMatrix(1, nnz, mxREAL);
	memcpy((void *) mxGetPr(mCol), (void *) &ROW[0], sizeof(double) * nnz);
	mVal = mxCreateDoubleMatrix(1, nnz, mxREAL);
	memcpy((void *) mxGetPr(mVal), (void *) &VAL[0], sizeof(double) * nnz);

	mAlpha = mxCreateDoubleMatrix(1, n, mxREAL);
	mW = mxCreateDoubleMatrix(1, d, mxREAL);

	engPutVariable(ep, "mRow", mRow);
	engPutVariable(ep, "mCol", mCol);
	engPutVariable(ep, "mVal", mVal);
	engPutVariable(ep, "mAlpha", mAlpha);
	engPutVariable(ep, "mW", mW);

	buffer[BUFSIZE] = '\0';
	engOutputBuffer(ep, buffer, BUFSIZE);


	engEvalString(ep, " A=sparse(mRow, mCol, mVal) ");






	printf("%s", buffer);



//
//	while (result == NULL) {
//	    char str[BUFSIZE+1];
//
//
//	    /*
//	     * Evaluate input with engEvalString
//	     */
//	    engEvalString(ep, str);
//
//	    /*
//	     * Echo the output from the command.
//	     */
//	    printf("%s", buffer);
//
//	    /*
//	     * Get result of computation
//	     */
//	    printf("\nRetrieving X...\n");
//	    if ((result = engGetVariable(ep,"X")) == NULL)
//	      printf("Oops! You didn't create a variable X.\n\n");
//	    else {
//		printf("X is class %s\t\n", mxGetClassName(result));
//	    }
//	}
//
//	/*
//	 * We're done! Free memory, close MATLAB engine and exit.
//	 */
//	printf("Done!\n");


	mxDestroyArray(result);
	engClose(ep);

	return EXIT_SUCCESS;
}

