//Use this program for converting a textual Track1 prediction file into a
//well formatted binary submission file.
//
//The program accepts two arguments:
//<prediction file> (input) and <submission file> (output)
//
//The input prediction file should contain 6005940 lines, corresponding
//to the 6005940 user-item pairs in the test set.
//Each line contains a predicted score (a real number between 0 and 100).
//The generated output file can be submitted to the KDD-Cup'11 evaluation
//system.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {

	if (argc<3) {
		fprintf(stderr, "Error: Must supply two command line arguments: <prediction file> (input) and <submission file> (output)\n");
		exit(1);
	}

	FILE * inFp = fopen(argv[1],"r");
	FILE * outFp = fopen(argv[2], "wb");

	const int ExpectedTestSize = 6005940;

	const int MaxLen=1<<12;
	char line[MaxLen];

	int lineNum = 0;
	double prediction;
	double sumPreds=0;
	while (fgets(line, MaxLen, inFp)) {
		lineNum += 1;
		sscanf(line,"%lf",&prediction);
		if (prediction<0 || prediction>100) {
			fprintf(stderr, "Error: out of bounds prediction at line %d\n", lineNum);
			exit(1);
		}
		unsigned char roundScore = (unsigned char)(2.55*prediction+0.5);
		fwrite(&roundScore,1,1,outFp);
		sumPreds += prediction;
	}
 
	if (lineNum!=ExpectedTestSize) {
		fprintf(stderr, "Error: expected %d predictions, but read %d ones\n",ExpectedTestSize,lineNum);
		exit(1);
	}
	fclose(inFp);
	fclose(outFp);

	fprintf(stderr, "**Completed successfully (mean prediction: %lf)**\n",sumPreds/ExpectedTestSize);

	return 0;
}
	

