//ulimit -s unlimited
//nvcc -lcusparse test.cu
//nvcc -lcublas  -lcusparse -arch sm_20  test.cu && ./a.out
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sstream>

#include "ttd.h"


#define TOTALTHREDSPERBLOCK 256
#define COLUMNLENGTH 4
#define NMAXITERKernel 100
#define ROWSCALE 2

using namespace std;

void oneDayPR(int row, int col, int exampleType, float executionTime,
		float lambda) {

	int dim1 = 14;
	int dim2 = 1;
	float PRCDMExecutionTime = 3600;
	float SRCDMExecutionTime = 0;

	int m, n;
	clock_t t1;
	float * A_TTD;
	int * Row_IDX;
	int * Col_IDX;
	int nnzElements;
	int* nodeDescription;
	//-------------------------------GET BOUNDARY POINTS
	int * boundary;
	int boundarySize = 0;
	getBoundaryVector(row, col, exampleType, &boundary, &boundarySize);
	cout << "Boundary size is " << boundarySize << "\n";
	//-------------------------------GENERATE PROBLEMS
	t1 = clock();
	generateTTDProblem(row, col, exampleType, &m, &n, &A_TTD, &Col_IDX,
			&Row_IDX, &nnzElements, boundary, boundarySize, &nodeDescription,
			ROWSCALE);

	for (int i=0;i<n;i++)
	{
		printf("Bars  [%d,%d] [%d,%d]\n",nodeDescription[i*4],nodeDescription[i*4+1],nodeDescription[i*4+2],nodeDescription[i*4+3]);
	}

	cout << "Dimension of your problem is " << m << " x  " << n << "\n";
	printf("Number of NNZ: %d\n", nnzElements);

}


int main(void) {

	//	timeComparison(50, 50, 7, 10000, 0.0001;);
	//	timeComparison(60, 60, 7, 1, 0.0001);


	oneDayPR(3, 7, 2, 1, 0.000001);

	//	niceRCDM(200, 200, 2, 80000, 0.0001);
//	niceAC(200, 200, 2, 80000, 0.0001);
	//	niceProblem(200, 200, 2, 100000, 0.0001);
	//	greedyTTD();
	//	calculateTTDProblem();
	return 1;
}
