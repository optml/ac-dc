//ulimit -s unlimited
//nvcc -lcusparse test.cu

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


#define MaxTTD_NNZ_ELEMENTS  3000
#define TOTALTHREDSPERBLOCK 256
#define COLUMNLENGTH 4
#define NMAXITERKernel 1000
#define ROWSCALE 1

using namespace std;
void generateTTDProblem(int row, int col, int exampleType, long* mOut,
		long* nOut, float** A, int** ColIDX, int ** RowIDX, long * nnzElements,
		int* boundary, int boundarySize, int** nodeDescription, float rowScale);
void getBoundaryVector(int row, int col, int exampleType, int** boundary,
		int* boudarySize);
void getForceVector(float ** d, int m, int r, int c, int exampleType);
int GCD(int a, int b);

void checkCUDAError(const char *msg);
void scaleBetta(int* betta, float* beta, float scale);
int minimabs(int * betta);
int maximabs(int * betta);
void print_time_message(clock_t* t1, char message[200]) {
	clock_t t2 = clock();
	double diff = ((float) t2 - (float) (*t1)) / 1000000.0F;
	printf("%s: %f sec.\n", message, diff);
	*t1 = clock();
}
double getElapsetTime(clock_t* t1) {
	clock_t t2 = clock();
	double diff = ((float) t2 - (float) (*t1)) / 1000000.0F;
	*t1 = clock();
	return diff;
}



void greedyTTD() {
	int col, row, exampleType;
	cout << "Enter number of columns: ";
//	col = 85;
				cin >> col;
	cout << "Enter number of rows: ";
	row = 85;
				cin >> row;
	//	cout << "Enter example type: ";
	//	cin >> exampleType;
	exampleType = 5;
	long m, n;
	clock_t t1;//, t2;
	float * A_TTD;
	int * Row_IDX;
	int * Col_IDX;
	long nnzElements;
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
	print_time_message(&t1, "Getting problem dimension");
	cout << "Dimension of your problem is " << m << " x  " << n << "\n";
	printf("Number of NNZ: %d\n", nnzElements);
	//-------------------------------GET FORCE VECTORS

}

int main(void) {

	greedyTTD();
	//	calculateTTDProblem();
	return 1;
}

void generateTTDProblem(int row, int col, int exampleType, long* mOut,
		long* nOut, float** A, int** ColIDX, int ** RowIDX, long * nnzElements,
		int* boundary, int boundarySize, int** nodeDescription, float rowScale) {
	float kappa = 1;
	float scale = sqrt(kappa);
	long m = row * col;
	float A_values[MaxTTD_NNZ_ELEMENTS];
	int Rows_A[MaxTTD_NNZ_ELEMENTS];
	int Cols_A[MaxTTD_NNZ_ELEMENTS];
	*nodeDescription = new int[MaxTTD_NNZ_ELEMENTS*4];
	int nodeDescriptionId = 0;
	long nnz = 0;
	long node = 0;
	int tt = col;
	if (row > tt)
		tt = row;
	int GSDS[tt + 1][tt + 1];
	for (int i = 1; i <= tt; i++) {
		for (int j = i; j <= tt; j++) {
			GSDS[i][j] = GCD(i, j);
			GSDS[j][i] = GCD(i, j);
		}
	}
	int betta[2];
	float beta[2];
	for (int j = 1; j <= col; j++) {
		for (int i = 1; i <= col; i++) {
			for (int k = 1; k <= row; k++) {
				for (int l = 1; l <= col - j; l++) {
					betta[0] = rowScale * l;// for node (i,j) we add bars to all (k,j+i)
					betta[1] = -k + i;
					scaleBetta(betta, beta, scale);
					betta[0] = l;
					if (col == j)
						continue;

					if (l > 1) {
						int skip = 0;
						int ta = minimabs(betta);
						int tb = maximabs(betta);
						if (ta == 0) {
							skip = 1;
						} else {
							if (GSDS[ta][tb] > 1) {
								skip = 1;
							}
						}
						if (skip)
							continue;
					}
					int totalMatchA = 0;
					int totalMatchB = 0;
					for (int bi = 0; bi < boundarySize; bi++) {
						if (boundary[bi] == row * (j - 1) + i)
							totalMatchA++;
						if (boundary[bi] == row * (j) + k + (l - 1) * row)
							totalMatchB++;
					}
					if (totalMatchA + totalMatchB < 2) {
						node++;
						int tmp = row * (j - 1) + i;
//						A_values[nnz] = -(1 - totalMatchA) * beta[0];
//						Rows_A[nnz] = tmp;
//						Cols_A[nnz] = node;
						nnz++;
//						A_values[nnz] = -(1 - totalMatchA) * beta[1];
//						Rows_A[nnz] = tmp + m;
//						Cols_A[nnz] = node;
						nnz++;
						tmp = row * (j) + k + (l - 1) * row;
//						A_values[nnz] = (1 - totalMatchB) * beta[0];
//						Rows_A[nnz] = tmp;
//						Cols_A[nnz] = node;
						nnz++;
//						A_values[nnz] = (1 - totalMatchB) * beta[1];
//						Rows_A[nnz] = tmp + m;
//						Cols_A[nnz] = node;
						nnz++;

//						(*nodeDescription)[nodeDescriptionId] = i;
//						(*nodeDescription)[nodeDescriptionId + 1] = j;
//						(*nodeDescription)[nodeDescriptionId + 2] = k;
//						(*nodeDescription)[nodeDescriptionId + 3] = l + j;
//						nodeDescriptionId += 4;
					}
				}
			}
			if (i < row) {
				int tmp = i + (j - 1) * row;
				int totalMatchA = 0;
				int totalMatchB = 0;
				for (int bi = 0; bi < boundarySize; bi++) {
					if (boundary[bi] == tmp)
						totalMatchA++;
					if (boundary[bi] == tmp + 1)
						totalMatchB++;
				}
				if (totalMatchA + totalMatchB < 2) {
					node = node + 1;
//					A_values[nnz] = -(1 - totalMatchA);
//					Rows_A[nnz] = tmp + m;
//					Cols_A[nnz] = node;
					nnz++;
//					A_values[nnz] = 0; //fake node
//					Rows_A[nnz] = tmp + 1;
//					Cols_A[nnz] = node;
					nnz++;
//					A_values[nnz] = 0; //fake node
//					Rows_A[nnz] = tmp + 2;
//					Cols_A[nnz] = node;
					nnz++;

//					A_values[nnz] = (1 - totalMatchB);
//					Rows_A[nnz] = tmp + m + 1;
//					Cols_A[nnz] = node;
					nnz++;
//					(*nodeDescription)[nodeDescriptionId] = i;
//					(*nodeDescription)[nodeDescriptionId + 1] = j;
//					(*nodeDescription)[nodeDescriptionId + 2] = i + 1;
//					(*nodeDescription)[nodeDescriptionId + 3] = j;
//					nodeDescriptionId += 4;
				}
			}
		}
	}

//	*A = new float[nnz];
//	*ColIDX = new int[nnz];
//	*RowIDX = new int[nnz];
//	for (int i = 0; i < nnz; i++) {
//		(*A)[i] = A_values[i];
//		(*ColIDX)[i] = Cols_A[i];
//		(*RowIDX)[i] = Rows_A[i];
//	}
	*nOut = node;
	*mOut = row * col * 2;
	*nnzElements = nnz;
}

void getBoundaryVector(int row, int col, int exampleType, int** boundary,
		int* boudarySize) {

	switch (exampleType) {
	case 1:
		*boundary = new int[row];
		for (int i = 0; i < row; i++) {
			(*boundary)[i] = i + 1;
		}
		//		boundaryIDX(1, i) = 1;
		//		boundaryIDX(2, i) = i;
		*boudarySize = row;
		break;
	case 2:
		*boundary = new int[4];
		(*boundary)[0] = row;
		(*boundary)[1] = row * col;
		(*boundary)[2] = row * col - row * (col / 3);
		(*boundary)[3] = row + row * (col / 3);
		*boudarySize = 4;
		break;
	case 3:
		*boundary = new int[4];
		(*boundary)[0] = row-5;
		(*boundary)[1] = row * col-5;
		(*boundary)[2] = row * col - row * (col / 3);
		(*boundary)[3] = row + row * (col / 3);
		*boudarySize = 4;
		break;


	case 5:
		*boundary = new int[0];
		*boudarySize = 0;
		break;

	default:
		break;
	}
}
//		if (boundarytype==3) %bridge
//
//		boundary(1,1)=r;
//		boundary(1, 4) = r + r * (floor(c / 3));
//		boundary(1, 3) = r * c - r * floor(c / 3);
//		boundary(1, 2) = r * c;
//
//		boundaryIDX(1, 1) = 1;
//		boundaryIDX(2, 1) = r;
//		boundaryIDX(1, 2) = c;
//		boundaryIDX(2, 2) = r;
//
//		boundaryIDX(1, 3) = floor(c / 3) + 1;
//		boundaryIDX(2, 3) = r;
//		boundaryIDX(1, 4) = c - floor(c / 3);
//		boundaryIDX(2, 4) = r;
//
//		end
//		if (boundarytype==4) %
//
//		boundary(1,1)=r;
//		% boundary(1,4)=r+r*(floor(c/5));
//		% boundary(1,3)=r*c-r*floor(c/5);
//		boundary(1, 2) = r * c;
//
//		boundaryIDX(1, 1) = 1;
//		boundaryIDX(2, 1) = r;
//		boundaryIDX(1, 2) = c;
//		boundaryIDX(2, 2) = r;
//
//		% boundaryIDX(1,3)=floor(c/5)+1;
//		% boundaryIDX(2,3)=r;
//		% boundaryIDX(1,4)=c-floor(c/5);
//		% boundaryIDX(2,4)=r;
//
//		end
//
//		if (boundarytype==5) %
//
//		end
//
//		if (boundarytype==6) %
//
//		boundary(1,1)=r;
//
//		boundary(1, 4) = r + r * (floor(c / 5));
//		boundary(1, 3) = r * c - r * floor(c / 5);
//		boundary(1, 5) = r + r * 2 * (floor(c / 5));
//		boundary(1, 6) = r * c - r * 2 * floor(c / 5);
//
//		boundary(1, 2) = r * c;
//
//		boundaryIDX(1, 1) = 1;
//		boundaryIDX(2, 1) = r;
//		boundaryIDX(1, 2) = c;
//		boundaryIDX(2, 2) = r;
//
//		boundaryIDX(1, 3) = floor(c / 5) + 1;
//		boundaryIDX(2, 3) = r;
//		boundaryIDX(1, 4) = c - floor(c / 5);
//		boundaryIDX(2, 4) = r;
//
//		boundaryIDX(1, 5) = 2 * floor(c / 5) + 1;
//		boundaryIDX(2, 5) = r;
//		boundaryIDX(1, 6) = c - 2 * floor(c / 5);
//		boundaryIDX(2, 6) = r;
//
//	end

void scaleBetta(int* betta, float* beta, float scale) {
	float tmp = scale / (betta[0] * betta[0] + betta[1] * betta[1]);
	beta[0] = betta[0] * tmp;
	beta[1] = betta[1] * tmp;
}

int GCD(int a, int b) {
	while (1) {
		a = a % b;
		if (a == 0)
			return b;
		b = b % a;

		if (b == 0)
			return a;
	}
}

void getForceVector(float ** d, int m, int r, int c, int exampleType) {
	//	int mid = r / 2;
	//	int midr =(r/2)+1;
	//	int midc = (c/2)+1;
	*d=new float[m];
	m = m / 2;
	int tmp;

	switch (exampleType) {
	case 1:
		//		tmp = r * (c - 1) + mid + 1;
		tmp = r * c;
		(*d)[tmp + m - 1] = -1;
		(*d)[tmp - 1] = 2;
		break;
	case 2:
		for (int cc = 2; cc < c; cc++) {
			(*d)[-2 + r * cc + m - 1] = -1;
		}
		break;
	case 3:
		for (int cc = 2; cc < c; cc++) {
			(*d)[-2 + r * cc + m - 1] = -1;
		}
		break;

	case 5:

		break;

	default:
		break;
	}

	//	if (boundarytype==3)
	//	    midr = floor(r/2)+1
	//	    midc = floor(c/2)+1
	//
	//	    for cc=2:c-1
	//	        d(-2+r*cc+m) = -1;
	//	    end
	//	end
	//
	//	if (boundarytype==6)
	//	    midr = floor(r/2)+1
	//	    midc = floor(c/2)+1
	//
	//	    for cc=2:c-1
	//	        d(-1+r*cc+m) = -1;
	//	    end
	//	end
	//
	//	if (boundarytype==4)
	//	    midr = floor(r/2)+1
	//	    midc = floor(c/2)+1
	//
	//
	//
	//	    for cc=3:c-2
	//	        d(-12+r*cc+m) = -1;
	//	    end
	//
	//	   for asdf=1:13
	//	    for cc=6:c-5
	//	         %d(-8-asdf+r*cc+m) = -1;
	//	    end
	//	   end
	//
	//
	//	    for asdf=1:17
	//	        for cc=6:6
	//	         d(-12-asdf+r*cc+m) = -1;
	//	        if (asdf<17)
	//	         d(-12-asdf+r*cc) = (-1)^asdf;
	//	        end
	//	        end
	//	    end
	//
	//
	//	end


}

int maximabs(int * betta) {
	if (abs(betta[0]) >= abs(betta[1]))
		return abs(betta[0]);
	else
		return abs(betta[1]);
}

int minimabs(int * betta) {
	if (abs(betta[0]) <= abs(betta[1]))
		return abs(betta[0]);
	else
		return abs(betta[1]);
}
