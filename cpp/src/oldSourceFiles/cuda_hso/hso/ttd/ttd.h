/*
 * This file contains functions to generate TTD problem on uniform 2D or 3D grid
 * with sample force and boundary vectors
 */

void scaleBetta(int* betta, float* beta, float scale);
int minimabs(int * betta);
int maximabs(int * betta);

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

void getThrust2DProblemTTD(int row, int col, int exampleType, int* mOut,
		int* nOut, thrust::host_vector<float>* A_values, thrust::host_vector<
				int>* Cols_A, thrust::host_vector<int>* Rows_A,
		thrust::host_vector<int> boundary,
		thrust::host_vector<int>* nodeDescription, int rowScale,
		int max_possible_bars) {
	float kappa = 1;
	float scale = sqrt(kappa);
	int boundarySize = boundary.size();
	int m = row * col;
	long totalLength = max_possible_bars;
	(*A_values).resize(max_possible_bars);
	(*Cols_A).resize(max_possible_bars);
	(*Rows_A).resize(max_possible_bars);
	totalLength = totalLength * 4;
	(*nodeDescription).resize(totalLength);
	int nodeDescriptionId = 0;
	int nnz = 0;
	int node = 0;
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
		for (int i = 1; i <= row; i++) {
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
						(*A_values)[nnz] = -(1 - totalMatchA) * beta[0];
						(*Rows_A)[nnz] = tmp;
						(*Cols_A)[nnz] = node;
						nnz++;
						(*A_values)[nnz] = -(1 - totalMatchA) * beta[1];
						(*Rows_A)[nnz] = tmp + m;
						(*Cols_A)[nnz] = node;
						nnz++;
						tmp = row * (j) + k + (l - 1) * row;
						(*A_values)[nnz] = (1 - totalMatchB) * beta[0];
						(*Rows_A)[nnz] = tmp;
						(*Cols_A)[nnz] = node;
						nnz++;
						(*A_values)[nnz] = (1 - totalMatchB) * beta[1];
						(*Rows_A)[nnz] = tmp + m;
						(*Cols_A)[nnz] = node;
						nnz++;

						(*nodeDescription)[nodeDescriptionId] = i;
						(*nodeDescription)[nodeDescriptionId + 1] = j;
						(*nodeDescription)[nodeDescriptionId + 2] = k;
						(*nodeDescription)[nodeDescriptionId + 3] = l + j;
						nodeDescriptionId += 4;
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
					(*A_values)[nnz] = -(1 - totalMatchA);
					(*Rows_A)[nnz] = tmp + m;
					(*Cols_A)[nnz] = node;
					nnz++;
					(*A_values)[nnz] = 0; //fake node
					(*Rows_A)[nnz] = tmp + 1;
					(*Cols_A)[nnz] = node;
					nnz++;
					(*A_values)[nnz] = 0; //fake node
					(*Rows_A)[nnz] = tmp + 2;
					(*Cols_A)[nnz] = node;
					nnz++;

					(*A_values)[nnz] = (1 - totalMatchB);
					(*Rows_A)[nnz] = tmp + m + 1;
					(*Cols_A)[nnz] = node;
					nnz++;
					(*nodeDescription)[nodeDescriptionId] = i;
					(*nodeDescription)[nodeDescriptionId + 1] = j;
					(*nodeDescription)[nodeDescriptionId + 2] = i + 1;
					(*nodeDescription)[nodeDescriptionId + 3] = j;
					nodeDescriptionId += 4;
				}
			}
		}
	}
	(*A_values).resize(nnz);
	(*Cols_A).resize(nnz);
	(*Rows_A).resize(nnz);

	*nOut = node;
	*mOut = row * col * 2;
}

void getThrust2DFixedPointsVector(int row, int col, int exampleType,
		thrust::host_vector<int>* boundary) {

	switch (exampleType) {
	case 1:
		(*boundary).resize(row);
		for (int i = 0; i < row; i++) {
			(*boundary)[i] = i + 1;
		}
		break;
	case 2:
		(*boundary).resize(4);
		(*boundary)[0] = row;
		(*boundary)[1] = row * col;
		(*boundary)[2] = row * col - row * (col / 3);
		(*boundary)[3] = row + row * (col / 3);
		break;
	case 3:
		(*boundary).resize(4);
		(*boundary)[0] = row - 5;
		(*boundary)[1] = row * col - 5;
		(*boundary)[2] = row * col - row * (col / 3);
		(*boundary)[3] = row + row * (col / 3);
		break;

	case 5:
		(*boundary).resize(0);
		break;
	case 7:
		(*boundary).resize(2);
		(*boundary)[0] = row;
		(*boundary)[1] = row * col;
		break;
	case 8:
		(*boundary).resize(4);
		(*boundary)[0] = row;
		(*boundary)[1] = row * col;
		(*boundary)[2] = row * col - row * (col / 3);
		(*boundary)[3] = row + row * (col / 3);
		break;
	default:
		break;
	}
}

void scaleBetta(int* betta, float* beta, float scale) {
	float tmp = scale / (betta[0] * betta[0] + betta[1] * betta[1]);
	beta[0] = betta[0] * tmp;
	beta[1] = betta[1] * tmp;
}

void getThrus2DForceVector(thrust::host_vector<float>* d, int m, int r, int c,
		int exampleType) {
	m = m / 2;
	int tmp;

	switch (exampleType) {
	case 1:
		tmp = r * c;
		(*d)[tmp + m - 1] = -1;
		(*d)[tmp - 1] = 2;
		break;
	case 2:
		for (int cc = 2; cc < c - 1; cc++) {
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
	case 7:
		for (int cc = 3; cc < c - 3 + 1; cc++) {
			(*d)[-5 + r * cc + m - 1] = -1;
		}
		break;
	case 8:
		for (int cc = 2; cc < c - 1; cc++) {
			(*d)[-2 + r * cc + m - 1 - 5] = -1;
		}
		break;
	default:
		break;
	}

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

