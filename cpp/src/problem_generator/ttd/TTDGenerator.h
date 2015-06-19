/*
 * TTDGenerator.h
 *
 *  Created on: Sep 9, 2013
 *      Author: taki
 */

#ifndef TTDGENERATOR_H_
#define TTDGENERATOR_H_
#include "../../class/Context.h"
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

template<typename T>
class TTDGenerator {
	Context & context;
public:
	int columnLength;
	int row;
	int col;
	int dim1;
	int dim2;
	int dim3;
	int exampleType;
	int mOut;
	int nOut;
	std::vector<float> A_values;
	std::vector<int> dim2s_A;
	std::vector<int> dim1s_A;
	std::vector<int> boundary;
	std::vector<int> nodeDescription_from;
	std::vector<int> nodeDescription_to;
	std::vector<float> d;
	float max_length_of_bars;
	int max_possible_bars;
	float kappa;
	TTDGenerator(Context &_context) :
			context(_context) {
		columnLength = 0;
		kappa = 1;
		max_length_of_bars = 100;
		dim1 = context.xDim;
		dim2 = context.yDim;
		dim3 = context.zDim;
		if (dim3 == 0) {
			row = dim1;
			col = dim2;
		}
		exampleType = context.experiment;

	}

	int isPointInBoundary(int point, std::vector<int> & boundary) {
		for (int j = 0; j < boundary.size(); j++) {
			if (boundary[j] == point)
				return 1;
		}
		return 0;
	}

	void generate3DProblem() {
		columnLength = 6;
		float scale = sqrt(kappa);
		int boundarySize = boundary.size();
		int totalNodes = dim1 * dim2 * dim3;
		long totalLength = 0;
		(A_values).resize(0);
		(dim2s_A).resize(0);
		(dim1s_A).resize(0);
		(nodeDescription_from).resize(0);
		(nodeDescription_to).resize(0);
		int nnz = 0;
		int tt = dim3;
		if (dim2 > tt)
			tt = dim2;
		if (dim3 > tt)
			tt = dim3;

		int GSDS[tt + 1][tt + 1][tt + 1];
		for (int i = 0; i <= tt; i++) {
			for (int j = 0; j <= tt; j++) {
				for (int k = 0; k <= tt; k++) {
					int tmpp = 2;
					if (i * j * k == 0) {
						if (i == 0) {
							if (j * k > 0) {
								tmpp = GCD(j, k);
							} else {
								if (k == 1) {
									tmpp = 1;
								} else if (j == 1) {
									tmpp = 1;
								}
							}
						} else if (j == 0) {
							if (i * k > 0) {
								tmpp = GCD(i, k);
							} else {
								if (k == 1) {
									tmpp = 1;
								} else if (i == 1) {
									tmpp = 1;
								}
							}
						} else if (k == 0) {
							if (i * j > 0) {
								tmpp = GCD(i, j);
							} else {
								if (j == 1) {
									tmpp = 1;
								} else if (i == 1) {
									tmpp = 1;
								}
							}
						}
					} else {
						tmpp = GCD(GCD(i, j), k);
					}
					GSDS[i][j][k] = tmpp;
				}
			}
		}

		int betta[3];
		float beta[3];
		int total_bars = 0;
		for (int k = 1; k <= dim3; k++) {
			int nLow = 1;
			int nHigh = dim3;
			if (k - max_length_of_bars > nLow)
				nLow = k - max_length_of_bars;
			if (k + max_length_of_bars < nHigh)
				nHigh = k + max_length_of_bars;
			for (int j = 1; j <= dim2; j++) {

				int mLow = 1;
				int mHigh = dim2;
				if (j - max_length_of_bars > mLow)
					mLow = j - max_length_of_bars;
				if (j + max_length_of_bars < mHigh)
					mHigh = j + max_length_of_bars;
				for (int i = 1; i <= dim1; i++) {

					//	    %[i,j,k] is initial point of bar
					int lLow = 1;
					int lHigh = dim1;
					if (i - max_length_of_bars > lLow)
						lLow = i - max_length_of_bars;
					if (i + max_length_of_bars < lHigh)
						lHigh = i + max_length_of_bars;

					for (int n = k; n <= nHigh; n++) {
						for (int m = mLow; m <= mHigh; m++) {

							for (int l = lLow; l <= lHigh; l++) {
								if (i == l && j == m && k == n) { // the same point
									continue;
								}
								if ((k == n && j == m && l < i)
										|| (k == n && m < j)) {
									continue;
								}

								betta[0] = l - i;
								betta[1] = m - j;
								betta[2] = n - k;
								if ((betta[0] * betta[0] + betta[1] * betta[1]
										+ betta[2] * betta[2])
										> max_length_of_bars) { // only bars with maximal length are allowed
									continue;
								}
								if (GSDS[(int) abs(betta[0])][(int) abs(
										betta[1])][(int) abs(betta[2])] > 1) {
									continue;
								}

								int from = getNodeIDFromVector(i, j, k, dim1,
										dim2, dim3);
								int to = getNodeIDFromVector(l, m, n, dim1,
										dim2, dim3);
								int ta = isPointInBoundary(from, boundary);
								int tb = isPointInBoundary(to, boundary);

								if (ta + tb < 2) {
									scaleBetta3D(betta, beta, scale);
									nodeDescription_from.push_back(from);
									nodeDescription_to.push_back(to);
									A_values.push_back(-beta[0] * (1 - ta));
									dim1s_A.push_back(from);
									dim2s_A.push_back(total_bars + 1);
									nnz++;
									A_values.push_back(-beta[1] * (1 - ta));
									dim1s_A.push_back(from + totalNodes);
									dim2s_A.push_back(total_bars + 1);
									nnz++;
									A_values.push_back(-beta[2] * (1 - ta));
									dim1s_A.push_back(from + totalNodes * 2);
									dim2s_A.push_back(total_bars + 1);
									nnz++;

									A_values.push_back(beta[0] * (1 - tb));
									dim1s_A.push_back(to);
									dim2s_A.push_back(total_bars + 1);
									nnz++;
									A_values.push_back(beta[1] * (1 - tb));
									dim1s_A.push_back(to + totalNodes);
									dim2s_A.push_back(total_bars + 1);
									nnz++;
									A_values.push_back(beta[2] * (1 - tb));
									dim1s_A.push_back(to + totalNodes * 2);
									dim2s_A.push_back(total_bars + 1);
									nnz++;

									total_bars++;
								}
							}
						}
					}
				}
			}
		}

		(A_values).resize(nnz);
		(dim2s_A).resize(nnz);
		(dim1s_A).resize(nnz);
		(nodeDescription_from).resize(total_bars);
		(nodeDescription_to).resize(total_bars);
		nOut = total_bars;
		mOut = dim1 * dim2 * dim3 * 3;
	}

	long get3DProblemTTD_number_of_bars(int dim1, int dim2, int dim3,
			int exampleType, int& mOut, int& nOut, std::vector<int> &boundary,
			int max_length_of_bars) {
		float kappa = 1;
		float scale = sqrt(kappa);
		int boundarySize = boundary.size();
		int nnz = 0;
		int tt = dim3;
		if (dim2 > tt)
			tt = dim2;
		if (dim3 > tt)
			tt = dim3;

		int GSDS[tt + 1][tt + 1][tt + 1];
		for (int i = 0; i <= tt; i++) {
			for (int j = 0; j <= tt; j++) {
				for (int k = 0; k <= tt; k++) {
					int tmpp = 2;
					if (i * j * k == 0) {
						if (i == 0) {
							if (j * k > 0) {
								tmpp = GCD(j, k);
							} else {
								if (k == 1) {
									tmpp = 1;
								} else if (j == 1) {
									tmpp = 1;
								}
							}
						} else if (j == 0) {
							if (i * k > 0) {
								tmpp = GCD(i, k);
							} else {
								if (k == 1) {
									tmpp = 1;
								} else if (i == 1) {
									tmpp = 1;
								}
							}
						} else if (k == 0) {
							if (i * j > 0) {
								tmpp = GCD(i, j);
							} else {
								if (j == 1) {
									tmpp = 1;
								} else if (i == 1) {
									tmpp = 1;
								}
							}
						}
					} else {
						tmpp = GCD(GCD(i, j), k);
					}
					GSDS[i][j][k] = tmpp;
				}
			}
		}

		int betta[3];
		float beta[3];
		long total_bars = 0;
		for (int k = 1; k <= dim3; k++) {
			for (int j = 1; j <= dim2; j++) {
				for (int i = 1; i <= dim1; i++) {
					//	    %[i,j,k] is initial point of bar
					for (int n = k; n <= dim3; n++) {
						for (int m = 1; m <= dim2; m++) {
							for (int l = 1; l <= dim1; l++) {
								if (i == l && j == m && k == n) { // the same point
									continue;
								}
								if ((k == n && j == m && l < i)
										|| (k == n && m < j)) {
									continue;
								}

								betta[0] = l - i;
								betta[1] = m - j;
								betta[2] = n - k;
								if ((betta[0] * betta[0] + betta[1] * betta[1]
										+ betta[2] * betta[2])
										> max_length_of_bars) { // only bars with maximal length are allowed
									continue;
								}
								if (GSDS[(int) abs(betta[0])][(int) abs(
										betta[1])][(int) abs(betta[2])] > 1) {
									continue;
								}

								int from = getNodeIDFromVector(i, j, k, dim1,
										dim2, dim3);
								int to = getNodeIDFromVector(l, m, n, dim1,
										dim2, dim3);
								int ta = isPointInBoundary(from, boundary);
								int tb = isPointInBoundary(to, boundary);

								if (ta + tb < 2) {
									scaleBetta3D(betta, beta, scale);

									nnz = nnz + 6;
									total_bars++;
								}
							}
						}
					}
				}
			}
		}

		nOut = total_bars;
		mOut = dim1 * dim2 * dim3 * 3;
		return total_bars;
	}

	void generate2DProblem() {
		float kappa = 1;
		float scale = sqrt(kappa);
		int boundarySize = boundary.size();
		int totalNodes = dim1 * dim2;
		long totalLength = 0;
		A_values.resize(0);
		dim2s_A.resize(0);
		dim1s_A.resize(0);
		nodeDescription_from.resize(0);
		nodeDescription_to.resize(0);
		int nnz = 0;
		int tt = dim2;
		if (dim1 > tt)
			tt = dim1;
		int GSDS[tt + 1][tt + 1];
		for (int i = 0; i <= tt; i++) {
			for (int j = i; j <= tt; j++) {
				if (i * j == 0) {
					if (i > 1 || j > 1) {
						GSDS[i][j] = 2;
						GSDS[j][i] = 2;
					} else {
						GSDS[i][j] = 1;
						GSDS[j][i] = 1;
					}
				} else {
					GSDS[i][j] = GCD(i, j);
					GSDS[j][i] = GCD(i, j);
				}
			}
		}
		int betta[2];
		float beta[2];
		int total_bars = 0;
		for (int j = 1; j <= dim2; j++) {
			for (int i = 1; i <= dim1; i++) {
				//	    %[i,j] is initial point of bar

				int mLow = 1;
				int mHigh = dim2;
				if (j - max_length_of_bars > mLow)
					mLow = j - max_length_of_bars;
				if (j + max_length_of_bars < mHigh)
					mHigh = j + max_length_of_bars;

				for (int m = mLow; m <= mHigh; m++) {

					int lLow = 1;
					int lHigh = dim1;
					if (i - max_length_of_bars > lLow)
						lLow = i - max_length_of_bars;
					if (i + max_length_of_bars < lHigh)
						lHigh = i + max_length_of_bars;

					for (int l = lLow; l <= lHigh; l++) {
						if (i == l && j == m) { // the same point
							continue;
						}
						if ((j == m && l < i) || (m < j)) {
							continue;
						}
						betta[0] = l - i;
						betta[1] = m - j;
						if ((betta[0] * betta[0] + betta[1] * betta[1])
								> max_length_of_bars) { // only bars with maximal length are allowed
							continue;
						}
						if (GSDS[(int) abs(betta[0])][(int) abs(betta[1])]
								> 1) {
							continue;
						}

						int from = getNodeIDFromVector(i, j, dim1, dim2);
						int to = getNodeIDFromVector(l, m, dim1, dim2);
						int ta = isPointInBoundary(from, boundary);
						int tb = isPointInBoundary(to, boundary);

						if (ta + tb < 2) {
							scaleBetta(betta, beta, scale);


							nodeDescription_from.push_back( from);
							nodeDescription_to.push_back(to);

							A_values.push_back( -beta[0] * (1 - ta));
							dim1s_A.push_back( from);
							dim2s_A.push_back( total_bars + 1);
							nnz++;
							A_values.push_back( -beta[1] * (1 - ta));
							dim1s_A.push_back( from + totalNodes);
							dim2s_A.push_back( total_bars + 1);
							nnz++;

							A_values.push_back( beta[0] * (1 - tb));
							dim1s_A.push_back( to);
							dim2s_A.push_back(  total_bars + 1);
							nnz++;
							A_values.push_back(  beta[1] * (1 - tb));
							dim1s_A.push_back(  to + totalNodes);
							dim2s_A.push_back( total_bars + 1);
							nnz++;
							total_bars++;
						}
					}
				}
			}
		}
		nOut = total_bars;
		mOut = dim1 * dim2 * 2;
	}

	void generate2DFixedPoints() {

		switch (exampleType) {
		case 1:
			boundary.resize(row);
			for (int i = 1; i <= row; i++) {
				boundary[i] = getNodeIDFromVector(i, 1, row, col);
			}
			break;
		case 2:
			boundary.resize(4);
			boundary[0] = getNodeIDFromVector(1, 1, row, col);
			boundary[1] = getNodeIDFromVector(row, 1, row, col);
			boundary[2] = getNodeIDFromVector(1 + row / 3, 1, row, col);
			boundary[3] = getNodeIDFromVector(row - row / 3, 1, row, col);
			break;
		case 3:
			boundary.resize(4);
			boundary[0] = getNodeIDFromVector(1, 1, row, col);
			boundary[1] = getNodeIDFromVector(row, 1, row, col);
			boundary[2] = getNodeIDFromVector(1 + row / 3, 1, row, col);
			boundary[3] = getNodeIDFromVector(row - row / 3, 1, row, col);
			break;
		default:
			break;
		}
	}

	void generate3DFixedPoints() {

		switch (exampleType) {
		case 1:
			(boundary).resize(4);
			(boundary)[0] = getNodeIDFromVector(1, 1, 1, dim1, dim2, dim3);
			(boundary)[1] = getNodeIDFromVector(dim1, 1, 1, dim1, dim2, dim3);
			(boundary)[2] = getNodeIDFromVector(1, dim2, 1, dim1, dim2, dim3);
			(boundary)[3] = getNodeIDFromVector(dim1, dim2, 1, dim1, dim2,
					dim3);
			break;
		case 2:
			(boundary).resize(4);
			(boundary)[0] = getNodeIDFromVector(1, 1, 1, dim1, dim2, dim3);
			(boundary)[1] = getNodeIDFromVector(dim1, 1, 1, dim1, dim2, dim3);
			(boundary)[2] = getNodeIDFromVector(1, dim2, 1, dim1, dim2, dim3);
			(boundary)[3] = getNodeIDFromVector(dim1, dim2, 1, dim1, dim2,
					dim3);
			break;
		case 3:
			(boundary).resize(4);
			(boundary)[0] = getNodeIDFromVector(1, 1, 1, dim1, dim2, dim3);
			(boundary)[1] = getNodeIDFromVector(dim1, 1, 1, dim1, dim2, dim3);
			(boundary)[2] = getNodeIDFromVector(1, dim2, 1, dim1, dim2, dim3);
			(boundary)[3] = getNodeIDFromVector(dim1, dim2, 1, dim1, dim2,
					dim3);
			break;
		default:
			break;
		}
	}

	void get2DForceVector() {
		int m = mOut / 2;
		int r = dim1;
		int c = dim2;
		d.resize(mOut);
		switch (exampleType) {
		case 1:
			d[getNodeIDFromVector(r / 2, c, r, c) + m - 1] = -1;
			d[getNodeIDFromVector(r / 2, c, r, c) - 1] = 2;
			break;
		case 2:
			for (int cc = 2; cc <= r - 1; cc++) {
				d[getNodeIDFromVector(cc, 3, r, c) - 1 + m] = -1;
			}
			break;
		case 3:
			for (int cc = 2; cc <= r - 1; cc++) {
				for (int rr = 2; rr <= c - 1; rr++) {
					d[getNodeIDFromVector(cc, rr, r, c) - 1 + m] = -1;
				}
			}
			break;
		default:
			break;
		}
	}

	void get3DForceVector() {
		int m = mOut / 3;
		d.resize(mOut, 0);
		switch (exampleType) {
		case 1:
			(d)[getNodeIDFromVector(dim1 / 2, dim2 / 2, dim3, dim1, dim2, dim3)
					+ 2 * m - 1] = -1;
			break;
		case 2:
			for (int i = 0; i < dim1; i++) {
				for (int j = 0; j < dim2; j++) {
					for (int k = 1; k < dim3; j++) {
						(d)[getNodeIDFromVector(i, j, k, dim1, dim2, dim3)
								+ 1 * m - 1] = -1;
					}
				}
			}
			break;
		case 3:
			for (int i = 0; i < dim1; i++) {
				for (int j = 0; j < dim2; j++) {
					(d)[getNodeIDFromVector(i, j, dim3, dim1, dim2, dim3)
							+ 2 * m - 1] = -1;
				}
			}
			break;
		default:
			break;
		}
	}

	int minimabs2D(int * betta) {
		if (abs(betta[0]) <= abs(betta[1]))
			return abs(betta[0]);
		else
			return abs(betta[1]);
	}

	void saveSolutionIntoFile(const char *msg, float* x, int n,
			int* nodeDescriptionFrom, int* nodeDescriptionTo, float treshHold) {
		int writtenBars = 0;
		FILE *fp;
		fp = fopen(msg, "w");
		for (int i = 0; i < n; i++) {
			if (abs(x[i]) > treshHold) {
				writtenBars++;
				fprintf(fp, "%d,%d,%f\n", nodeDescriptionFrom[i],
						nodeDescriptionTo[i], x[i]);
			}
		}
		fclose(fp);
		printf("Number of written bars:%d\n", writtenBars);
	}

	void storeProblemIntoFile() {
		std::ofstream file;
		std::stringstream ss;
		ss << context.resultFile << "_matrix.txt";
		file.open(ss.str().c_str());

		for (int i = 0; i < A_values.size(); i++) {
			file << std::setprecision(16) << dim1s_A[i] << " " << dim2s_A[i]
					<< " " << A_values[i] << std::endl;
		}

		file.close();
		ss.str("");
		ss << context.resultFile << "_forces.txt";
		file.open(ss.str().c_str());

		for (int i = 0; i < d.size(); i++) {
			file << d[i] << std::endl;
		}
		file.close();

	}

	int maximabs2D(int * betta) {
		if (abs(betta[0]) >= abs(betta[1]))
			return abs(betta[0]);
		else
			return abs(betta[1]);
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

	void scaleBetta(int* betta, float* beta, float scale) {
		float tmp = scale / (betta[0] * betta[0] + betta[1] * betta[1]);
		beta[0] = betta[0] * tmp;
		beta[1] = betta[1] * tmp;
	}

	int getNodeIDFromVector(int x, int y, int z, int dim1, int dim2, int dim3) {
		return x + (y - 1) * dim1 + (z - 1) * dim1 * dim2;
	}

	int getNodeIDFromVector(int x, int y, int dim1, int dim2) {
		return x + (y - 1) * dim1;
	}
	double computeTTDObjectiveValue(float* A_TTD, int* Row_IDX, int* Col_IDX,
			float* x, int n, int m, float* b, int nnz, float lambda) {
		double objectiveValue = 0;
		double residuals[m];
		for (int i = 0; i < m; i++) {
			residuals[i] = -b[i];
		}
		for (int i = 0; i < nnz; i++) {
			residuals[Row_IDX[i] - 1] += A_TTD[i] * x[Col_IDX[i] - 1];
		}
		double g_sq = 0;
		for (int i = 0; i < m; i++) {
			g_sq += residuals[i] * residuals[i];
		}
		double x_norm = 0;
		for (int i = 0; i < n; i++) {
			x_norm += abs(x[i]);
		}
		objectiveValue = x_norm * lambda + 0.5 * g_sq;

		return objectiveValue;
	}

	void scaleBetta3D(int* betta, float* beta, float scale) {
		float tmp = scale
				/ (betta[0] * betta[0] + betta[1] * betta[1]
						+ betta[2] * betta[2]);
		beta[0] = betta[0] * tmp;
		beta[1] = betta[1] * tmp;
		beta[2] = betta[2] * tmp;
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

	void getVectorFromNodeId(int idd, int dim1, int dim2, int dim3, int* x,
			int*y, int* z) {
		float id = idd;
		int zT = (int) ceil(id / (dim1 * dim2));
		id = id - (zT - 1) * dim1 * dim2;
		int yT = (int) ceil(id / dim1);
		id = id - (yT - 1) * dim1;
		int xT = (int) id;
		(*x) = xT;
		(*y) = yT;
		(*z) = zT;
	}

	void getVectorFromNodeId(int idd, int dim1, int dim2, int *x, int* y) {
		float id = idd;
		int yT = (int) ceil(id / dim1);
		id = id - (yT - 1) * dim1;
		int xT = (int) id;
		(*x) = xT;
		(*y) = yT;
	}

};

#endif /* TTDGENERATOR_H_ */
