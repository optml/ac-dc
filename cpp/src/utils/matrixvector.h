/*
 * A matrix multiples a vector
 *
 * 
 */
#include <vector>
#include <iostream>
using namespace std;

#ifndef MATRIX_VECTOR_H
#define MATRIX_VECTOR_H


template<typename L, typename D>
void matrixvector(std::vector<D> &A, std::vector<L> &Asub, std::vector<L> &xA, std::vector<D> &b, L &m, std::vector<D> &result) {

	for (L i = 0; i < m; i++){

		D temp = 0;
		L beginrow = xA[i];
		L endrow = xA[i+1] - 1;
		for (L j = beginrow; j <= endrow; j++)
			temp += A[j] * b[Asub[j]];
		result[i] = temp;
	}


}

template<typename L, typename D>
void vectormatrix( std::vector<D> &b, std::vector<D> &A, std::vector<L> &Asub, std::vector<L> &xA, L &m, std::vector<D> &result) {


	for (L i = 0; i < m; i++){
		L beginrow = xA[i];
		L endrow = xA[i+1] - 1;
		for (L j = beginrow; j <= endrow; j++)
			result[Asub[j]] += b[i] * A[j];
	}

}

#endif /* MATRIX_VECTOR_H */
