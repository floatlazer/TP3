#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"

namespace {
void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock, // 37.1575 secondes
                   const Matrix& A, const Matrix& B, Matrix& C) {
  for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
    for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
      for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
        C(i, j) += A(i, k) * B(k, j);
}
void prodSubBlocks2(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock, // reorder ijk 22.5051 secondes
                   const Matrix& A, const Matrix& B, Matrix& C) {

	for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
	  for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
	  	  for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
			C(i, j) += A(i, k) * B(k, j);
}
void prodSubBlocks3(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,  // mp parallel 6.18911 secondes
                   const Matrix& A, const Matrix& B, Matrix& C) {
# pragma omp parallel for schedule(static)
	for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
	  for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
	  	  for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
			C(i, j) += A(i, k) * B(k, j);
}

const int szBlock = 160;
void prodSubBlocks4(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,  // block sz160 2.3518 secondes 
                   const Matrix& A, const Matrix& B, Matrix& C) {
	for(int ib = 0; ib < A.nbRows; ib += szBlock)
		for(int jb = 0; jb < B.nbCols; jb += szBlock)
			for(int kb = 0; kb < A.nbCols; kb += szBlock )
				prodSubBlocks2(ib, jb, kb, szBlock, A, B, C);
}
void prodSubBlocks5(const Matrix& A, const Matrix& B, Matrix& C) { // block parallel sz160 2.3518 secondes 
# pragma omp parallel for collapse(2) // for first two loop
	for(int ib = 0; ib < A.nbRows; ib += szBlock)
		for(int jb = 0; jb < B.nbCols; jb += szBlock)
			for(int kb = 0; kb < A.nbCols; kb += szBlock )
				prodSubBlocks2(ib, jb, kb, szBlock, A, B, C);
}
}  // namespace

Matrix operator*(const Matrix& A, const Matrix& B) {
  Matrix C(A.nbRows, B.nbCols, 0.0);
  //prodSubBlocks2(0, 0, 0, std::max({A.nbRows, B.nbCols, A.nbCols}), A, B, C);
  prodSubBlocks5(A, B, C);
  return C;
}