# include <thread>
# include <cassert>
# include <iostream>
#if defined(_OPENMP)
# include <omp.h>
#endif
# include "ProdMatMat.hpp"

static void prodSubBlocks( int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                                      const Matrix& A, const Matrix& B, Matrix& C )
{
  for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB+szBlock); j++ )
    for ( int k = iColBlkA; k < std::min( A.nbCols, iColBlkA+szBlock); k++ )
      for ( int i = iRowBlkA; i < std::min( A.nbRows, iRowBlkA+szBlock); ++i )
                C(i,j) += A(i,k)*B(k,j);
}

static void naive_prodMatMat( const Matrix& A, const Matrix& B, Matrix& C  )
{
    prodSubBlocks( 0, 0, 0, A.nbRows, A, B, C );
}

static void ompnaive_prodMatMat( const Matrix& A, const Matrix& B, Matrix& C  )
{
# pragma omp parallel for
  for (int j = 0; j < B.nbCols; j++ )
    for ( int k = 0; k < A.nbCols; k++ )
      for ( int i = 0; i < A.nbRows; ++i )
          C(i,j) += A(i,k)*B(k,j);  
}

static int sizeBlock = 32;

static void block_prodMatMat( const Matrix& A, const Matrix& B, Matrix& C )
{
  for (int iBlock = 0; iBlock < A.nbRows; iBlock += sizeBlock )
    for ( int jBlock = 0; jBlock < B.nbCols; jBlock += sizeBlock )
      for ( int kBlock = 0; kBlock < A.nbCols; kBlock += sizeBlock )
	prodSubBlocks( iBlock, jBlock, kBlock, sizeBlock, A, B, C );
}

static void
ompblock1_prodMatMat( const Matrix& A, const Matrix& B, Matrix& C )
{
# pragma omp parallel for
  for (int iBlock = 0; iBlock < A.nbRows; iBlock += sizeBlock )
    for ( int jBlock = 0; jBlock < B.nbCols; jBlock += sizeBlock )
      for ( int kBlock = 0; kBlock < A.nbCols; kBlock += sizeBlock )
	prodSubBlocks( iBlock, jBlock, kBlock, sizeBlock, A, B, C );  
}

static void
ompblock2_prodMatMat( const Matrix& A, const Matrix& B, Matrix& C )
{
# pragma omp parallel for collapse(2)
  for (int iBlock = 0; iBlock < A.nbRows; iBlock += sizeBlock )
    for ( int jBlock = 0; jBlock < B.nbCols; jBlock += sizeBlock )
      for ( int kBlock = 0; kBlock < A.nbCols; kBlock += sizeBlock )
	prodSubBlocks( iBlock, jBlock, kBlock, sizeBlock, A, B, C );  
}

static std::function<void( const Matrix&, const Matrix&, Matrix& )> fct_prod(&naive_prodMatMat);

void setProdMatMat( prod_algo algo )
{
  switch(algo) {
  case naive:
    fct_prod = std::function<void( const Matrix&, const Matrix&, Matrix& )>(&naive_prodMatMat);
    break;
  case block:
    fct_prod = std::function<void( const Matrix&, const Matrix&, Matrix& )>(&block_prodMatMat);
    break;    
  case parallel_naive:
    fct_prod = std::function<void( const Matrix&, const Matrix&, Matrix& )>(&ompnaive_prodMatMat);
    break;
  case parallel_block1:
    fct_prod = std::function<void( const Matrix&, const Matrix&, Matrix& )>(&ompblock1_prodMatMat);
    break;
  case parallel_block2:
    fct_prod = std::function<void( const Matrix&, const Matrix&, Matrix& )>(&ompblock2_prodMatMat);
    break;
  default:
    std::cerr << "Unknown algorithm !\n";
    exit(-1);
  }
}

void setBlockSize( int size )
{
  assert(size > 0);
  sizeBlock = size;  
}

void setNbThreads( int n )
{
#if defined(_OPENMP)
  omp_set_num_threads(n);
#endif
}

Matrix operator* ( const Matrix& A, const Matrix& B )
{
  Matrix C(A.nbRows, B.nbCols, 0.0);
  fct_prod(A,B,C);
  return C;
}
