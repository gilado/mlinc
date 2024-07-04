/* Copyright (c) 2023-2024 Gilad Odinak */
/* SVD decomposistion */
#ifndef SVD_H
#define SVD_H
#include <math.h>
#include "array.h"

/* SVD - Performs SVD decomposition, using QR decomposition, of matrix A to
 *       obtain a left orthogonal matrix U, a vector of non-negative singular
 *       values S, and a right orthogonal matrix V, such that 
 *       A = U @ Sigma @ Vt.
 *
 *       The singular values in vector S represent the diagonal of the diagonal
 *       matrix Sigma, arranged in descending order.
 *
 *       Reference:
 *       https://en.wikipedia.org/wiki/Singular_value_decomposition
 *
 * Parameters:
 *   A   - Pointer to the matrix A to be decomposed.
 *   U   - Pointer to the left orthogonal matrix U (output).
 *   S   - Pointer to the vector of non-negative singular values S (output).
 *   Vt  - Pointer to the transpose of the right orthogonal matrix V (output).
 *   m   - Number of rows in matrix A.
 *   n   - Number of columns in matrix A.
 *
 * Returns:
 *   U   - Left orthogonal matrix U.
 *   S   - Vector of non-negative singular values.
 *   Vt  - Transpose of the right orthogonal matrix V.
 *
 * Notes:
 *   If m >= n then Vt may be NULL, in which case only U and S are returned;
 *   if both U and Vt are NULL, A is updated in place with the value of U.
 *
 *   If m < n then U may be NULL, in which case only Vt and S are returned;
 *   if both U and Vt are NULL, A is updated in place with the value of Vt.
 */
void SVD(const fArr2D A_/*[m][n]*/, 
         fArr2D U_/*[m][n]*/, 
         fVec S_  /*[n]*/,
         fArr2D Vt_/*[n][n]*/,
         int m, int n);

#endif
