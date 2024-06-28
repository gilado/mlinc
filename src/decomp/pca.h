/* Copyright (c) 2023-2024 Gilad Odinak */
/* Principal Component Analysis (PCA)   */
#ifndef PCA_H
#define PCA_H
#include <math.h>
#include "mem.h"
#include "array.h"
#include "svd.h"

/* Performs Principal Component Analysis, reducing the dimensionality 
 * of the input matrix A to the specified number of components.
 *
 * https://en.wikipedia.org/wiki/Principal_component_analysis
 * 
 * A  - an array mxn of m observations, each having n features.
 * nc - the desired number of principal components.
 * R  - a reduced array mxnc of m observations, each having nc components
 *
 * if nc > n, it is adjusted to n.
 *
 * Normally the values of A need to be scaled, mean centered and normalized
 * to unit variance, before calling this function.
 *
 */
static inline void PCA(const fArr2D A/*[m][n]*/, 
                       fArr2D R/*[m][nc]*/, int m, int n, int nc)
{
    if (nc <= 0)
        return;
    if (nc > n)
        nc = n;
    fArr2D U  = allocmem(m,n,float);
    fVec   S  = allocmem(1,n,float);
    fArr2D Vt = allocmem(n,n,float);
    SVD(A,U,S,Vt,m,n);
    /* Vt has n rows, multiply by only the transponsed first nc rows of Vt */
    matmulT(R,A,Vt,m,n,nc);
    freemem(U);
    freemem(S);
    freemem(Vt);
}

#endif
