/* Copyright (c) 2023-2024 Gilad Odinak */
/* QR decomposistion */
#include <stdint.h>
#include "mem.h"
#include "array.h"
#include "norm.h"
#include "qr.h"

/* QR - Performs QR decomposition of matrix M to an orthogonal matrix Q,
 *      and an upper right triangular matrix R, using Householder
 *      transformation.
 *
 *      If R is NULL, only returns Q matrix. If both Q and R are NULL,
 *      converts M to orthogonal matrix in place. In that case, M must
 *      be a square matrix m == n.
 *
 *      Note that if m <= n, then Q is an orthogonal squared matrix. If
 *      m > n, Q is neither square nor orthogonal, however, each column
 *      of Q has a length (norm) of 1.
 *
 *      Reference:
 *      https://en.wikipedia.org/wiki/QR_decomposition
 *
 * Parameters:
 *   M   - Pointer to the matrix M to be decomposed.
 *   Q   - Pointer to the orthogonal matrix Q (output).
 *   R   - Pointer to the upper right triangular matrix R (output).
 *   m   - Number of rows in matrix M.
 *   n   - Number of columns in matrix M.
 *
 * Returns:
 *   Q   - Orthogonal matrix.
 *   R   - Upper right triangular matrix.
 */
void QR(fArr2D M_/*[m][n]*/, 
        fArr2D Q_/*[m][min(n,m)]*/,
        fArr2D R_/*[min(m,n)][n]*/,
        int m, int n)
{
    int d = (m < n) ? m : n;
    typedef float (*ArrMM)[m];
    typedef float (*ArrMN)[n];
    typedef float(*VecM);
    ArrMM Q = allocmem(m,m,float);
    ArrMN R = allocmem(m,n,float);
    ArrMM Qk = allocmem(m,m,float);
    ArrMN Rr = allocmem(m,n,float);
    ArrMM Qr = allocmem(m,m,float);
    VecM x = allocmem(1,m,float);
    VecM v = allocmem(1,m,float);

    /* Set Q to identity matrix */
    fltclr(Q,m * m);
    for (int i = 0; i < m; i++)
        Q[i][i] = 1.0;

    fltcpy(R,M_,m * n);

    for (int k = 0; k < d; k++) {
        fltclr(x,m - k);
        for (int i = 0, j = k; j < m; i++, j++)
            x[i] = R[j][k];
        /* Construct the householder vector
         * v = x + sign(x[0]) * ||x|| * e
         * where 
         * sign(x): (x < 0) ? -1 : 1
         * e = [1,0,0,..]
         */
        fltclr(v,m - k);
        v[0] = vecnorm(x,m - k);
        if (x[0] < 0)
            v[0] = -v[0];
        for (int i = 0; i < m - k; i++)
            v[i] += x[i];
        /* Normalize it */
        float vn = vecnorm(v,m - k);
        for (int i = 0; i < m - k; i++)
            v[i] /= vn;
        /* Compute the Householder matrix Qk, which is
         * the lower right portion (m-k) x (m-k) of Q
         * Qk = I - 2 * v[k:] * v^T[k:] 
         */
        fltclr(Qk,m * m);
        for (int i = 0; i < m; i++)
            Qk[i][i] = 1.0;
        for (int i = k; i < m; i++)
            for (int j = k; j < m; j++)
                Qk[i][j] -= 2.0 * v[i - k] * v[j - k];

        /* Update R : R = Qk @ R */
        matmul(Rr,Qk,R,m,m,n);
        fltcpy(R,Rr,m*n);
        
        /* Update Q: Q = Q @ Qk.T */
        matmulT(Qr,Q,Qk,m,m,m);
        fltcpy(Q,Qr,m*m);
    }
    if (Q_ != NULL) {
        if (R_ != NULL) /* R[:min(m, n),:] copy the first d rows of R to R_ */
            fltcpy(R_,R,d * n);
        /* Q_ =  Q[:,:min(m, n)] (may point to M_ for inplace update)
         * copy the first d columns of Q to Q_
         */
        if (m != n) { /* Copy row by row, only d elements */
            float *p = (float *) Q_;
            for (int i = 0; i < m; i++, p += d)
                fltcpy(p,Q[i],d);
        }
        else
            fltcpy(Q_,Q,m * m);
    }
    else /* Assume m == n, update M_ in place */
        fltcpy(M_,Q,m * m);

    freemem(Q);
    freemem(R);
    freemem(Qk);
    freemem(Rr);
    freemem(Qr);
    freemem(x);
    freemem(v);
}
