/* Copyright (c) 2023-2024 Gilad Odinak */
/* SVD decomposistion */
#include <stdio.h>
#include <stdint.h>
#include "mem.h"
#include "array.h"
#include "svd.h"

#define SVD_TOL  3e-23 /* float tol is 1.401298e-45, but 3.0e-23 works better */
#define SVD_EPS  3e-13 /* float eps is 2.220446e-16, but 3.0e-13 works better */
#define MAX_ITER   100

static void svd_tall(int m, int n, fArr2D a_/*[m][n]*/, 
                     fVec q_/*[n]*/, fArr2D u_/*[m][n]*/, fArr2D v_/*[n][n]*/);

static void reorder_tall(int m,int n,
                         fVec q_/*[n]*/, fArr2D u_/*[m][n]*/, fArr2D vt_/*[n][n]*/);
static void reorder_wide(int m,int n,
                         fVec q_/*[m]*/, fArr2D u_/*[m][m]*/, fArr2D vt_/*[m][n]*/);
                
/* SVD - Performs SVD decomposition, using QR decomposition, of matrix A to
 *       obtain a left orthogonal matrix U, a vector of non-negative singular
 *       values S, and a right orthogonal matrix V, such that A = U @ Sigma @
 *       Vt.
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
 * This implementation follows the algorithm contributed by Golub and Reinsch
 * to Handbook Series Linear Algerbra Number.Math 14, pg 403-420 (1970)
 */
void SVD(const fArr2D A_/*[m][n]*/, 
         fArr2D U_/*[m][n]*/, 
         fVec S_  /*[n]*/, 
         fArr2D Vt_/*[n][n]*/,
         int m, int n)
{
    typedef float (*ArrNN)[n];
    typedef float (*ArrNM)[m];
    
    if (m >= n) {
        ArrNN V = allocmem(n,n,float);
        svd_tall(m,n,A_,S_,U_,V);
        transpose(V,Vt_,n,n);
        freemem(V);
        reorder_tall(m,n,S_,U_,Vt_);
    }
    else {
        ArrNM At = allocmem(n,m,float);
        ArrNM V = allocmem(n,m,float);
        transpose(A_,At,m,n);
        svd_tall(n,m,At,S_,V,U_);
        transpose(V,Vt_,n,m);
        freemem(At);
        freemem(V);
        reorder_wide(m,n,S_,U_,Vt_);
    }
}

/* Computes the singular values and complete orthogonal decomposition
 * of a real rectangular matrix A, A = U diag(q) V.T, U.T @ U = V.T @ V = I,
 * where the arrays a[m,n], u[m,n], v[n,n], q[n]              
 * represent A, U, V, q respectively. 
 * assume m >= n.
 */
static void svd_tall(int m, int n, fArr2D a_/*[m][n]*/, 
                     fVec q_/*[n]*/, fArr2D u_/*[m][n]*/, fArr2D v_/*[n][n]*/)
{
    typedef float (*ArrMN)[n];
    typedef float (*ArrNN)[n];
    typedef float (*VecN);
    ArrMN a = (ArrMN) a_;
    VecN  q = (VecN)  q_;
    ArrMN u = (ArrMN) u_;
    ArrNN v = (ArrNN) v_;

    float tol = SVD_TOL;
    float eps = SVD_EPS;
    int max_iter = MAX_ITER;

    int i, j, k, l;
    float c, f, g, h, s, x, y, z;

    float e[n];

    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            u[i][j] = a[i][j];
            
    /* Householder reduction to bidiagonal form */

    g = x = 0.0;
    for (i = 0; i < n; i++) {
        e[i] = g;
        l = i + 1;

        s = 0.0;
        for (j = i; j < m; j++)
            s += u[j][i] * u[j][i];

        if (s < tol)
            g = 0.0;
        else {
            f = u[i][i];
            g = (f < 0.0) ? sqrt(s) : -sqrt(s);
            h = f * g - s;
            u[i][i] = f - g;
            
            for (j = l; j < n; j++) {
                s = 0.0;
                for (k = i; k < m; k++)
                    s += u[k][i] * u[k][j];

                f = s / h;
                for (k = i; k < m; k++)
                    u[k][j] += f * u[k][i];
            }
        }
                    
        q[i] = g;
        s = 0.0;
        for (j = l; j < n; j++)
            s += u[i][j] * u[i][j];

        if (s < tol)
            g = 0.0;
        else {
            f = u[i][i + 1];
            g = (f < 0.0) ? sqrt(s) : -sqrt(s);
            h = f * g - s;
            
            u[i][i + 1] = f - g;
            for (j = l; j < n; j++)
                e[j] = u[i][j] / h;

            for (j = l; j < m; j++) {
                s = 0.0;
                for (k = l; k < n; k++)
                    s += u[j][k] * u[i][k];

                for (k = l; k < n; k++)
                    u[j][k] += s * e[k];
            }
        }

        y = fabsf(q[i]) + fabsf(e[i]);
        if (y > x)
            x = y;
    }
    
    /* accumulation of right-hand transformations */
    l = n;    
    for (i = n - 1; i >= 0; i--) {
        if (g != 0.0) {
            h = u[i][i + 1] * g;
            for (j = l; j < n; j++)
                v[j][i] = u[i][j] / h;

            for (j = l; j < n; j++) {
                s = 0.0;
                for (k = l; k < n; k++)
                    s += u[i][k] * v[k][j];

                for (k = l; k < n; k++)
                    v[k][j] += s * v[k][i];
            }
        }
        for (j = l; j < n; j++)
            v[i][j] = v[j][i] = 0.0;

        v[i][i] = 1.0;
        g = e[i];
        l = i;
    }
                
    /* accumulation of left-hand transformations */
    for (i = n - 1; i >= 0; i--) {
        g = q[i];
        l = i + 1;
        for (j = l; j < n; j++)
            u[i][j] = 0.0;     

        if (g != 0.0) {
            h = u[i][i] * g;
            for (j = l; j < n; j++) {
                s = 0.0;
                for (k = l; k < m; k++)
                    s += u[k][i] * u[k][j];

                f = s / h;
                for (k = i; k < m; k++)
                    u[k][j] += f * u[k][i];
            }
            for (j = i; j < m; j++)
                u[j][i] /= g;
        }
        else {
            for (j = i; j < m; j++)
                u[j][i] = 0.0;
        }        

        u[i][i] += 1.0;
    }

    /* diagonalization of the bidiagonal form */
    eps *= x;
    for (k = n - 1; k >= 0; k--) {
        /* test for splitting */
        for (int iter = 0; iter < max_iter; iter++) {
            /* convergence test */
            for (int once = 1; once > 0; once--) {

                /* Note: in the paper, in this loop, l reaches 0 => bug */
                for (l = k; l > 0; l--) {
                    if (fabsf(e[l]) <= eps)
                        break; /* test for convergence */
                    if (fabsf(q[l - 1]) <= eps)
                        break; /* cancellation */
                } 
                if (fabsf(e[l]) <= eps || l == 0)
                    break; /* test for convergence */

                /* cancellation (l > 0) */
                c = 0.0;
                s = 1.0;
                for (i = l; i < k; i++) {
                    f = s * e[i];
                    e[i] = c * e[i];
                    if (fabsf(f) <= eps)
                        break; /* test for convergence */

                    g = q[i];
                    q[i] = sqrt(f * f + g * g);
                    h = q[i];
                    c = g / h;
                    s = -f / h;

                    for (j = 0; j < m; j++) {
                        y = u[j][l - 1];
                        z = u[j][i];
                        u[j][l - 1] = y * c + z * s;
                        u[j][i] = -y * s + z * c;
                    }
                }
            }        

            /* test for convergence */
            z = q[k];
            if (l == k)
                break; /* convergence */

            /* shift from bottom 2 X 2 minor */
            
            x = q[l]; 
            y = q[k - 1]; 
            g = e[k - 1]; 
            h = e[k]; 
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y); 
            g = sqrt(f * f + 1.0); 
            float t = (f < 0.0) ? f - g : f + g;
            f = ((x - z) * (x + z) + h * (y / t - h)) / x;

            /* Next Q R transformation */

            c = s = 1.0;
            for (i = l + 1; i <= k; i++) {
                g = e[i]; 
                y = q[i]; 
                h = s * g; 
                g = c * g; 
                z = sqrt(f * f + h * h);
                e[i - 1] = z; 
                c = f / z; 
                s = h / z; 
                f = x * c + g * s; 
                g = -x * s + g * c; 
                h = y * s; 
                y = y * c; 

                for (j = 0; j < n; j++) {
                    x = v[j][i - 1]; 
                    z = v[j][i]; 
                    v[j][i - 1] = x * c + z * s; 
                    v[j][i] = -x * s + z * c;
                }
                z = sqrt(f * f + h * h);
                q[i - 1] = z; 
                c = f / z; 
                s = h / z; 
                f = c * g + s * y; 
                x = -s * g + c * y;

                for (j = 0; j < m; j++) {
                    y = u[j][i - 1]; 
                    z = u[j][ i]; 
                    u[j][i - 1] = y * c + z * s; 
                    u[j][i] = -y * s + z * c;
                }
            }
            e[l] = 0; 
            e[k] = f; 
            q[k] = x; 
        }

        /* convergence */
        if (q[k] < 0.0) { /* ensure q values are positive */
            q[k] = -z;
            for (j = 0; j < n; j++)
                v[j][k] = -v[j][k];
        }
    }
}

/* Reorders the values of q[n] in descending order, 
 * and the columns  of u[m,n] and the rows of vt[n,n]
 * such that a = u diag(q) vt is preserved.
 * assume m >= n
 */
static void reorder_tall(int m,int n,
                         fVec q_/*[n]*/, fArr2D u_/*[m][n]*/, fArr2D vt_/*[n][n]*/)
{
    typedef float (*VecN);
    typedef float (*ArrMN)[n];
    typedef float (*ArrNN)[n];
    VecN  q =  (VecN)  q_;
    ArrMN u =  (ArrMN) u_;
    ArrNN vt = (ArrNN) vt_;

    int i, j, k;
    float q_t;
    float u_t[m];
    float vt_t[n];

    /* Check whether q already is ordered */
    for (i = 1; i < n - 1; i++)
        if (q[i - 1] < q[i])
            break;
    if (i == n)
        return;
    
    for(i = 1; i < n; i++) {
        q_t = q[i];
        
        for (k = 0; k < m; k++)
            u_t[k] = u[k][i];
        for (k = 0; k < n; k++)
            vt_t[k] = vt[i][k];
            
        for (j = i - 1; j >= 0 && q[j] < q_t; j--) {
            q[j + 1] = q[j];
            for (k = 0; k < m; k++) 
                u[k][j + 1] = u[k][j];
            for (k = 0; k <  n; k++) 
                vt[j + 1][k] = vt[j][k];
        }
        q[j + 1] = q_t;

        for (k = 0; k < m; k++)
            u[k][j + 1] = u_t[k];

        for (k = 0; k < n; k++)
            vt[j + 1][k] = vt_t[k];
    }
}

/* Reorders the values of q[m] in descending order, 
 * and the columns  of u[m,m] and the rows of vt[m,n]
 * such that a = u diag(q) vt is preserved.
 * assume m < n
 */
static void reorder_wide(int m,int n,
                         fVec q_/*[m]*/, fArr2D u_/*[m][m]*/, fArr2D vt_/*[m][n]*/)
{
    typedef float (*VecM);
    typedef float (*ArrMM)[m];
    typedef float (*ArrMN)[n];
    VecM  q =  (VecM)  q_;
    ArrMM u =  (ArrMM) u_;
    ArrMN vt = (ArrMN) vt_;

    int i, j, k;
    float q_t;
    float u_t[m];
    float vt_t[n];

    /* Check whether q already is ordered */
    for (i = 1; i < m - 1; i++)
        if (q[i - 1] < q[i])
            break;
    if (i == m)
        return;

    for(i = 1; i < m; i++) {
        q_t = q[i];
        
        for (k = 0; k < m; k++)
            u_t[k] = u[k][i];
        for (k = 0; k < n; k++)
            vt_t[k] = vt[i][k];
            
        for (j = i - 1; j >= 0 && q[j] < q_t; j--) {
            q[j + 1] = q[j];
            for (k = 0; k < m; k++) 
                u[k][j + 1] = u[k][j];
            for (k = 0; k <  n; k++) 
                vt[j + 1][k] = vt[j][k];
        }
        q[j + 1] = q_t;

        for (k = 0; k < m; k++)
            u[k][j + 1] = u_t[k];

        for (k = 0; k < n; k++)
            vt[j + 1][k] = vt_t[k];
    }
}
