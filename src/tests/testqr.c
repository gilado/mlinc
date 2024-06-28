/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <math.h>
#include "random.h"
#include "array.h"
#include "arrayio.h"
#include "norm.h"
#include "qr.h"

float A0[4][4] = {
    {0., 0., 0., 2.},
    {0.,-6.,-4.,-8.},
    {6., 6., 2., 5.},
    {0., 0.,-4.,-2.}
};
float Q0[4][4] = {
    {0., 0., 0., 1.},
    {0.,-1., 0., 0.},
    {1., 0., 0., 0.},
    {0., 0.,-1., 0.}
};
float R0[4][4] = {
    {6., 6., 2., 5.},
    {0., 6., 4., 8.},
    {0., 0., 4., 2.},
    {0., 0., 0., 2.}
};

float A1[3][4] = {
 { 0., 0., 0., 2.},
 { 0.,-6.,-4.,-8.},
 { 6., 6., 2., 5.}
};
float Q1[3][3] = {
 {-0., 0., 1.},
 { 0.,-1., 0.},
 {-1., 0., 0.}
};
float R1[3][4] = {
 {-6.,-6.,-2.,-5.},
 { 0., 6., 4., 8.},
 { 0., 0., 0., 2.}
};

float A2[2][3] = {
 { 1., 2., 3.},
 { 4., 5., 6.}
};
float Q2[2][2] = {
 {-0.24253563,-0.9701425},
 {-0.9701425 , 0.24253563}
};
float R2[2][3] = {
 {-4.12310563, -5.33578375, -6.54846188},
 { 0.        , -0.72760688, -1.45521375}
};

float A3[4][3] = {
 { 6.,  2.,  5.},
 { 0., -4., -8.},
 {-1.,  0.,  2.},
 { 2.,  2.,  7.}
};
float Q3[4][3] = {
 {-0.937043,-0.081035, 0.118048},
 { 0.      ,-0.949262,-0.283315},
 { 0.156174, 0.092611,-0.661069},
 {-0.312348, 0.289409,-0.684679}
};
float R3[3][3] = {
 {-6.403124,-2.498780,-6.559298},
 { 0.      , 4.213798, 9.400012},
 {-0.      , 0.      ,-3.258126},
};

int smoke_test(fArr2D A_/*[M][N]*/,
               fArr2D Q_/*[M][min(M,N)]*/,
               fArr2D R_/*[min(M,N)][N]*/,
               int M, int N)
{
    int D = (M < N) ? M : N;
    typedef float (*ArrMN)[N];
    typedef float (*ArrMD)[D];
    typedef float (*ArrDN)[N];
    ArrMN A = (ArrMN) A_;
    ArrMD Q = (ArrMD) Q_;
    ArrDN R = (ArrDN) R_;
    
    
    float Qm[M][D];
    float Rm[D][N];
    QR(A,Qm,Rm,M,N);
    float err = 0.0;
    
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D; j++)
            err += fabsf(fabsf(Qm[i][j]) - fabsf(Q[i][j]));
    for (int i = 0; i < D; i++) 
        for (int j = 0; j < N; j++)
            err += fabsf(fabsf(Rm[i][j]) - fabsf(R[i][j]));
            
    int ok = err < 1e-4;
    if (!ok) {
        printf("err: %e\n",err);
        print_array(A,M,N,"A","%12.8f",0);
        print_array(Q,M,D,"Q","%12.8f",0);
        print_array(Qm,M,D,"Qm","%12.8f",0);
        print_array(R,D,N,"R","%12.8f",0);
        print_array(Rm,D,N,"Rm","%12.8f",0);
    }
    
    return ok;
}


int full_test()
{
    int ok = 1;
    float eps = 1e-5;
    for (int t = 0; t < 100; t++) {
        const int M = (int) (urand(2,100) + 0.5);
        const int N = (int) (urand(2,100) + 0.5);
        const int D = (M < N) ? M : N;
        float A[M][N];
        float mean = urand(-3,3);
        float std = urand(0.01,3);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                A[i][j] = nrand(mean,std);
        float Q[M][D];
        float R[D][N];
        QR(A,Q,R,M,N);

        float Qt[D][M];
        transpose(Q,Qt,M,D);        
        float QQt[M][M];
        matmul(QQt,Q,Qt,M,D,M);
        int Q_ok = 1;
        if (M <= N) { /* Check that Q is orthogonal */
            /* Check that QQt is the identity matrix */
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    float QQt_e = fabsf(QQt[i][j]);
                    if (i == j && fabsf(QQt_e - 1.0f) > eps * M)
                        Q_ok = 0;
                    if (i != j && fabsf(QQt_e) > eps * M)
                        Q_ok = 0;
                }
            }
        }
        else { /* Check that thw columns of Q have length (norm) of 1 */
            /* Check that the rows of Qt have length (norm) of 1 */
            for (int i = 0; i < D; i++) {
                float rn = vecnorm(Qt[i],M);
                rn = fabsf(rn);
                if (fabsf(rn - 1.0f) > eps * M)
                    Q_ok = 0;
            }
        }
        float Ar[M][N];
        matmul(Ar,Q,R,M,D,N);
        /* Check that Ar = A = Q @ R */
        int QR_ok = 1;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double A_e = fabs(A[i][j]);
                double Ar_e = fabs(Ar[i][j]);
                if (fabs(Ar_e - A_e) > eps * M)
                    QR_ok = 0;
            }
        }
        if (!Q_ok || !QR_ok) {
            printf("test %d failed: Q is%s orthogonal, A %s Q @ R\n",
                   t + 1,Q_ok ? "" : " not",QR_ok ? "==" : "!=");
            ok = 0;
        }
    }
    return ok;
}

/* Note that full test may fail os some machines with avx instruction set
 * due to rounding errors. Recompiling with USEDOUBLE=Yes, or with -mno-avx
 * flags resolves that,
 */
int main()
{
    printf("smoke test 4 x 4 %s\n",smoke_test(A0,Q0,R0,4,4) ? "ok" : "failed");
    printf("smoke test 3 x 4 %s\n",smoke_test(A1,Q1,R1,3,4) ? "ok" : "failed");
    printf("smoke test 2 x 3 %s\n",smoke_test(A2,Q2,R2,2,3) ? "ok" : "failed");
    printf("smoke test 4 x 3 %s\n",smoke_test(A3,Q3,R3,4,3) ? "ok" : "failed");
    printf("full test %s\n",full_test() ? "ok" : "failed");
    return 0;
}

