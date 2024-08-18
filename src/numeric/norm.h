/* Copyright (c) 2023-2024 Gilad Odinak */
/* Vector and Array mormalization function */
#ifndef NORM_H
#define NORM_H
#include <math.h>
#include "array.h"

/* Calculates and returns the norm of the vector x.
 */
static inline float vecnorm(const fVec restrict x_/*[N]*/, int N)
{
    typedef float (*VecN);
    const VecN x = (const VecN) x_; 
    float sum = 0;
    for (int i = 0; i < N; i++) 
        sum += x[i] * x[i];
    return sqrt(sum);
}

/* Calculates and returns the norm of the matrix m.
 */
static inline float matnorm(const fArr2D restrict m_/*[M][N]*/, int M, int N)
{
    typedef float (*ArrMN)[N];
    const ArrMN m = (const ArrMN) m_; 
    float sum = 0;
    for (int i = 0; i < M; i++) 
        for (int j = 0; j < N; j++) 
        sum += m[i][j] * m[i][j];
    return sqrt(sum);
}

#endif
