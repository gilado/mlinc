/* Copyright (c) 2023-2024 Gilad Odinak */
/* Simple test program for adam optimizer with weight decay              */
#include <stdio.h>
#include <math.h>
#include "mem.h"
#include "array.h"
#include "adamw.h"


#define M 4
#define N 3

void test_adamw(float learning_rate, float weight_decay, float error_eps)
{
    // Target values
    const float t[M][N] = {
        {-0.92, 0.57,-0.31},
        { 0.24,-0.88, 0.65},
        { 0.09,-0.63, 0.72},
        { 0.81,-0.20, 0.46}
    };
 
    // Starting values
    const float s[M][N] = {
        {-0.114728, -0.061041, 0.106305},
        {0.210453, 0.207873, 0.089201},
        {-0.186953, 0.084362, -0.142827},
        {0.081038, 0.093246, 0.124387}
    };   
    
    float w[M][N];
    float g[M][N];
    float m[M][N];
    float v[M][N];

    fltcpy(w,s,M*N);
    fltclr(g,M*N);
    fltclr(m,M*N);
    fltclr(v,M*N);
    
    int update_step = 0;

    error_eps = fabsf(error_eps); 
    float error = 1.0 + error_eps;
    printf("learning_rate %g weight_decay %g error_eps %g \n",
           learning_rate,weight_decay,error_eps);
    while (error >= error_eps) {
        /* Calculate error */
        error = 0.0;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                error += pow(w[i][j] - t[i][j],2);
        error /= M*N;
        printf("    step %4d error %12.3e\r",update_step,error);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                g[i][j] = w[i][j] - t[i][j];
        update_step++;
        adamw_update(w,g,m,v,M,N,learning_rate,weight_decay,update_step);
    }
    printf("    converged in %d steps error %g\n",update_step,error);
}


int main()
{
    test_adamw(0.001,0.01,1e-6);
    test_adamw(0.01,0.01,1e-6);
    test_adamw(0.01,0.1,1e-6);
    test_adamw(0.1,0.1,1e-6);
    return 0;
}

