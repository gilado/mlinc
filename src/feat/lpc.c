/* Copyright (c) 2023-2024 Gilad Odinak */
/* Linear Prediction Coefficients (LPC) functions */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "random.h"
#include "lpc.h"

static int lpc(
    const float *x, /* windowed input signal */
    int n,          /* # samples in x */
    int order,      /* predictor order required */
    double *lpcc,   /* returned predictor coefficients */
    double *pe      /* returned predictor error */
    )               /* returns 0=OK, 1=zero power, 2=fail */
{
    double  r[order + 1];   /* autocorrelations */
    double  pc[order + 1];  /* predictor coefficients */
    double  ai, aj, akk;    /* temporary coefficient values */
    double  sum;            /* accumulator */
    double  err;            /* predicator error */    
    int     i, k;           /* loop counters */

    if (order < 1)
        return 2;
        
    /* compute autocorrelations */
    for (i = 0; i <= order; i++) {
        sum = 0;
        for (k = 0; k < n - i; k++)
            sum += x[k] * x[k + i];
        r[i] = sum;
    }

    /* compute predictor coefficients */
    if (r[0] == 0) /* no signal ! */
        return 1;
    err = r[0];
    pc[0] = 1.0;
    /* levinson-durbin */
    for (k = 1; k <= order; k++) {
        sum = 0;
        for (i = 1; i <= k; i++)
            sum -= pc[k - i] * r[i];
        akk = sum / (err);
        /* new predictor coefficients */
        pc[k] = akk;
        for (i = 1; i <= (k / 2); i++) {
            ai = pc[i];
            aj = pc[k-i];
            pc[i] = ai + akk * aj;
            pc[k-i] = aj + akk * ai;
        }
        /* new prediction error */
        err = err * (1.0 - akk*akk);
        if (err <= 0) /* negative/zero error ! */
            return 2;
    }
    for (i = 0; i <= order; i++)
        lpcc[i] = pc[i];
    *pe = err;
    return 0;
}

float computeLPC(const float* samples, int numSamples, int order, double *lpcc)
{
    double error = 0.0;
    for (int i = 0; i <= order; i++)
        lpcc[i] = 0.0;
    lpc(samples,numSamples,order,lpcc,&error);
    return (float) error;
}

static void lpc2samples(const double *lpcc, int order, float sigma,
                                      float *output_signal, int signal_length)
{
    sigma = (sigma > 1e-13) ? sigma : 0;
    sigma = sqrt(sigma / signal_length);
    for (int m = 0; m < signal_length && m < order; m++)
        output_signal[m] = 0.0;
    for (int m = order; m < signal_length; m++) {
        double sample = sigma * nrand(0,1);
        for (int n = 1; n <= order; n++)
            sample -= lpcc[n] * output_signal[m - n];
        output_signal[m] = (float) sample;
    }
}


void LPCsynthesis(const double* lpcc, int order, float sigma,
                                               int num_samples, float* samples)
{
    lpc2samples(lpcc,order,sigma,samples,num_samples);
}
