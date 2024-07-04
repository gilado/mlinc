/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "float.h"
#include "filter.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define arrclr(arr,cnt) { for (int i = 0; i < (cnt); i++) (arr)[i] = 0.0; }

int initFilter(FILTER* filter, int order, const char *type, 
                                               int sampleRate, int cutoffFreq)
{
    if (order < 1 || order > MAX_FILTER_ORDER)
        return -1;
    if (*type != 'l' && *type != 'h')
        return -1;
    if (sampleRate < 16)
        return -1;
    if (cutoffFreq < 2 || cutoffFreq * 2 >= sampleRate)
        return -1;

    int n = order;
    double cutoff = -((double) cutoffFreq) / ((double) sampleRate) * 2.0 * M_PI;
    double invert = (type[0] == 'l') ? 1.0 : -1.0;

    double yf0[5], yf1[5], xf[5];
    arrclr(yf0,MAX_FILTER_ORDER + 1); 
    arrclr(yf1,MAX_FILTER_ORDER + 1); 
    arrclr(xf,MAX_FILTER_ORDER + 1);
    yf0[0] = -1.0; yf1[0] = 0.0; xf[0] = 1.0;

    double scale = 1.0;
    for (int i = 1; i <= n; i++) {
        double angle = (((double) i) - 0.5) / ((double) n) * M_PI;
        double sin2 = 1.0 - sin(cutoff) * sin(angle);
        double rcof0 = cos(cutoff) / sin2;
        double rcof1 = sin(cutoff) * cos(angle) / sin2;
        yf0[i] = 0; yf1[i] = 0;
        for (int j = i; j > 0; j--) {
            yf0[j] += (rcof0 * yf0[j - 1] + rcof1 * yf1[j - 1]);
            yf1[j] += (rcof0 * yf1[j - 1] - rcof1 * yf0[j - 1]);
        }
        scale *= sin2 * 2.0 / (1.0 - cos(cutoff) * invert);
        xf[i] = xf[i - 1] * invert * ((double) (n - i + 1)) / ((double) i);
    }
    filter->order = order;
    filter->type = type[0];
    filter->sampleRate = sampleRate;
    filter->cutoffFreq = cutoffFreq;
    
    scale = sqrt(scale);
    for (int i = 0; i <= n; i++)
        filter->bCoeff[i] =xf[i] / scale;
    for (int i = 0; i <= n; i++)
        filter->aCoeff[i] = yf0[i] * ((i % 2) ? 1.0 : -1.0);

    arrclr(filter->xPrev,MAX_FILTER_ORDER + 1);
    arrclr(filter->yPrev,MAX_FILTER_ORDER + 1);
    return 0; 
}

void runFiler(FILTER* restrict filter, 
              const float *inSamples, float *outSamples, int numSamples)
{
    for (int i = 0; i < numSamples; i++) {
        for (int j = filter->order; j > 0; j--)
            filter->xPrev[j] = filter->xPrev[j - 1];
        filter->xPrev[0] = inSamples[i];
        for (int j = filter->order; j > 0; j--)
            filter->yPrev[j] = filter->yPrev[j - 1];
        double yPrev0 = filter->bCoeff[0] * filter->xPrev[0];
        for (int j = 1; j <= filter->order; j++) {
             yPrev0 += filter->bCoeff[j] * filter->xPrev[j] - 
                       filter->aCoeff[j] * filter->yPrev[j];
        }                         
        filter->yPrev[0] = yPrev0;
        outSamples[i] = (float) filter->yPrev[0];
    }
}

