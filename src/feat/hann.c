/* Copyright (c) 2023-2024 Gilad Odinak */
#include "hann.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int hannWindowInit(HANNWIN* hannWin, int winSize)
{
    if (winSize < 2 || winSize > MAX_WINDOW_SIZE)
        return -1;
    if (winSize % 2 != 0)
        return -1;
    hannWin->winSize = winSize;
    for (int n = 0; n < winSize / 2; n++)
        hannWin->coeff[n] = 0.5 * (1 - cos((2 * M_PI * n)/(winSize - 1)));
    return 0;
}

int hannWindow(HANNWIN* hannWin, const float* inData, float* outData, int winSize)
{
    if (hannWin->winSize != winSize)
        return -1;
    const double *coeff = hannWin->coeff;
    for (int i = 0; i < winSize / 2; i++)
        outData[i] = (float) (inData[i] * coeff[i]);
    for (int i = winSize / 2; i < winSize; i++)
        outData[i] = (float) (inData[i] * coeff[winSize - i - 1]);
    return 0;
}
