/* Copyright (c) 2023-2024 Gilad Odinak */
#include "zcr.h"

double zeroCrossings(const float* samples, int numSamples)
{
    int zeros = 0;
    for (int i = 1; i < numSamples; i++) {
        if ((samples[i - 1] >= 0 && samples[i] < 0) ||
            (samples[i - 1] < 0 && samples[i] >= 0)) {
            zeros++;
        }
    }
    return ((double) zeros) / numSamples;
}
