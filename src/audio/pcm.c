/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdint.h>
#include "pcm.h"

void pcm2flt(const int16_t* pcmData, float* floatData, int numSamples)
{
    const float scale = (float)INT16_MAX;

    for (int i = 0; i < numSamples; ++i)
        floatData[i] = (float)pcmData[i] / scale;
}

void flt2pcm(const float* floatData, int16_t* pcmData, int numSamples)
{
    const float scale = (float)INT16_MAX + 2.0;

    for (int i = 0; i < numSamples; ++i) {
        float s = (floatData[i] * scale);
        if (s > 1.0) s = 1.0; else if (s < -1.0) s = -1.0;
        pcmData[i] = (int16_t) s;
    }
}
