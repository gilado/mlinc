/* Copyright (c) 2023-2024 Gilad Odinak */
#define MAX_FILTER_ORDER 4
typedef struct {
    int order;        // 1 to 4
    char type;        // "h"[ighpass] or "l"[owpass]
    int sampleRate;   // >= 16
    int cutoffFreq;   // 2 to (samplerate / 2 - 1)
    double aCoeff[MAX_FILTER_ORDER + 1];
    double bCoeff[MAX_FILTER_ORDER + 1];
    double xPrev[MAX_FILTER_ORDER + 1];
    double yPrev[MAX_FILTER_ORDER + 1];
} FILTER;

int initFilter(FILTER* restrict  filter, 
               int order, const char *type, int sampleRate, int cutoffFreq);

void runFiler(FILTER* restrict filter, 
              const float *inSamples, float *outSamples, int numSamples);
