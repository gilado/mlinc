/* Copyright (c) 2023-2024 Gilad Odinak */
#ifndef HANN_H
#define HANN_H

#define MAX_WINDOW_SIZE 1024
typedef struct hannwin_s {
    int winSize; // Must be multiple of 2
    double coeff[MAX_WINDOW_SIZE/2];
} HANNWIN;

int hannWindowInit(HANNWIN* hannWin, int winSize);
int hannWindow(HANNWIN* hannWin, const float* inData, float* outData, int winSize);

#endif
