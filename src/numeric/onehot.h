/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to encode and decode one-hot vectors */
#ifndef ONEHOT_H
#define ONEHOT_H

static inline void onehot_encode(const int* yc, fArr2D yv_, int M, int N)
{
    typedef float (*ArrMN)[N];
    ArrMN yv = (ArrMN) yv_;
    fltclr(yv,M * N);
    for (int i = 0; i < M; i++)
        yv[i][yc[i]] = 1.0;
}

static inline void onehot_decode(const fArr2D yv_, iVec yc, int M, int N) 
{
    typedef float (*ArrMN)[N];
    ArrMN yv = (ArrMN) yv_;
    for (int i = 0; i < M; i++) {
        int mj = 0;
        for (int j = 1; j < N; j++)
            if (yv[i][j] > yv[i][mj])
                mj = j;
        yc[i] = mj;
    }
}

#endif
