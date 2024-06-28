/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to clip array values       */
#ifndef CLIP_H
#define CLIP_H

/* Clips the values of gradients array g, so their magnitude is never larger
 * than gmax, nor smaller than gmin. This prevents numerical instability due
 * to "exploding" gradients, and convergence stagnation because of "vanishing"
 * gradients. The down side is potential slower convergence, missing the true
 * local minima. However, in practice clipping leads to better accuracy.
 */

static inline void clip_gradients(fArr2D ga_/*[M][N]*/, 
                                  int M, int N, 
                                  float gmin, float gmax)
{
    typedef float (*ArrMN)[N];
    ArrMN ga = (ArrMN) ga_;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float g = ga[i][j];
            float m = fabsf(g);
            if (m > gmax)
                g = (g > 0) ? gmax : -gmax;
            else
            if (m < gmin)
                g = (g > 0) ? gmin : -gmin;
            ga[i][j] = g;
        }
    }
}

#endif
