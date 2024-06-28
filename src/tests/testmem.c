/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include "mem.h"

int main() {
    int m, n;
    printf("\n");
    m = 100; n = 20;
    printf("allocating %d x %d matrix of ints (%lu bytes) ... ",m,n,m*n*sizeof(int));
          fflush(stdout);
    float (*a)[n] = (float (*)[n]) allocmem(m,n,int);
    printf("freeing allocated memory\n\n");
    freemem(a);
    m = 100; n = 20;
    printf("allocating %d x %d matrix of floats (%lu bytes) ... ",m,n,m*n*sizeof(float));
          fflush(stdout);
    float (*b)[n] = (float (*)[n]) allocmem(m,n,float);
    printf("freeing allocated memory\n\n");
    freemem(b);
    printf("allocating and freeing memory of increased size until failure\n\n");
    m = 1000000; n = 1000;
    for (;;) {
          printf("allocating %d x %d matrix of floats (%.0f GiB) ... ",
                 m,n,(float) m * n * sizeof(float) / (1000.0 * 1000 * 1000));
          fflush(stdout);
          float (*c)[n] = (float (*)[n]) allocmem(m,n,float);
          printf("freeing allocated memory\n");
          freemem(c);
          m *= 2; n *= 2;
    }
    return 0;
}
