/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __linux__
#include <cblas.h>
#elif defined __APPLE__
#include <Accelerate/Accelerate.h>
#endif

double nrand(double mean, double stddev) 
{
    double u1 = rand() / ((double)RAND_MAX + 1);
    double u2 = rand() / ((double)RAND_MAX + 1);
    // Box-Muller transform
    double z = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
    return mean + stddev * z; // Shift and scale
}

#define ASIZE 1024

double a[ASIZE][ASIZE];
double b[ASIZE][ASIZE];
double c[ASIZE][ASIZE];


void main (int argc, char **argv) 
{
    int cblas = 0;
    int iter = 0;
    if (argc == 1) {
        fprintf(stderr,"syntax: cblastest [-cblas] [iterations]\n");
        return;
    }
    if (argc >= 2) {
        if (argc >= 3)
            iter = atoi(argv[2]);      
        if (strcmp(argv[1],"-cblas") == 0)
            cblas = 1;
        else
            iter = atoi(argv[1]);
    }
    printf("cblas %s iterations %d\n",(cblas)?"true":"false",iter);        
            
    for (int i = 0; i < ASIZE; i++) {
        for (int j = 0; j < ASIZE; j++) {
            a[i][j] = nrand(0.0,0.8);
            b[i][j] = nrand(0.0,0.4);
        }
    }    

    for (int i = 0; i < iter; i++) {
        if (cblas) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ASIZE, ASIZE, ASIZE,
                1.0, (const double *) a, ASIZE,
                (const double *) b, ASIZE,
                1.0, (double *) c, ASIZE);
        }
        else {    
            for (int i = 0; i < ASIZE; i++) {
                for (int j = 0; j < ASIZE; j++) {
                    for (int k = 0; k < ASIZE; k++) {
                        c[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
        }
    }        
}
