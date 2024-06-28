/* Copyright (c) 2023-2024 Gilad Odinak */
/* Test Principal Component Analysis    */
#include <stdio.h>
#include <math.h>
#include "mem.h"
#include "array.h"
#include "scaler.h"
#include "pca.h"
#include "irisfile.h"

int main(int argc, char** argv)
{
    char* irisfile = "data/iris/iris.csv";
    if (argc > 1)
        irisfile = argv[1];
    printf("Principal Component Analysis using the Iris Plants Database\n");
    printf("Using %s (to use another file run testpca <filepath>)\n",irisfile);

    const int nsamples = IRIS_SAMPLE_CNT;
    const int nfeatures = IRIS_FEAT_CNT;
    const int nclasses = IRIS_CLASS_CNT;
    const int ncomponents = 2;
    
    float x[nsamples][nfeatures];
    float r[nsamples][ncomponents];
    int y[nsamples];

    int ok = read_iris_file(irisfile,nsamples,x,y);
    if (!ok)
        return -1;

    SCALER* s = scaler_init(0,nfeatures,0);
    scaler_normalize(s,x,nsamples,1);
    scaler_free(s);
    PCA(x,r,nsamples,nfeatures,ncomponents);
#ifdef HAS_PLOT
    {
        #include "../plot/plot.h"
        plot_pca(r,y,nsamples,nclasses,iris_class_names,30.0,
                                             "PCA using SVD of Iris Dataset");
    }
#else
    (void) nclasses;
#endif
    printf("Done\n");
    return 0;
}


