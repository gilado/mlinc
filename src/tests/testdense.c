/* Copyright (c) 2023-2024 Gilad Odinak */
/* Simple test program for the Dense layer implementation */
#include <stdio.h>
#include <math.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "loss.h"
#include "dense.h"

static void dense_update_lin(fArr2D Wx_,fArr2D gWx_,float lr,int D, int S);

/* Trains a Multi Layer Perceptronn to predict 
 * the values of f(x) = (x**2 + 10* sin(x))
 *
 * layers array contains the size of each layer. One additional output layer
 * is implied.  So for example if layers contains two elements, 32 and 16,
 * three layers will be created with size of 32, 16, and 1.
 *
 * range[0] - lowest input (x) value
 * range[1] - highest input (x) value (exclusive)
 * range[2] - increment between x values
 */
int test_dense(const float range[3], const int layers[], int layers_cnt,
                                               float learning_rate, int epochs)
{    
    float f(float x) { return (pow(x,2) + 10.0 * sin(x)); }
    char* title = "f(x) = (x**2 + 10* sin(x))";
    const int L = layers_cnt + 1;
    const int M = (int) ((range[1] - range[0]) / range[2] + 0.5);
    printf("%d layers (including output layer), %d input samples\n",L,M);
    const int D = 2;  /* Input vector dimension (including bisas)   */
    const int N = 1;  /* Output vector dimension                    */
    float X[M][D];    /* X[][0] is x values, X[][1] is bias == 1.0  */
    float yt[M][N];   /* True labels vector yt = f(X)               */
    float y[M][N];    /* Output prediction (single dimension)       */
    float x = range[0];
    /* Initialize data */
    for (int i = 0; i < M ; i++) {
        X[i][0] = x;
        X[i][1] = 1.0;
        yt[i][0] = f(x);
        x += range[2];
    }
    /* Create layers */
    DENSE* l[L];
    for (int j = 0; j < L - 1; j++)
        l[j] = dense_create(layers[j],"relu");
    l[L - 1] = dense_create(N,"none");

    /* Initialize layers */
    dense_init(l[0],D,M);
    for (int j = 1; j < L; j++)
        dense_init(l[j],layers[j - 1],M);

    /* Allocate memory for gradients */
    fArr2D dy[L];  /* Gradients with respect to the inputs  */
    fArr2D gWx[L]; /* Gradients with respect to the weights */
    for (int j = 0; j < L; j++) {
        dy[j] = allocmem(l[j]->B,l[j]->S,float);
        gWx[j] = allocmem(l[j]->D,l[j]->S,float);
    }

    float losses[epochs];
    
    for (int i = 0; i < epochs; i++) {
        fArr2D yp[L]; /* pointers to layers' prediction arrays */
        /* Forward pass */
        yp[0] = dense_forward(l[0],X,0);
        for (int j = 1; j < L; j++)
            yp[j] = dense_forward(l[j],yp[j - 1],j);
        fltcpy(y,yp[L - 1],M * N); /* Save final forward pass result */
        float loss = mean_square_error(y,yt,M,N);
        losses[i] = loss;
        printf("epoch %5d loss %10.3f\r",i+1,loss<999999?loss:999999);
        fflush(stdout);
        /* Backward pass */
        dLdy_mean_square_error(y,yt,dy[L - 1],M,N);
        for (int j = L - 1; j > 0; j--) {
            dense_backward(l[j],dy[j],yp[j - 1],
                           gWx[j],l[j]->activation,dy[j - 1],0);
        }
        dense_backward(l[0],dy[0],X,gWx[0],l[0]->activation,NULL,0);
        /* Update weights */
        for (int j = 0; j < L; j++)
            dense_update_lin(l[j]->Wx,gWx[j],learning_rate,l[j]->D,l[j]->S);
    }
    printf("\n");
    printf("X:  ");
    for (int i = 0; i < M; i++)
        printf("%6.1f ",X[i][0]);
    printf("\nyt: ");
    for (int i = 0; i < M; i++)
        printf("%6.1f ",yt[i][0]);
    printf("\ny:  ");
    for (int i = 0; i < M; i++)
        printf("%6.1f ",y[i][0]);
    printf("\n");
    for (int i = 0; i < L; i++) {
        dense_free(l[i]);
        freemem(dy[i]);
        freemem(gWx[i]);
    }
#ifdef HAS_PLOT
    {
        #include "../plot/plot.h"
        float x[M];
        for (int i = 0; i < M; i++)
            x[i] = X[i][0];
        plot_graph(x,(float*)y,(float*)yt,M,
                   epochs,losses,NULL,NULL,NULL,title);
    }
#else
    (void) losses;
    (void) title;
#endif
    return 0;
}

/* Updates dense layer's weights in a linear way: 
 * weight = weight - learning_rate * weight_gradient
 * Wx:  weight matrix DxS
 * gWx: gradient matrix DxS
 * lr:  learning_rate
 */
static void dense_update_lin(fArr2D Wx_/*[D][S]*/,
                             fArr2D gWx_/*[D][S]*/,
                             float lr,
                             int D, int S)
{
    typedef float (*ArrDS)[S];
    ArrDS Wx = (ArrDS) Wx_;
    ArrDS gWx = (ArrDS) gWx_;
    for (int i = 0; i < D; i++)
        for (int j = 0; j < S; j++)
            Wx[i][j] -= lr * gWx[i][j];
}

int main()
{
    init_lrng(42);
    const int layers[3] = {64,128,16};
    const float range[3] = {0.0,5.0,0.1};
    test_dense(range,layers,3,0.0002,60000);

    return 0;
}

