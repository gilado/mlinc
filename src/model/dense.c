/* Copyright (c) 2023-2024 Gilad Odinak */
/* Dense (feed forward) neural network functions */
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include "mem.h"
#include "array.h"
#include "random.h"
#include "dense.h"

/* Creates a feed forward neural network.
 *
 * Parameters:
 *   units      - Number of cells (hidden size)
 *   activation - String, can be one of "none", "sigmoid", "relu", or "Softmax"
 *
 * Returns:
 *   Pointer to a dense neural network layer.
 *
 * Notes:
 *   - The neural network needs to be further intialized using dense_init()
 *     before it can be used.
 */
DENSE* dense_create(int units, char* activation)
{
    DENSE* l = allocmem(1,1,DENSE);
    l->S = units;
    if (!strcasecmp("none",activation)) l->activation = 'n';
    if (!strcasecmp("sigmoid",activation)) l->activation = 's';
    if (!strcasecmp("relu",activation)) l->activation = 'r';
    if (!strcasecmp("softmax",activation)) l->activation = 'S';
    if (l->activation == 0) {
        freemem(l);
        fflush(stdout);
        fprintf(stderr,"dense_create: invalid activation '%s'\n",activation);
        exit(-1);
    }
    return l;
}

/* Initializes a feed forward neural network created by dense_create().
 *
 *   input_dim  - Size of input vectors (must include bias dimension)
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   - The layer's weights are initialized using glorot normal distribution 
 */
void dense_init(DENSE* l, int input_dim, int batch_size)
{
    l->D = input_dim;
    l->B = batch_size;
    l->Wx = allocmem(l->D,l->S,float);
    l->h = allocmem(l->B,l->S,float);

    typedef float (*ArrDS)[l->S];
    ArrDS Wx = (ArrDS) l->Wx;
    float scale = sqrt(2.0 / (l->D + l->S));
    for (int i = 0; i < l->D; i++)
        for (int j = 0; j < l->S; j++)
            Wx[i][j] = nrand(0.0,scale);
}

/* Sets a new batch size.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   If this function is called before dense_init(), it does nothing.
 *   The network's hidden state is resized and re-initialized
 */
void dense_set_batch_size(DENSE* l, int batch_size)
{
    if (l->B == 0)
        return;
    if (batch_size != l->B) {
        freemem(l->h);
        l->B = batch_size;
        l->h = allocmem(l->B,l->S,float);
    }
    else
        fltclr(l->h,l->B * l->S);
}

/* Frees the memory allocated by dense_create() / dense_init()
 * 
 * Parameters:
 *   l - Pointer to the neural network to be freed
 */
void dense_free(DENSE* l)
{
    freemem(l->h);
    freemem(l->Wx);
    freemem(l);
}

/* Resets the network hidden state.
 * 
 * Parameters:
 *   l - Pointer to the DENSE neural network layer to be reset
 */
void dense_reset(DENSE* l)
{
    (void) l; /* Do nothing */
}

