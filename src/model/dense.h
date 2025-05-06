/* Copyright (c) 2023-2024 Gilad Odinak */
/* Dense (feed forward) neural network layer data structures and functions */
#ifndef DENSE_H
#define DENSE_H
#include "array.h"
#include "activation.h"

typedef struct dense_s {
  int D;           /* Input vector dimension (including bias)  */
  int S;           /* Number of units, size of hidden state    */
  int B;           /* Number of input vectors in a batch       */
  char activation; /* n(one) s(igmoid) r(elu) (S)oftmax        */
  fArr2D h;        /* Hidden State matrix [B][S]               */
  fArr2D Wx;       /* Weights matrix [D][S]                    */
} DENSE;

/* Creates a feed forward neural network.
 *
 * Parameters:
 *   units      - Number of cells (hidden size)
 *   activation - String, can be one of "none", "sigmoid", "relu", or "Softmax"
 *
 * Returns:
 *   Pointer to a dense neural network.
 *
 * Notes:
 *   - The neural network needs to be further intialized using dense_init()
 *     before it can be used.
 */
DENSE* dense_create(int units, char* activation);

/* Initializes a feed forward neural network created by dense_create().
 *
 * Parameters:
 *   input_dim  - Size of input vectors (must include bias dimension)
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   The network's weights are initialized using glorot normal distribution 
 */
void dense_init(DENSE* l, int input_dim, int batch_size);

/* Sets a new batch size.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   If this function is called before dense_init(), it does nothing.
 *   Otherwise, the network's hidden state is resized and re-initialized
 */
void dense_set_batch_size(DENSE* l, int batch_size);

/* Frees the memory allocated by dense_create() / dense_init()
 * 
 * Parameters:
 *   l - Pointer to the neural network to be freed
 */
void dense_free(DENSE* l);

/* Resets the network hidden state.
 * 
 * Parameters:
 *   l - Pointer to the DENSE neural network layer to be reset
 */
void dense_reset(DENSE* l);

/* Performs dense layer training/prediction's forward pass.
 *
 * Parameters:
 *   l   - Pointer to the dense layer's data
 *   X   - An array of input vectors (BxD dimensions, which includes bias)
 *   lyr - The ordinal number of this layer in a model (not used)
 *
 * Returns:
 *   Pointer to the predicted values.
 * 
 * Note that in a multi-layered neural network, after the first layer
 * X is the (activated) output of a previous layer.
 */
static inline fArr2D dense_forward(DENSE* restrict l, 
                                   const fArr2D restrict X/*[B][D]*/, int lyr)
{
    (void) lyr;
    /* h = X @ Wx */
    matmul(l->h,X,l->Wx,l->B,l->D,l->S);
    switch (l->activation) {
        case 's' : sigmoid(l->h,l->B,l->S); break;
        case 'r' : relu(l->h,l->B,l->S); break;
        case 'S' : softmax(l->h,l->B,l->S); break;
    }
    return l->h;
}

/* Performs dense layer training's backward pass.
 *
 * Parameters:
 *   l   - Pointer to the dense layer's data
 *   dy  - The output vector gradient of dense_create's units dimension
 *   X   - An input vector (D dimensions, which includes bias)
 *   lyr - The ordinal number of this layer in a model (not used)
 *
 * Calculates the weight matrix gradients with respect to the weights 
 * and adds them to the matrix gWx
 *
 * Calculates the input vector gradient and returns it in dx, if dx is not NULL
 *
 * Note that in a multi-layered neural network, except the last layer,
 * dy is the gradient of the previous layer's input (dx), thus, the dimension
 * of dx (this layer's D) must equal the dimension of the previous layer's
 * dy (previous layer's S).
 */
static inline void dense_backward(DENSE* restrict l, 
                                  const fArr2D restrict dy/*[B][S]*/, 
                                  const fArr2D restrict X/*[B][D]*/,
                                  fArr2D restrict gWx/*[D][S]*/,
                                  fArr2D restrict dx/*[B][D]*/,
                                  int lyr)
{
    (void) lyr;
    /* Gradient with respect to weights: gWx = X.T @ dy */
    Tmatmul(gWx,X,dy,l->D,l->B,l->S);
    if (dx != NULL) {
        /* Gradient with respect to inputs: 
         * dx = (dy @ Wx.T) * (gradient of activation (input))
         * The stored hidden state already is activated, and the gradient
         * functions below are adjusted accordingly (see loss.h)
         */
        /* dx = (dy @ Wx.T) */
        matmulT(dx,dy,l->Wx,l->B,l->S,l->D);
        switch (l->activation) {
            case 's' : d_sigmoid(dx,X,l->B,l->D); break;
            case 'r' : d_relu(dx,X,l->B,l->D); break;
            /* REVIEW: applying d_softmax() degrades convergence - why? */
            /* case 'S' : d_softmax(dx,X,yt,l->B,l->D); break;          */
        }
    }
}
#endif
