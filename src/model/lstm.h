/* Copyright (c) 2023-2024 Gilad Odinak */
/* LSTM (reccurent) neural network  layer data structures and functions */
#ifndef LSTM_H
#define LSTM_H
#include "array.h"
#include "activation.h"

typedef struct lstm_s {
  int D;           /* Input vector dimension (including bias)       */
  int S;           /* Number of units, size of hidden state         */
  int B;           /* Number of input vectors in a batch            */
  char activation; /* n(one) s(igmoid) r(elu) (S)oftmax             */
  int stateful;    /* 1: maintain state between batches             */
  fArr2D Wf;       /* Weights matrix [D][S]                         */
  fArr2D Wi;       /* Weights matrix [D][S]                         */
  fArr2D Wc;       /* Weights matrix [D][S]                         */
  fArr2D Wo;       /* Weights matrix [D][S]                         */
  fArr2D Uf;       /* Weights matrix [S][S]                         */
  fArr2D Ui;       /* Weights matrix [S][S]                         */
  fArr2D Uc;       /* Weights matrix [S][S]                         */
  fArr2D Uo;       /* Weights matrix [S][S]                         */
  fArr2D f;        /* Forget gate matrix [B][S]                     */
  fArr2D i;        /* Input  gate matrix [B][S]                     */
  fArr2D o;        /* Output gate matrix [B][S]                     */
  fArr2D cc;       /* Cell candidate matrix [B+1][S]                */
  fArr2D c;        /* Cell matrix [B+1][S]                          */
  fArr2D h;        /* Hidden state matrix [B+1][S]                  */
  fVec ph;         /* Previous batch last hidden state vector [S]   */
  fVec pc;         /* Previous batch last cell state vector [S]     */
} LSTM;

/* Creates a long short term memory (LSTM) neural network.
 *
 * Parameters:
 *   units      - Number of cells (hidden size)
 *   activation - String, can be one of "none", "sigmoid", "relu", or "Softmax"
 *   stateful   - If not zero, maintain state across batches.
 * 
 * Returns:
 *   Pointer to an LSTM neural network layer.
 *
 * Notes:
 *   - The neural network needs to be further intialized using lstm_init()
 *     before it can be used.
 */
LSTM* lstm_create(int units, char* activation, int stateful);

/* Initializes an LSTM neural network created by lstm_create().
 *
 * Parameters:
 *   input_dim  - Size of input vectors (must include bias dimension)
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   - Kernel weights (Wx) are initialized using Glorot normal distribution.
 *   - Recurrent weights (Ux) are initialized using orthogonal uniform 
 *     distribution.
 */
void lstm_init(LSTM* l, int input_dim, int batch_size);

/* Sets a new batch size.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 *
 * Notes:
 *   If this function is called before ilstm_init(), it does nothing.
 *   Otherwise, the network's hidden states are resized and re-initialized
 */
void lstm_set_batch_size(LSTM* l, int batch_size);

/* Frees the memory allocated by lstm_create() / lstm_init().
 * 
 * Parameters:
 *   l - Pointer to the LSTM neural network layer to be freed
 */
void lstm_free(LSTM* l);

/* Resets the LSTM internal state.
 * 
 * Parameters:
 *   l - Pointer to the LSTM neural network layer to be reset
 */
void lstm_reset(LSTM* l);

/* Performs LSTM layer training/prediction's forward pass.
 *
 * Parameters:
 *   l   - Pointer to the LSTM layer's data
 *   X   - Array of input vectors BxD, where B is the number of 
 *         input vectors, and D is the number of features in each vector
 *   lyr - Ordinal number of this layer in a model (not used)
 * 
 * Returns:
 *   Pointer to the predicted values.
 * 
 * Note:
 *   In a multi-layered neural network, after the first layer,
 *   X is the (activated) output of a previous layer.
 */
static inline fArr2D lstm_forward(LSTM* restrict l, 
                                  const fArr2D restrict X /*[B][D]*/, int lyr)
{
    (void) lyr;
    inline void activate(fVec v, int S, char activation)
    {
        switch (activation) {
           case 's' : sigmoid((fArr2D)v,1,S); break;
           case 'r' : relu((fArr2D)v,1,S); break;
           case 'S' : softmax((fArr2D)v,1,S); break;
        }
    }
    const int D = l->D;
    const int S = l->S;
    const int B = l->B;
    typedef float (*ArrTD)[D];
    ArrTD x = (ArrTD) X; 
    /* Clear state. Note that for arrays with B+1 rows row t is the state
     * at time step t-1. Addvance the local pointers accordingly.
     */
    fltclr(l->f,B*S);
    fltclr(l->i,B*S);
    fltclr(l->o,B*S);
    fltclr(l->cc,(B+1)*S);
    fltclr(l->c,(B+1)*S);
    fltclr(l->h,(B+1)*S);
    typedef float (*ArrBS)[S];
    ArrBS f = (ArrBS) l->f;
    ArrBS i = (ArrBS) l->i;
    ArrBS o = (ArrBS) l->o;
    typedef float (*ArrB1S)[S];
    ArrBS cc = ((ArrB1S) l->cc) + 1; /* cc[-1] -> l->cc[0] */
    ArrBS c = ((ArrB1S) l->c) + 1;   /* c[-1]  -> l->c[0]  */
    ArrBS h = ((ArrB1S) l->h) + 1;   /* h[-1]  -> l->h[0]  */
    /* Set state to value from previous batch 
     * ph - Vector 1xS containing the hidden state at the last time step 
     *      of the previous batch of data of this layer
     * pc - Vector 1xS containing the cell state at the last time step 
     *      of the previous batch of data of this layer
     */
    if (l->stateful) {
        fltcpy(h[-1],l->ph,S);
        fltcpy(c[-1],l->pc,S);
    }
    else {
        fltclr(h[-1],S);
        fltclr(c[-1],S);
    }

    for (int t = 0; t < B; t++) {
        /* f[t] = activate(X[t] @ Wf + h[t-1] * Uf) */
        addvecmatmul(f[t],x[t],l->Wf,D,S);    
        addvecmatmul(f[t],h[t-1],l->Uf,S,S);
        activate(f[t],S,l->activation);
        /* i[t] = activate(X[t] @ Wi + h[t-1] * Ui) */
        addvecmatmul(i[t],x[t],l->Wi,D,S);    
        addvecmatmul(i[t],h[t-1],l->Ui,S,S);
        activate(i[t],S,l->activation);
        /* o[t] = activate(X[t] @ Wo + h[t-1] * Uo) */
        addvecmatmul(o[t],x[t],l->Wo,D,S);    
        addvecmatmul(o[t],h[t-1],l->Uo,S,S);
        activate(o[t],S,l->activation);
        /* cc[t] = tanh(X[t] @ Wc + h[t-1] @ Uc) */
        addvecmatmul(cc[t],x[t],l->Wc,D,S);    
        addvecmatmul(cc[t],h[t-1],l->Uc,S,S);
        for (int j = 0; j < S; j++)
            cc[t][j] = tanh(cc[t][j]);
        /* c[t] = f[t] * c[t-1] + i[t] * cc[t] */ /* REVIEW - not cc[t-1] ? */
        for (int j = 0; j < S; j++)
            c[t][j] = f[t][j] * c[t-1][j] + i[t][j] * cc[t][j];
        /* h[t] = o[t] * tanh(c[t])  */
        for (int i = 0; i < l->S; i++)
            h[t][i] = o[t][i] * tanh(c[t][i]);
    }
    /* Save last time step cell and hidden state for next batch of data */
    fltcpy(l->ph,h[B-1],S); 
    fltcpy(l->pc,c[B-1],S); 
    return h;
}

/* Performs LSTM layer training's backward pass.
 * 
 * Parameters:
 *   l    - Pointer to the LSTM layer's data
 *   dY   - Output vector gradient of lstm_create's units dimension
 *   X    - Array of input vectors BxD, where B is the number of input
 *          vectors, and D is the number of features in each vector
 *   g    - Array of 8 gradient matrices Wf Wi Wc Wo Uf Ui Uc Uo of the 
 *          same dimensions as their corresponding weight matrices
 *   activation - String, can be one of "none", "sigmoid", or "relu"
 *   dX   - Output parameter for the input vector gradient (if not NULL)
 *   lyr  - Ordinal number of this layer in a model (not used)
 * 
 * Returns:
 *   None
 * 
 * Note:
 *   - Calculates the weight matrices gradients with respect to the weights 
 *     and adds them to the matrices in g.
 *   - Calculates the input vector gradient and returns it in dx, if dx is 
 *     not NULL
 *   - Calculates and returns nh and nc
 *   - In a multi-layered neural network, except the last layer, dy is the 
 *     gradient of the previous layer's input (dx), thus, the dimension of 
 *     dx (this layer's D) must equal the dimension of the previous layer's
 *     dy (previous layer's S)
 */
static inline void lstm_backward(LSTM* restrict l,
                                 const fArr2D restrict dY/*[B][S]*/, 
                                 const fArr2D restrict X/*[B][D]*/,
                                 fArr2D* g/*Gradient matrices*/,
                                 fArr2D restrict dX/*[B][D]*/,
                                 int lyr)
{
    (void) lyr;
    inline float d_activate(float x, char activation)
    {
        switch (activation) {
           case 's' : return d_sigmoid_1(x);
           case 'r' : return d_relu_1(x);
           /* REVIEW: applying d_softmax() degrades convergence - why? */
           /* case 'S' : return d_softmax_1(x); // y = x * (yt - x); break; */
        }
        return x;
    }
    const int D = l->D;
    const int S = l->S;
    const int B = l->B;
    typedef float (*ArrBD)[D];
    ArrBD x = (ArrBD) X; 
    ArrBD dx = (ArrBD) dX;
    typedef float (*ArrBS)[S];
    ArrBS dy = (ArrBS) dY;
    /* Layer's state */
    ArrBS f = (ArrBS) l->f;
    ArrBS i = (ArrBS) l->i;
    ArrBS o = (ArrBS) l->o;
    typedef float (*ArrB1S)[S];
    ArrBS cc = ((ArrB1S) l->cc)+ 1;
    ArrBS c = ((ArrB1S) l->c) + 1;
    ArrBS h = ((ArrB1S) l->h) + 1;
    /* Layer's gradients */
    typedef float (*ArrDS)[S];
    ArrDS gWf = (ArrDS) g[0];
    ArrDS gWi = (ArrDS) g[1];
    ArrDS gWc = (ArrDS) g[2];
    ArrDS gWo = (ArrDS) g[3];
    for (int i = 0; i < 4; i++)
        fltclr(g[i],D * S);
    typedef float (*ArrS2)[S];
    ArrS2 gUf = (ArrS2) g[4];
    ArrS2 gUi = (ArrS2) g[5];
    ArrS2 gUc = (ArrS2) g[6];
    ArrS2 gUo = (ArrS2) g[7];
    for (int i = 4; i < 8; i++)
        fltclr(g[i],S * S);
    /* Future time step gradient */
    float dh_next[S];
    float dc_next[S];
    fltclr(dh_next,S);
    fltclr(dc_next,S);
    /* Backward pass loop */
    for (int t = B - 1; t >= 0; t--) {
        /* Calculate the gradient loss with respect to the hidden state */
        /* dh = dy[t] + dh_next */
        float dh[S];
        for (int j = 0; j < S; j++)
            dh[j] = dy[t][j] + dh_next[j];

        /* Update output gate gradient */
        float do_[S]; /* 'do' is a C keyword, use do_ for variable name */
        for (int j = 0; j < S; j++)
            do_[j] = dh[j] * tanh(c[t][j]) * d_activate(o[t][j],l->activation);
        addoutermul(gWo,x[t],do_,D,S);
        addoutermul(gUo,h[t-1],do_,S,S);
        /* Update cell state gradient */
        /* dc = dh * o[t] * tanh_derivative(c[t]) + dc_next */
        float dc[S];
        for (int j = 0; j < S; j++)
            dc[j] = dh[j] * o[t][j] * d_tanh(c[t][j]) + dc_next[j];

        /* Notice cc[t] already is activated (i.e. tanh applied) 
         * in forward so instead of d_tanh use d_tanh_x 
         * dcc = dc * i[t] * tanh_x_derivative(cc[t]) 
         */
        float dcc[S];
        for (int j = 0; j < S; j++)
            dcc[j] = dc[j] * i[t][j] * d_tanh_x(cc[t][j]); /* REVIEW - not cc[t-1] */
        addoutermul(gWc,x[t],dcc,D,S);
        addoutermul(gUc,h[t-1],dcc,S,S);

        /* Update input gate gradient */
        float di[S];
        for (int j = 0; j < S; j++)
            di[j] = dc[j] * cc[t][j] * d_activate(i[t][j],l->activation); /* REVIEW - not cc[t-1] */
        addoutermul(gWi,x[t],di,D,S);
        addoutermul(gUi,h[t-1],di,S,S);

        /* Update forget gate gradient */
        float df[S];
        for (int j = 0; j < S; j++)
            df[j] = dc[j] * c[t-1][j] * d_activate(f[t][j],l->activation);
        addoutermul(gWf,x[t],df,D,S);
        addoutermul(gUf,h[t-1],df,S,S); 
        
        /* Compute gradients for the previous layer */
        fltclr(dh_next,S);
        addinnermul(dh_next,df,l->Uf,S,S);
        addinnermul(dh_next,di,l->Ui,S,S);
        addinnermul(dh_next,dcc,l->Uc,S,S);
        addinnermul(dh_next,do_,l->Uo,S,S);
        for (int j = 0; j < S; j++)
            dc_next[j] = f[t][j] * dc[j];
        if (dx != NULL) {
            fltclr(dx[t],D);
            addinnermul(dx[t],df,l->Wf,D,S);
            addinnermul(dx[t],di,l->Wi,D,S);
            addinnermul(dx[t],dcc,l->Wc,D,S);
            addinnermul(dx[t],do_,l->Wo,D,S);
        }
    }
    /* Save last time step cell and hidden state for next batch of data */
    fltcpy(l->ph,h[B-1],S); 
    fltcpy(l->pc,c[B-1],S); 
}
#endif
