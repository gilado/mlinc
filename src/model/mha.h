/* Copyright (c) 2026 Gilad Odinak */
/* Multi-Head Attention layer data structures and functions */
/* Reference: Attention Is All You Need https://arxiv.org/pdf/1706.03762v7 */
#ifndef MHA_H
#define MHA_H
#include "float.h"
#include "array.h"
#include "activation.h"

typedef struct {
    /* dimensions */
    int B;      /* batch size */
    int T;      /* sequence length */
    int D;      /* model dim */
    int H;      /* heads */
    int Dh;     /* D / H */
    int BT;     /* B * T */

    fArr2D Wq;  /* [D][D] */
    fArr2D Wk;  /* [D][D] */
    fArr2D Wv;  /* [D][D] */
    fArr2D Wo;  /* [D][D] */

    fArr2D Q;   /* [BT][D] */
    fArr2D K;   /* [BT][D] */
    fArr2D V;   /* [BT][D] */

    fArr2D Qh;      /* [T][Dh] */
    fArr2D Kh;      /* [T][Dh] */
    fArr2D Vh;      /* [T][Dh] */

    fArr2D Scores;  /* [T][T] */
    fArr2D Att;     /* [T][T] */
    fArr2D Oh;      /* [T][Dh] */

    fArr2D Out;     /* [BT][D] */

    /* backward buffers */
    fArr2D dOut;    /* [BT][D] */

    fArr2D dQ;      /* [BT][D] */
    fArr2D dK;      /* [BT][D] */
    fArr2D dV;      /* [BT][D] */

    fArr2D dQh;     /* [T][Dh] */
    fArr2D dKh;     /* [T][Dh] */
    fArr2D dVh;     /* [T][Dh] */

    fArr2D dOh;     /* [T][Dh] */
    fArr2D dAtt;    /* [T][T] */
    fArr2D dScores; /* [T][T] */

    /* parameter gradients */
    fArr2D gWq;     /* [D][D] */
    fArr2D gWk;     /* [D][D] */
    fArr2D gWv;     /* [D][D] */
    fArr2D gWo;     /* [D][D] */

} MHA;

MHA* mha_create(int heads, int steps);

void mha_init(MHA* l, int input_dim, int batch_size, int training);

void mha_free(MHA* l);

/*
 * mha_forward - forward pass of Multi-Head Attention (MHA) layer
 *
 * This function computes the multi-head attention output for a batch
 * of sequences, including optional causal masking and padding masks.
 * It supports training-time backward buffers but does not modify them
 * in this function.
 *
 * Parameters:
 *   l         : Pointer to the MHA layer structure containing weights,
 *               buffers, and dimensions.
 *   X         : Input 2D array of shape [B*T][D], where:
 *               B = batch size, T = sequence length, D = model dimension.
 *   pad_mask  : Optional integer array of length [B*T];
 *               entries are 1 for real tokens, 0 for padding.
 *               If NULL, no padding mask is applied.
 *   Y         : Output 2D array of shape [B*T][D];
                 if NULL, output projection is skipped.
 *   mask      : Integer flag. If non-zero, causal (future) masking is
 *               applied to prevent attention to future positions in each
 *               sequence (used in decoder).
 *   lyr       : Layer index.
 *
 * Behavior:
 *   1. Projects input X into query (Q), key (K), and value (V) matrices
 *      using learned weights.
 *   2. Splits Q, K, V into H attention heads of dimension Dh = D/H.
 *   3. Computes scaled dot-product attention for each head:
 *      - Qh @ Kh^T / sqrt(Dh)
 *      - Applies causal mask if mask != 0
 *      - Applies column-wise padding mask if pad_mask is provided
 *      - Applies row-wise softmax to get attention weights
 *      - Multiplies attention weights with Vh
 *   4. Concatenates head outputs and applies the output projection
 *      Wo if Y != NULL.
 *
 * Note: Padding is not allowed at the beginning of a sequence; only
 *       trailing (right-aligned) padding is supported.
 *
 * References:
 *   - Vaswani et al., "Attention Is All You Need", 2017
 */
static inline void mha_forward(MHA* restrict l,
                               const fArr2D restrict X/*[BT][D]*/,
                               const iVec restrict pad_mask/*[BT]*/,
                               fArr2D Y/*[BT][D]*/,
                               int mask,
                               int lyr)
{
    (void) lyr;
    const int B = l->B;
    const int T = l->T;
    const int D = l->D;
    const int H = l->H;
    const int Dh = l->Dh;
    const int BT = l->BT;

    typedef float (*ArrDD)[D];
    typedef float (*ArrBTD)[D];
    typedef float (*ArrTDh)[Dh];
    typedef float (*ArrTT)[T];

    const ArrDD Wq = (ArrDD) l->Wq;
    const ArrDD Wk = (ArrDD) l->Wk;
    const ArrDD Wv = (ArrDD) l->Wv;
    const ArrDD Wo = (ArrDD) l->Wo;

    const ArrBTD Q = (ArrBTD) l->Q;
    const ArrBTD K = (ArrBTD) l->K;
    const ArrBTD V = (ArrBTD) l->V;

    const ArrTDh Qh = (ArrTDh) l->Qh;
    const ArrTDh Kh = (ArrTDh) l->Kh;
    const ArrTDh Vh = (ArrTDh) l->Vh;

    const ArrTT Scores = (ArrTT) l->Scores;
    const ArrTT Att = (ArrTT) l->Att;
    const ArrTDh Oh = (ArrTDh) l->Oh;

    const ArrBTD Out = (ArrBTD) l->Out;

    /* Linear projection of all heads - page 4. sec. 3.2.2 */
    matmul(Q, X, Wq, BT, D, D);
    matmul(K, X, Wk, BT, D, D);
    matmul(V, X, Wv, BT, D, D);

    fltclr(Out, BT * D);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {

            /* Split heads - page 5. sec. 3.2.2: definition of head */
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                fltcpy(&Qh[t][0], &Q[r][h*Dh], Dh);
                fltcpy(&Kh[t][0], &K[r][h*Dh], Dh);
                fltcpy(&Vh[t][0], &V[r][h*Dh], Dh);
            }

            /* Compute attention scores - Eq. (1), page 4: Q @ K.T */
            matmulT(Scores, Qh, Kh, T, Dh, T);

            /* Scale by sqrt(dk) (numerical stability) - Eq. (1) */
            float s = 1.0f / sqrtf((float)Dh);
            for(int i = 0; i < T; i++)
                for(int j = 0; j < T; j++)
                    Scores[i][j] *= s;

            if (mask) {
                /* Prevent attending to future positions - Fig. 2, page 4 */
                /* Also section 3.2.3, page 5 */
                for (int i = 0; i < T; i++)
                    for (int j = i + 1; j < T; j++)
                        Scores[i][j] = -1e9f;
            }

            if (pad_mask != NULL) {
                for (int i = 0; i < T; i++)
                    if (!pad_mask[b * T + i])
                        for (int j = 0; j < T; j++)
                            Scores[j][i] = -1e9f;
            }

            /* Compute attention probabilities - Softmax - Eq. (1) */
            softmax(Scores, T, T);

            /* Att and Scores could be same buffer, but this is clearer */
            fltcpy(Att, Scores, T * T);

            /* Eq. (1): Attention @ V */
            matmul(Oh, Att, Vh, T, T, Dh);

            /* Concatenate heads - page 5. sec. 3.2.2: definition of MultiHead() */
            for (int t = 0;t < T; t++) {
                int r= b * T + t;
                fltcpy(&Out[r][h * Dh], &Oh[t][0], Dh);
            }
        }
    }

    /* output projection - Eq. (2) */
    if (Y != NULL)
      matmul(Y, Out, Wo, BT, D, D);
}

static inline void mha_backward(MHA* restrict l,
                                fArr2D restrict dY /*[BT][D]*/,
                                const fArr2D restrict X/*[BT][D]*/,
                                fArr2D dX,
                                int lyr)
{
    (void) lyr;
    const int B = l->B;
    const int T = l->T;
    const int D = l->D;
    const int H = l->H;
    const int Dh = l->Dh;
    const int BT = l->BT;

    typedef float (*ArrBTD)[D];
    typedef float (*ArrTDh)[Dh];
    typedef float (*ArrTT)[T];
    typedef float (*ArrDD)[D];

    const ArrBTD dOut = (ArrBTD) l->dOut;

    const ArrBTD dQ = (ArrBTD) l->dQ;
    const ArrBTD dK = (ArrBTD) l->dK;
    const ArrBTD dV = (ArrBTD) l->dV;

    const ArrTDh dQh = (ArrTDh) l->dQh;
    const ArrTDh dKh = (ArrTDh) l->dKh;
    const ArrTDh dVh = (ArrTDh) l->dVh;

    const ArrTDh dOh = (ArrTDh) l->dOh;
    const ArrTT dAtt = (ArrTT) l->dAtt;
    const ArrTT dScores = (ArrTT) l->dScores;

    const ArrDD gWq = (ArrDD) l->gWq;
    const ArrDD gWk = (ArrDD) l->gWk;
    const ArrDD gWv = (ArrDD) l->gWv;
    const ArrDD gWo = (ArrDD) l->gWo;

    const ArrDD Wq = (ArrDD) l->Wq;
    const ArrDD Wk = (ArrDD) l->Wk;
    const ArrDD Wv = (ArrDD) l->Wv;
    const ArrDD Wo = (ArrDD) l->Wo;

    const ArrTDh Qh = (ArrTDh) l->Qh;
    const ArrTDh Kh = (ArrTDh) l->Kh;
    const ArrTDh Vh = (ArrTDh) l->Vh;

    const ArrTT Att = (ArrTT) l->Att;

    const ArrBTD Out = (ArrBTD) l->Out;

    fltclr(gWo, D * D);
    fltclr(dOut, BT * D);

    /* Y = Out Wo  →  Eq. (2) */
    Tmatmul(gWo, Out, dY, D, BT, D);
    matmulT(dOut, dY, Wo, BT, D, D);

    fltclr(dQ, BT * D);
    fltclr(dK, BT * D);
    fltclr(dV, BT * D);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {

            /* split dOut */
            for (int t = 0; t < T; t++) {
                int r = b * T + t;
                fltcpy(&dOh[t][0], &dOut[r][h * Dh], Dh);
            }

            /* Oh = Att @ Vh  - Eq. (1) */
            Tmatmul(dVh, Att, dOh, T, T, Dh);
            matmulT(dAtt, dOh, Vh, T, Dh, T);

            /* softmax backward - Eq. (1), Jacobian */
            /* dAtt and dScores could be same buffer, but this is clearer */
            softmax_backward(dScores, dAtt, Att, T, T);

            float s = 1.0f / sqrtf((float)Dh);
            for(int i = 0; i < T; i++)
              for(int j = 0; j < T; j++)
                dScores[i][j] *= s;

            /* Scores = Qh Khᵀ */
            matmul(dQh, dScores, Kh, T, T, Dh);
            matmulT(dKh, dScores, Qh, T, Dh, T);

            /* accumulate into full tensors */
            for (int a = h * Dh, t = 0; t < T; t++) {
                int r=b*T+t;
                for (int k = 0; k < Dh; k++)
                    dQ[r][a + k] += dQh[t][k];
                for (int k = 0; k < Dh; k++)
                    dK[r][a + k] += dKh[t][k];
                for (int k = 0; k < Dh; k++)
                    dV[r][a + k] += dVh[t][k];
            }
        }
    }

    /* Q = XWq, etc. - Eq. (3) */
    fltclr(gWq, D * D);
    fltclr(gWk, D * D);
    fltclr(gWv, D * D);

    Tmatmul(gWq, X, dQ, D, BT, D);
    Tmatmul(gWk, X, dK, D, BT, D);
    Tmatmul(gWv, X, dV, D, BT, D);

    if (dX) {
        matmulT(dX, dQ, Wq, BT, D, D);
        addMatmulT(dX, dK, Wk, BT, D, D);
        addMatmulT(dX, dV, Wv, BT, D, D);
    }
}

#endif
