/* Copyright (c) 2026 Gilad Odinak */
/* Tests for the decoder-only transformer layer */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "transformer.h"

#define EPS  1e-3
#define TOL  5e-2

/* Loss is L = sum(dY * Y) */
static float transformer_loss(TRANSFORMER* l, 
                              float* X_flat, 
                              const float* dY_flat,
                              int BT, int D)
{
    typedef float (*ArrBTD)[D];
    ArrBTD X = (ArrBTD) X_flat;

    float Y[BT][D];
    fltclr(Y, BT * D);
    transformer_forward(l, (fArr2D) X, NULL, (fArr2D) Y, 0);

    float L = 0;
    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            L += dY_flat[i * D + j] * Y[i][j];
    return L;
}

/* Compute numerical gradient for a single weight element */
static inline float numerical_grad(float* w, TRANSFORMER* l, 
                                   float* X,
                                   const float* dY, 
                                   int BT, int D)
{
    float old = *w;
    *w = old + EPS;
    float Lp = transformer_loss(l, X, dY, BT, D);
    *w = old - EPS;
    float Ln = transformer_loss(l, X, dY, BT, D);
    *w = old;
    return (Lp - Ln) / (2 * EPS);
}

/* Check all elements of a weight matrix against their analytical gradients */
static void check_weight(fArr2D W, fArr2D gW, int rows, int cols,
                         const char* name, TRANSFORMER* l, float* X,
                         const float* dY, int BT, int D)
{
    float* w  = (float*) W;
    float* gw = (float*) gW;
    for (int k = 0; k < rows * cols; k++) {
        float num = numerical_grad(&w[k], l, X, dY, BT, D);
        if (fabsf(num - gw[k]) > TOL) {
            printf("FAIL %s[%d][%d]: expected=%g calculated=%g\n",
                   name, k / cols, k % cols, num, gw[k]);
            exit(1);
        }
    }
}

/* Test 1: zero forward
 * Zero input + zero weights -> zero output.
 */
void test_transformer_zero_forward(TRANSFORMER* l)
{
    printf("Test: transformer zero forward\n");

    const int BT = l->BT;
    const int D  = l->D;

    float X[BT][D];
    float Y[BT][D];

    fltclr(X, BT * D);
    fltclr(Y, BT * D);

    fltclr(l->mha->Wq, D * D);
    fltclr(l->mha->Wk, D * D);
    fltclr(l->mha->Wv, D * D);
    fltclr(l->mha->Wo, D * D);
    fltclr(l->ffn1->Wx, D * l->Dff);
    fltclr(l->ffn2->Wx, l->Dff * D);

    transformer_forward(l, (fArr2D) X, NULL, (fArr2D) Y, 0);

    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < D; j++) {
            if (fabsf(Y[i][j]) > 1e-6f) {
                printf("FAIL: Y[%d][%d] = %g, expected 0\n", i, j, Y[i][j]);
                exit(1);
            }
        }
    }
    printf("  OK\n");
}

/* Test 2: finite difference gradient check
 * Validates dX and all weight gradients from transformer_backward.
 *
 * Loss is L = sum(dY * Y) so the numerical and analytical gradients are
 * checking the same objective (see transformer_loss above).
 *
 * Uses a shadow transformer instance (ls) that shares weights with l but
 * has separate scratch buffers. Numerical gradient evaluation runs forward
 * passes on ls so l's internal state (sdev, xhat, etc.) is not corrupted
 * between the backward call and the gradient checks.
 */
void test_transformer_finite_diff(TRANSFORMER* l)
{
    printf("Test: transformer finite-difference gradients\n");

    const int BT  = l->BT;
    const int D   = l->D;
    const int Dff = l->Dff;
    const int B   = l->B;
    const int T   = l->T;
    const int H   = l->mha->H;

    float X[BT][D];
    float Y[BT][D];
    float dX[BT][D];
    float dY[BT][D];

    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < D; j++) {
            X[i][j]  = urand(-1.0f, 1.0f);
            dY[i][j] = urand(-1.0f, 1.0f);
            dX[i][j] = 0.0f;
        }
    }

    fltclr(Y, BT * D);
    transformer_forward(l, (fArr2D) X, NULL, (fArr2D) Y, 0);
    transformer_backward(l, (fArr2D) dY, (fArr2D) X, (fArr2D) dX, 0);

    /* Shadow instance for numerical gradient evaluation.
     * Shares weights with l via pointer — perturbations to l's weights
     * are visible here, but internal state is separate so forward passes
     * during finite difference don't corrupt l's buffers.
     */
    TRANSFORMER* ls = transformer_create(H, T, D, Dff);
    transformer_init(ls, B, 0, 0.0f);

    /* Point ls weights to l's weights so perturbations are shared */
    freemem(ls->mha->Wq);      ls->mha->Wq      = l->mha->Wq;
    freemem(ls->mha->Wk);      ls->mha->Wk      = l->mha->Wk;
    freemem(ls->mha->Wv);      ls->mha->Wv      = l->mha->Wv;
    freemem(ls->mha->Wo);      ls->mha->Wo      = l->mha->Wo;
    freemem(ls->ffn1->Wx);     ls->ffn1->Wx     = l->ffn1->Wx;
    freemem(ls->ffn2->Wx);     ls->ffn2->Wx     = l->ffn2->Wx;
    freemem(ls->norm1->gamma); ls->norm1->gamma = l->norm1->gamma;
    freemem(ls->norm1->beta);  ls->norm1->beta  = l->norm1->beta;
    freemem(ls->norm2->gamma); ls->norm2->gamma = l->norm2->gamma;
    freemem(ls->norm2->beta);  ls->norm2->beta  = l->norm2->beta;

    float* x_flat  = (float*) X;
    float* dy_flat = (float*) dY;
    float* dx_flat = (float*) dX;

    /* Check dX */
    for (int k = 0; k < BT * D; k++) {
        float num = numerical_grad(&x_flat[k], ls, x_flat, dy_flat, BT, D);
        if (fabsf(num - dx_flat[k]) > TOL) {
            printf("FAIL dX[%d][%d]: expected=%g calculated=%g\n",
                   k / D, k % D, num, dx_flat[k]);
            exit(1);
        }
    }

    /* Check weight gradients */
    check_weight(l->mha->Wq,  l->mha->gWq, D,   D,   "gWq",  ls, x_flat, dy_flat, BT, D);
    check_weight(l->mha->Wk,  l->mha->gWk, D,   D,   "gWk",  ls, x_flat, dy_flat, BT, D);
    check_weight(l->mha->Wv,  l->mha->gWv, D,   D,   "gWv",  ls, x_flat, dy_flat, BT, D);
    check_weight(l->mha->Wo,  l->mha->gWo, D,   D,   "gWo",  ls, x_flat, dy_flat, BT, D);
    check_weight(l->ffn1->Wx, l->gWx1,     D,   Dff, "gWx1", ls, x_flat, dy_flat, BT, D);
    check_weight(l->ffn2->Wx, l->gWx2,     Dff, D,   "gWx2", ls, x_flat, dy_flat, BT, D);

    check_weight((fArr2D) l->norm1->gamma, (fArr2D) l->dg1, 1, D, "dg1", ls, x_flat, dy_flat, BT, D);
    check_weight((fArr2D) l->norm1->beta,  (fArr2D) l->db1, 1, D, "db1", ls, x_flat, dy_flat, BT, D);
    check_weight((fArr2D) l->norm2->gamma, (fArr2D) l->dg2, 1, D, "dg2", ls, x_flat, dy_flat, BT, D);
    check_weight((fArr2D) l->norm2->beta,  (fArr2D) l->db2, 1, D, "db2", ls, x_flat, dy_flat, BT, D);

    /* Null out shared pointers before freeing ls to avoid double-free */
    ls->mha->Wq = ls->mha->Wk = ls->mha->Wv = ls->mha->Wo = NULL;
    ls->ffn1->Wx = ls->ffn2->Wx = NULL;
    ls->norm1->gamma = ls->norm1->beta = NULL;
    ls->norm2->gamma = ls->norm2->beta = NULL;
    transformer_free(ls);

    printf("  OK\n");
}

/* Test 3: causal mask
 * Changing a future token must not affect past token outputs.
 */
void test_transformer_causal_mask(TRANSFORMER* l)
{
    printf("Test: transformer causal mask\n");

    const int BT = l->BT;
    const int T  = l->T;
    const int D  = l->D;

    float X[BT][D];
    float Y1[BT][D];
    float Y2[BT][D];

    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            X[i][j] = urand(-1.0f, 1.0f);

    fltclr(Y1, BT * D);
    transformer_forward(l, (fArr2D) X, NULL, (fArr2D) Y1, 0);

    /* Perturb the last token of the first batch item */
    int last = T - 1;
    for (int j = 0; j < D; j++)
        X[last][j] += 1.0f;

    fltclr(Y2, BT * D);
    transformer_forward(l, (fArr2D) X, NULL, (fArr2D) Y2, 0);

    /* Output at position 0 (past) must be unchanged */
    for (int j = 0; j < D; j++) {
        if (fabsf(Y1[0][j] - Y2[0][j]) > 1e-5f) {
            printf("FAIL causal mask: Y[0][%d] changed from %g to %g "
                   "when future token was perturbed\n", j, Y1[0][j], Y2[0][j]);
            exit(1);
        }
    }

    /* Output at position last must have changed */
    float diff = 0;
    for (int j = 0; j < D; j++)
        diff += fabsf(Y1[last][j] - Y2[last][j]);
    if (diff < 1e-6f) {
        printf("FAIL causal mask: perturbing token %d had no effect on its own output\n", last);
        exit(1);
    }

    printf("  OK\n");
}

/* Test 4: dropout
 * With training=1 and dropout_rate>0, two forward passes should differ.
 * With training=0 (inference), two passes should be identical.
 */
void test_transformer_dropout(TRANSFORMER* l_train, TRANSFORMER* l_infer)
{
    printf("Test: transformer dropout\n");

    const int BT = l_train->BT;
    const int D  = l_train->D;

    float X[BT][D];
    float Y1[BT][D];
    float Y2[BT][D];

    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            X[i][j] = urand(-1.0f, 1.0f);

    /* Training: two passes should differ */
    fltclr(Y1, BT * D);
    transformer_forward(l_train, (fArr2D) X, NULL, (fArr2D) Y1, 0);
    fltclr(Y2, BT * D);
    transformer_forward(l_train, (fArr2D) X, NULL, (fArr2D) Y2, 0);

    float diff = 0;
    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            diff += fabsf(Y1[i][j] - Y2[i][j]);
    if (diff < 1e-6f) {
        printf("FAIL dropout: training passes produced identical outputs\n");
        exit(1);
    }

    /* Inference: two passes must be identical */
    fltclr(Y1, BT * D);
    transformer_forward(l_infer, (fArr2D) X, NULL, (fArr2D) Y1, 0);
    fltclr(Y2, BT * D);
    transformer_forward(l_infer, (fArr2D) X, NULL, (fArr2D) Y2, 0);

    diff = 0;
    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            diff += fabsf(Y1[i][j] - Y2[i][j]);
    if (diff > 0.0f) {
        printf("FAIL dropout: inference passes produced different outputs\n");
        exit(1);
    }

    printf("  OK\n");
}

int main(void)
{
//  unsigned int seed = 42; // 1782848145 1550305486
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    unsigned int seed =
        (unsigned int)(ts.tv_sec ^ ts.tv_nsec);

    printf("seed %d\n",seed);
    init_lrng(seed);


    const int batch_size = 8;
    const int seq_len    = 4;
    const int model_dim  = 12;
    const int ffn_dim    = 36;  /* 4 * model_dim */
    const int num_heads  = 3;
    printf("EPS %g TOL %g\n",EPS,TOL);
    printf("batch_size %d seq_len %d\n",batch_size,seq_len);
    printf("model_dim %d ffn_dim %d num_heads %d\n",model_dim,ffn_dim,num_heads);

    TRANSFORMER* l;

    /* Test 1: zero forward */
    l = transformer_create(num_heads, seq_len, model_dim, ffn_dim);
    transformer_init(l, batch_size, 1, 0.0);
    test_transformer_zero_forward(l);
    transformer_free(l);

    /* Test 2: finite difference */
    l = transformer_create(num_heads, seq_len, model_dim, ffn_dim);
    transformer_init(l, batch_size, 1, 0.0);
    test_transformer_finite_diff(l);
    transformer_free(l);

    if (seq_len > 1) {
        /* Test 3: causal mask */
        l = transformer_create(num_heads, seq_len, model_dim, ffn_dim);
        transformer_init(l, batch_size, 0, 0.0);
        test_transformer_causal_mask(l);
        transformer_free(l);
    }

    /* Test 4: dropout */
    TRANSFORMER* l_train = transformer_create(num_heads, seq_len, model_dim, ffn_dim);
    transformer_init(l_train, batch_size, 1, 0.3);

    TRANSFORMER* l_infer = transformer_create(num_heads, seq_len, model_dim, ffn_dim);
    transformer_init(l_infer, batch_size, 0, 0.0);

    /* Copy weights from train to infer so they're comparable */
    memcpy(l_infer->mha->Wq,   l_train->mha->Wq,   model_dim * model_dim * sizeof(float));
    memcpy(l_infer->mha->Wk,   l_train->mha->Wk,   model_dim * model_dim * sizeof(float));
    memcpy(l_infer->mha->Wv,   l_train->mha->Wv,   model_dim * model_dim * sizeof(float));
    memcpy(l_infer->mha->Wo,   l_train->mha->Wo,   model_dim * model_dim * sizeof(float));
    memcpy(l_infer->ffn1->Wx,  l_train->ffn1->Wx,  model_dim * ffn_dim   * sizeof(float));
    memcpy(l_infer->ffn2->Wx,  l_train->ffn2->Wx,  ffn_dim   * model_dim * sizeof(float));
    memcpy(l_infer->norm1->gamma, l_train->norm1->gamma, model_dim * sizeof(float));
    memcpy(l_infer->norm1->beta,  l_train->norm1->beta,  model_dim * sizeof(float));
    memcpy(l_infer->norm2->gamma, l_train->norm2->gamma, model_dim * sizeof(float));
    memcpy(l_infer->norm2->beta,  l_train->norm2->beta,  model_dim * sizeof(float));

    test_transformer_dropout(l_train, l_infer);
    transformer_free(l_train);
    transformer_free(l_infer);

    printf("\nALL TESTS PASSED\n");
    return 0;
}
