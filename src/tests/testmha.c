/* Copyright (c) 2026 Gilad Odinak */
/* Simple test to check mha layer correctness */
#include <stdio.h>
#include <math.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "mha.h"

#define EPS 1e-3f
#define TOL 1e-4f

static inline float frand(void) { return urand(-1.0f, 1.0f); }
static inline int irand(int lo, int hi) { return lo + (int)(lrng() * (hi - lo)); }

float l2diff(const float* a, const float* b, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return sqrtf(s);
}

void assert_close(const char* msg, float a, float b)
{
    if (fabsf(a - b) > TOL) {
        printf("FAIL: %s  (%g vs %g)\n", msg, a, b);
        exit(1);
    }
}

void test_mha_zero_forward(MHA* m)
{
    printf("Test: MHA zero forward\n");

    int BT = m->BT;
    int D  = m->D;

    float* X = calloc(BT * D, sizeof(float));
    float* Y = calloc(BT * D, sizeof(float));

    fltclr(m->Wq, D*D);
    fltclr(m->Wk, D*D);
    fltclr(m->Wv, D*D);
    fltclr(m->Wo, D*D);

    mha_forward(m, (fArr2D)X, (fArr2D)Y, 0);

    for (int i = 0; i < BT * D; i++)
        assert_close("Y should be zero", Y[i], 0.0f);

    free(X);
    free(Y);

    printf("  OK\n");
}

float mha_loss(MHA* m, float* X)
{
    int BT = m->BT;
    int D  = m->D;

    float* Y = calloc(BT * D, sizeof(float));
    mha_forward(m, (fArr2D)X, (fArr2D)Y, 0);

    float L = 0.0f;
    for (int i = 0; i < BT * D; i++)
        L += Y[i];

    free(Y);
    return L;
}

void test_mha_finite_diff(MHA* m)
{
    printf("Test: MHA finite-difference gradients\n");

    int BT = m->BT;
    int D  = m->D;

    float* X  = malloc(sizeof(float) * BT * D);
    float* dX = calloc(BT * D, sizeof(float));
    float* dY = malloc(sizeof(float) * BT * D);

    for (int i = 0; i < BT * D; i++) {
        X[i]  = frand() * 0.1f;
        dY[i] = 1.0f;   /* dL/dY = 1 */
    }

    for (int i = 0; i < D * D; i++) {
        ((float*)m->Wq)[i] = frand() * 0.1f;
        ((float*)m->Wk)[i] = frand() * 0.1f;
        ((float*)m->Wv)[i] = frand() * 0.1f;
        ((float*)m->Wo)[i] = frand() * 0.1f;
    }

    /* analytic gradients */
    mha_forward(m, (fArr2D)X, NULL, 0);
    mha_backward(m, (fArr2D)dY, (fArr2D)X, (fArr2D)dX, 0);

    /* check a few X entries */
    for (int idx = 0; idx < 5; idx++) {
        int i = irand(0,BT * D);

        float old = X[i];
        X[i] = old + EPS;
        float Lp = mha_loss(m, X);
        X[i] = old - EPS;
        float Ln = mha_loss(m, X);
        X[i] = old;

        float num = (Lp - Ln) / (2 * EPS);
        float ana = dX[i];

        if (fabsf(num - ana) > EPS) {
            printf("FAIL dX[%d]: num=%g ana=%g\n", i, num, ana);
            exit(1);
        }
    }

    /* check a few Wq entries */
    float* Wq = (float*)m->Wq;
    float* gWq = (float*)m->gWq;

    for (int k = 0; k < 5; k++) {
        int i = irand(0,D * D);

        float old = Wq[i];
        Wq[i] = old + EPS;
        float Lp = mha_loss(m, X);
        Wq[i] = old - EPS;
        float Ln = mha_loss(m, X);
        Wq[i] = old;

        float num = (Lp - Ln) / (2 * EPS);
        float ana = gWq[i];

        if (fabsf(num - ana) > EPS) {
            printf("FAIL gWq[%d]: num=%g ana=%g\n", i, num, ana);
            exit(1);
        }
    }

    free(X);
    free(dX);
    free(dY);

    printf("  OK\n");
}

float softmax_loss(float* Scores, float* dAtt, int T)
{
    float Att[T*T];
    memcpy(Att, Scores, sizeof(Att));
    softmax((fArr2D)Att, T, T);

    float L = 0.0f;
    for (int i = 0; i < T*T; i++)
        L += Att[i] * dAtt[i];

    return L;
}

void test_softmax_backward(void)
{
    printf("Test: softmax_backward isolated\n");

    const int T = 4;
    float Scores[T*T];
    float dAtt[T*T];
    float dScores[T*T];
    float Att[T*T];

    for (int i = 0; i < T*T; i++) {
        Scores[i] = frand();
        dAtt[i]   = frand();
    }

    memcpy(Att, Scores, sizeof(Scores));
    softmax((fArr2D)Att, T, T);

    memcpy(dScores, dAtt, sizeof(dAtt));
    softmax_backward((fArr2D)dScores, (fArr2D)dAtt, (fArr2D)Att, T, T);

    for (int k = 0; k < 5; k++) {
        int i = irand(0,T * T);

        float old = Scores[i];
        Scores[i] = old + EPS;
        float Lp = softmax_loss(Scores, dAtt, T);
        Scores[i] = old - EPS;
        float Ln = softmax_loss(Scores, dAtt, T);
        Scores[i] = old;

        float num = (Lp - Ln) / (2 * EPS);
        float ana = dScores[i];

        if (fabsf(num - ana) > EPS) {
            printf("FAIL softmax dScores[%d]: num=%g ana=%g\n",
                   i, num, ana);
            exit(1);
        }
    }

    printf("  OK\n");
}

int main(void)
{
    init_lrng(42);

    MHA* m= mha_create(2,2);
    mha_init(m,4,2);

    test_softmax_backward();
    test_mha_zero_forward(m);
    test_mha_finite_diff(m);

    mha_free(m);

    printf("\nALL TESTS PASSED\n");
    return 0;
}

