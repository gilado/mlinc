/* Copyright (c) 2026 Gilad Odinak */
/* Simple test to check mha layer correctness */
#include <stdio.h>
#include <math.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "mha.h"

#define EPS 1e-3f

void test_mha_zero_forward(MHA* m)
{
    printf("Test: MHA zero forward\n");

    int BT = m->BT;
    int D  = m->D;

    float X[BT][D];
    float Y[BT][D];

    fltclr(X, BT*D);
    fltclr(Y, BT*D);
    fltclr(m->Wq, D*D);
    fltclr(m->Wk, D*D);
    fltclr(m->Wv, D*D);
    fltclr(m->Wo, D*D);

    mha_forward(m, X, NULL, Y, 0, 0);

    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < D; j++) {
            if (fabsf(Y[i][j]) > 1e-9f) {
                printf("Y[%d][%d] should be zero but is %g\n",i,j,Y[i][j]);
                exit(1);
            }
        }
    }

    printf("  OK\n");
}

float mha_loss(MHA* m, fArr2D X)
{
    int BT = m->BT;
    int D  = m->D;

    float Y[BT][D];
    fltclr(Y, BT * D);
    
    mha_forward(m, X, NULL, Y, 0, 0);

    float L = 0;
    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            L += Y[i][j];

    return L;
}

void test_mha_finite_diff(MHA* m)
{
    printf("Test: MHA finite-difference gradients\n");

    int BT = m->BT;
    int D  = m->D;

    float X[BT][D];
    float dX[BT][D];
    float dY[BT][D];

    typedef float (*ArrDD)[D];

    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < D; j++) {
            X[i][j]  = urand(-0.1, 0.1);
            dX[i][j] = 0;
            dY[i][j] = 1;   /* dL/dY = 1 */
        }
    }

    ArrDD Wq = (ArrDD) m->Wq;
    ArrDD Wk = (ArrDD) m->Wk;    
    ArrDD Wv = (ArrDD) m->Wv;
    ArrDD Wo = (ArrDD) m->Wo;    
    ArrDD gWq = (ArrDD) m->gWq;

    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            Wq[i][j] = urand(-0.1, 0.1);
            Wk[i][j] = urand(-0.1, 0.1);
            Wv[i][j] = urand(-0.1, 0.1);
            Wo[i][j] = urand(-0.1, 0.1);
        }
    }

    mha_forward(m, X, NULL, NULL, 0, 0);
    mha_backward(m, dY, X, dX, 0);

    for (int i = 0; i < BT; i++) {
        for (int j = 0; j < D; j++) {

            float old = X[i][j];
            X[i][j] = old + EPS;
            float Lp = mha_loss(m, X);
            X[i][j] = old - EPS;
            float Ln = mha_loss(m, X);
            X[i][j] = old;

            float num = (Lp - Ln) / (2 * EPS);
            float ana = dX[i][j];

            if (fabsf(num - ana) > EPS) {
                printf("FAIL dX[%d][%d]: num=%g ana=%g\n", i, j, num, ana);
                exit(1);
            }
        }
    }

    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {

            float old = Wq[i][j];
            Wq[i][j] = old + EPS;
            float Lp = mha_loss(m, X);
            Wq[i][j] = old - EPS;
            float Ln = mha_loss(m, X);
            Wq[i][j] = old;

            float num = (Lp - Ln) / (2 * EPS);
            float ana = gWq[i][j];

            if (fabsf(num - ana) > EPS) {
                printf("FAIL gWq[%d][%d]: num=%g ana=%g\n", i, j, num, ana);
                exit(1);
            }
        }
    }

    printf("  OK\n");
}

float softmax_loss(fArr2D Scores, fArr2D dAtt_, int T)
{
    float Att[T][T];
    fltcpy(Att, Scores, T * T);
    softmax(Att, T, T);

    typedef float (*ArrTT)[T];
    const ArrTT dAtt = (ArrTT) dAtt_;

    float L = 0;
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++)
        L += Att[i][j] * dAtt[i][j];

    return L;
}

void test_softmax_backward(void)
{
    printf("Test: softmax_backward isolated\n");

    const int T = 4;
    float Scores[T][T];
    float dAtt[T][T];
    float dScores[T][T];
    float Att[T][T];

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            Scores[i][j] = urand(-1, 1);
            dAtt[i][j]   = urand(-1, 1);
        }
    }

    memcpy(Att, Scores, sizeof(Scores));
    softmax((fArr2D)Att, T, T);

    memcpy(dScores, dAtt, sizeof(dAtt));
    softmax_backward(dScores, dAtt, Att, T, T);

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {

            float old = Scores[i][j];
            Scores[i][j] = old + EPS;
            float Lp = softmax_loss(Scores, dAtt, T);
            Scores[i][j] = old - EPS;
            float Ln = softmax_loss(Scores, dAtt, T);
            Scores[i][j] = old;

            float num = (Lp - Ln) / (2 * EPS);
            float ana = dScores[i][j];

            if (fabsf(num - ana) > EPS) {
                printf("FAIL softmax dScores[%d][%d]: num=%g ana=%g\n",
                       i, j, num, ana);
                exit(1);
            }
        }
    }

    printf("  OK\n");
}

void test_mask(MHA* m) {
    printf("Test: MHA mask\n");

    int B = m->B;
    int T = m->T;
    int D = m->D;
    int BT = m->BT;

    float X[BT][D];
    float Y[BT][D];

    typedef float (*ArrDD)[D];
    typedef float (*ArrTT)[T];

    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            X[i][j] = urand(-0.1, 0.1);

    fltclr(m->Wq, D * D);
    fltclr(m->Wk, D * D);
    fltclr(m->Wv, D * D);
    fltclr(m->Wo, D * D);

    ArrDD Wq = (ArrDD) m->Wq;
    ArrDD Wk = (ArrDD) m->Wk;    
    for (int i = 0; i < D / 2; i++) {
        Wq[i][i] = 1;
        Wk[i][i] = 1;
    }

    mha_forward(m, X, NULL, Y, 1, 0);

    ArrTT Scores = (ArrTT) m->Scores;

    for (int b = 0; b < B; b++) {
        for (int i = 0; i < T; i++) {
            for (int j = i + 1; j < T; j++) {
                if (fabsf(Scores[i][j]) > 1e-9) {
                    printf("FAIL mask: Scores[%d][%d] = %g not masked\n", i, j, Scores[i][j]);
                    exit(1);
                }
            }
        }
    }

    printf("  OK\n");
}

void test_padding_mask(MHA* m) {
    printf("Test: MHA padding mask\n");

    int B = m->B;
    int T = m->T;
    int D = m->D;
    int BT = m->BT;

    float X[BT][D];
    float Y[BT][D];

    for (int i = 0; i < BT; i++)
        for (int j = 0; j < D; j++)
            X[i][j] =  urand(-0.1, 0.1);

    // Create pad_mask array - length B*T
    // The last two tokens of the last batch are masked
    int pad_mask[BT];
    for (int i = 0; i < BT; i++)
        pad_mask[i] = 1;
    pad_mask[(B - 1) * T + (T - 2)] = 0;
    pad_mask[(B - 1) * T + (T - 1)] = 0;

    mha_forward(m, X, pad_mask, Y, 0, 0);

    typedef float (*ArrTT)[T];
    ArrTT Scores = (ArrTT)m->Scores;

    // Check Scores corresponding to padded tokens are close to zero
    for (int t = T - 2; t < T; t++) {
        for (int j = 0; j < T; j++) {
            if (Scores[j][t] > 1e-8) {
                printf("FAIL pad mask col: Scores[%d][%d] = %g not masked\n", j, t, Scores[j][t]);
                exit(1);
            }
        }
    }

    printf("  OK\n");
}

int main(void)
{
    init_lrng(42);

    MHA* m = mha_create(2, 4);  // batch=2, seq_len=4
    mha_init(m, 8, 2, 1);       // D=8, H=2

    test_softmax_backward();
    test_mha_zero_forward(m);
    test_mha_finite_diff(m);

    test_mask(m);
    test_padding_mask(m);

    mha_free(m);

    printf("\nALL TESTS PASSED\n");
    return 0;
}

