/* Copyright (c) 2023-2025 Gilad Odinak */

/* LPC to LSP conversion routines
 * Reference https://www.ece.mcgill.ca/~pkabal/papers/1986/Kabal1986.pdf
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "lsp.h"

/* eval_cheb_poly - Evaluate a Chebyshev polynomial using recurrence
 *
 * This function evaluates a Chebyshev polynomial of the first kind at
 * a given point x. The polynomial is expressed in terms of Chebyshev
 * basis functions T_n(x), and the coefficients are provided in coef[].
 *
 * The recurrence used is:
 *     T_0(x) = 1
 *     T_1(x) = x
 *     T_n(x) = 2 * x * T_{n-1}(x) - T_{n-2}(x)  for n >= 2
 *
 * coef[] should contain (order / 2) + 1 elements, corresponding to the
 * even-order Chebyshev polynomial derived from LPC analysis.
 *
 * Parameters:
 *     coef  - array of Chebyshev coefficients, highest degree first
 *     x     - value at which to evaluate the polynomial (in [-1, 1])
 *     order - LPC order (must be even)
 *
 * Returns:
 *     The value of the Chebyshev polynomial at x
 */
static inline double eval_cheb_poly(const double coef[], double x, int order)
{
  double T[(order / 2) + 1];
  T[0] = 1.0;
  T[1] = x;
  for (int i = 2; i <= order / 2; i++)
    T[i] = 2 * x * T[i - 1] - T[i - 2];

  double sum = 0.0;
  for (int i = 0; i <= order / 2; i++)
    sum += coef[(order / 2) - i] * T[i];

  return sum;
}

/* lpc2lsp - Convert LPC coefficients to LSP (Line Spectral Pair) frequencies
 *
 * This function converts a set of linear predictive coding (LPC) coefficients
 * into their corresponding line spectral pair (LSP) frequencies using 
 * Chebyshev polynomial root finding.
 *
 * It constructs the symmetric and antisymmetric polynomials (P and Q) from 
 * the LPC coefficients, then searches for roots of these polynomials on the
 * interval [-1, 1] using a grid search followed by bisection refinement.
 *
 * Parameters:
 *     lpc   - input LPC coefficients, array of size order+1
 *     lsp   - output array to store LSP frequencies in radians, size order+1
 *     order - LPC order (must be even)
 *
 * Returns:
 *     Number of roots found (should equal 'order' if successful)
 *
 * Note:
 *     The number of bisections is 17.
 *     The search interval step is 0.005.
 */
int lpc2lsp(const double lpc[], double lsp[], int order)
{
  const int bisectcnt = 17;  
  const double step = 0.005;
  double P[order + 1], Q[order + 1];

  for (int i = 0; i < order; i++) lsp[i] = 0.0;
  
  P[0] = Q[0] = 1.0;
  for (int i = 1; i <= order / 2; i++) {
    P[i] = lpc[i] + lpc[order + 1 - i] - P[i - 1];
    Q[i] = lpc[i] - lpc[order + 1 - i] + Q[i - 1];
  }
  for (int i = 0; i < order / 2; i++) {
    P[i] *= 2.0;
    Q[i] *= 2.0;
  }
  
  int roots = 0;
  double xl = -1.0, xm = 0.0, xr = 1.0;
  for (int j = 0; j < order; j++) {
    double *pq = (j % 2 == 0) ? P : Q;
    double sl, sm, sr;
    sr = eval_cheb_poly(pq,xr,order);

    while (xl >= -1.0) {
      xl = xr - step;
      sl = eval_cheb_poly(pq,xl,order);

      if (sr * sl <= 0.0) {
        sm = sr;
        for (int k = 0; k < bisectcnt; k++) {
          xm = (xr + xl) / 2;
          sm = eval_cheb_poly(pq,xm,order);
          if (sr * sm > 0.0) {
            xr = xm;
            sr = sm;
          } 
          else {
            xl = xm;
            sl = sm;
          }
        }
        lsp[j] = acos(xm);
        xr = xm;
        roots++;
        break;
      } 
      sr = sl;
      xr = xl;
    }
  }
  return roots;
}

/* lsp2lpc - Convert Line Spectral Pairs to Linear Prediction Coefficients 
 *
 * This function transforms a set of LSP frequencies (in radians) into LPC
 * coefficients using polynomial reconstruction. It constructs two real
 * polynomials, P(z) and Q(z), from the LSPs, and combines them to produce
 * the LPC prediction filter A(z).
 *
 * Arguments:
 *   lsp   - array of LSP frequencies in radians (length = order)
 *   lpc   - output array of LPC coefficients (length = order + 1)
 *   order - order of the LPC filter (must be even)
 *
 * The LPC polynomial is computed using:
 *   A(z) = 0.5 * [ (1 + z⁻¹) * P(z) + (1 - z⁻¹) * Q(z) ]
 * where P(z) and Q(z) are constructed from the cosine of the LSP frequencies.
 */
void lsp2lpc(double lsp[], double lpc[], int order)
{
  double freq[order];
  double Pz[order + 3], Qz[order + 3];

  for (int i = 0; i < order; i++) 
    freq[i] = cos(lsp[i]);

  for (int i = 0; i < order + 3; i++)
    Pz[i] = Qz[i] = 0.0;

  /* Reconstruct P(z) and Q(z) by evaluating the recursive form of
   * y[n] = x[n] - 2 * x * y[n - 1] + y[n - 2]
   */ 
  double Pi = 1.0, Qi = 1.0;
  for (int j = 0; j <= order; j++) {
    double Po, Qo;
    for (int i = 0; i < (order / 2); i++) {
      Po = Pi - 2 * freq[2 * i] * Pz[i * 2] + Pz[i * 2 + 1];
      Qo = Qi - 2 * freq[2 * i + 1] * Qz[i * 2] + Qz[i * 2 + 1];
      Pz[i * 2 + 1] = Pz[i * 2];
      Qz[i * 2 + 1] = Qz[i * 2];
      Pz[i * 2] = Pi;
      Qz[i * 2] = Qi;
      Pi = Po;
      Qi = Qo;
    }
    Po = Pi + Pz[order  + 2];
    Qo = Qi - Qz[order  + 2];
    lpc[j] = (Po + Qo) / 2;
    Pz[order + 2] = Pi;
    Qz[order + 2] = Qi;
    Pi = 0.0;
    Qi = 0.0;
  }
}
