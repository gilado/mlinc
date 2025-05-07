/* Copyright (c) 2023-2024 Gilad Odinak */

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
int lpc2lsp(const double lpc[], double lsp[], int order);

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
void lsp2lpc(double lsp[], double lpc[], int order);
