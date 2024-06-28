/* Copyright (c) 2023-2024 Gilad Odinak */
/* Converts LPC coefficients to LSP coefficients.
 * For given order=N expects N+1 coefficients in lpc[]
 * The coefficients are notmalized: lpc[0] is always 1.0
 * Returns N+! LSP coefficients in lsp[].
 */
void lpc2lsp(double* lpc, double* lsp, int order);

/* Converts LSP coefficiencts back to LPC coefficients.
 * For given order=N lsp[] contains N+1 coeficients. 
 * Returns N+! LPC coefficients in lpc[].
 */ 
void lsp2lpc(double* lsp, double* lpc, int order);
