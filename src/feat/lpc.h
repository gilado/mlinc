/* Copyright (c) 2023-2024 Gilad Odinak */
/* Linear Prediction Coefficients (LPC) functions */

/* Computes LPC coefficients for the passed in signal samples.
 * For given order=N returns N+1 coefficients in lpcc[]
 * The coefficients are notmalized: lpcc[0] is always 1.0
 * Returns the variance of the signal residual (error)
 */
float computeLPC(const float* samples, int numSamples, int order, double *lpcc);

/* Synthesizes signals' samples for signal's LPC's
 * For given order=N lpcc[] contains N+1 coeficients, with lpcc[0]=1.0
 * Sigma is the square root of the original signal residual (error)
 */ 
void LPCsynthesis(const double* lpcc, int order, float sigma, int num_samples, float* samples);
