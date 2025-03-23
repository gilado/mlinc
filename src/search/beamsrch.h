/* Copyright (c) 2023-2024 Gilad Odinak */
/* Beam search decoder             */
#ifndef BEAMSRCH_H
#define BEAMSRCH_H

/*  Performs beam search decoding on a set of probabilities over time steps.
 *
 * Parameters:
 *   probabilities - A 2D array [T][C] of probabilities, where T is the number
 *                   of time steps and C is the number of classes (symbols).
 *   T             - The number of time steps.
 *   C             - The number of classes (symbols) at each time step.
 *   beam_width    - The beam width, i.e., the number of sequences to keep
 *                   at each step.
 *   sequences     - A 2D array [B][T+1] to store the resulting sequences,
 *                   where B is the beam width.
 *   scores        - A 1D array [B] to store the scores of the 
 *                   resulting sequences.
 *
 * Note:
 *   This function allocates heap memoryif (B * C) * ( T + 1) > 65536 bytes.
 */
void beam_search(fArr2D probabilities, 
                 int T, int C, int beam_width, 
                 iArr2D sequences, fVec scores);

#endif
