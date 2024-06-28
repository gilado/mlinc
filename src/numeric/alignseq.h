/* Copyright (c) 2023-2024 Gilad Odinak */
/* Sequence alignment function     */
#ifndef ALIGNSEQ_H
#define ALIGNSEQ_H

/* Aligns two sequences to have equal length and smallest edit distance.
 *
 * This function uses the Needleman Wunsch algorithm to find the optimal
 * alignment.  In order to align the sequences it may need to create gaps
 * in either or both sequences. The gaps are filled with 'blank' value, 
 * that should not be present in the sequences to be aligned.
 *
 * Parameters:
 *   p     - First sequence to be aligned with second sequence
 *   plen  - The length of the first sequence
 *   t     - Second sequence to be abligned with the first sequence
 *   tlen  - The length of the second sequence
 *   rp    - An array that is filled with the aligned first sequence
 *   rt    - An array that is filled with the aligned second sequence
 *   rlen  - The size of rp, rt arrays. Should be twice the size fo the
 *           longer of p, t.
 *   blank - The value to be used to fill gaps in the aligned sequences
 *
 * Returns:
 *   The edit distance between the aligned sequences.
 *
 * Note:
 *   If the aligned sequences are shorter than rlen, they are terminated
 *   with a 'blank'.
 */ 
int alignseq(const int* p, int plen, 
             const int* t, int tlen,
             int* rp, int* rt, int rlen, int blank);

#endif
