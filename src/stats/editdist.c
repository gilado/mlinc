/* Copyright (c) 2023-2024 Gilad Odinak */
/* Levenshtein edit distance function   */
#include "mem.h"
#include "editdist.h"

static inline int minimum(int a, int b, int c) 
{
    int min = a;
    if (b < min)
        min = b;
    if (c < min)
        min = c;
    return min;
}

/* Computes the Levenshtein distance between two sequences.
 * 
 * Calculates the minimum number of single-token edits (insertions, 
 * deletions, or substitutions) required to change one sequence into the other.
 *
 * Parameters:
 *   p - a sequence of token predictions
 *   m - length of sequence pointed by p
 *   t - a sequence of true tokens
 *   n - length of sequence pointed by t
 *
 * Returns:
 *   The edit distance between the two sequences.
 *
 * Reference:
 *   https://en.wikipedia.org/wiki/Levenshtein_distance
 */
int edit_dist(const int* p, int n, const int* t, int m) 
{
    const int heapmem = n >= 10000;
    if (n <= 0) return m;
    if (m <= 0) return n;
    int v0_[heapmem ? 1 : n + 1];
    int v1_[heapmem ? 1 : n + 1];
    int *v0 = heapmem ? allocmem(1,n + 1,int) : v0_;
    int *v1 = heapmem ? allocmem(1,n + 1,int) : v1_;

    for (int i = 0; i <= n; i++)
        v0[i] = i;

    for (int i = 0; i < m; i++) {
        v1[0] = i + 1;
        for (int j = 0; j < n; j++) {
            int del = v0[j + 1] + 1;
            int ins = v1[j] + 1;
            int sub = p[j] == t[i] ? v0[j] : v0[j] + 1;
            v1[j + 1] = minimum(del,ins,sub);
        }
        int *tmp = v0;
        v0 = v1;
        v1 = tmp;
    }
    int dist = v0[n];
    if (heapmem) {
        freemem(v0);
        freemem(v1);
    }
    return dist;
}
