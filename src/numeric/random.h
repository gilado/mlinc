/* Copyright (c) 2023-2024 Gilad Odinak */
/* Random number generation routines    */
#ifndef RANDOM_H
#define RANDOM_H
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Initializes the random number generator
 */
extern int32_t lrng_seed;
void init_lrng(int seed);

/* lrng returns a pseudo-random real number uniformly distributed 
 * between 0.0 and 1.0. 
 * Lehmer random number generator - Steve Park & Dave Geyer
 */
static inline float lrng(void)
{
    const int32_t modulus = 2147483647; /* 0x7FFFFFFF   */
    const int32_t multiplier = 48271;   /* Prime number */
    const int32_t q = modulus / multiplier;
    const int32_t r = modulus % multiplier;
    int32_t t = multiplier * (lrng_seed % q) - r * (lrng_seed / q);
    lrng_seed = (t > 0) ? t : t + modulus;
    float num = ((float) lrng_seed / modulus);
    return num;
}

/* Return a random number following a uniform distribution
 */
static inline float urand(float min, float max) 
{
    float num = lrng() * (max - min) + min;
    return num;
}

/* Return a random number following a normal distribution
 * with the provided mean and standard deviation.
 */
static inline float nrand(float mean, float stddev) 
{
    /* Box-Muller transform */
    float z = sqrt(-2.0 * log(lrng())) * sin(2.0 * M_PI * lrng());
    float num =  mean + stddev * z; /* Shift and scale */
    return num;
}

#endif
