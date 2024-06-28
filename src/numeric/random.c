/* Copyright (c) 2023-2024 Gilad Odinak */
/* Random number generation routines    */
#include "random.h"

int32_t lrng_seed = 96431; /* Prime number */
void init_lrng(int seed)
{
    seed = seed & 0x7FFFFFFF;
    lrng_seed = seed;
}
