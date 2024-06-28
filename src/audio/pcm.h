/* Copyright (c) 2023-2024 Gilad Odinak */
void pcm2flt(const int16_t* pcmData, float* floatData, int numSamples);
void flt2pcm(const float* floatData, int16_t* pcmData, int numSamples);
