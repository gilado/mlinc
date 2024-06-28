/* Copyright (c) 2023-2024 Gilad Odinak */
void ulaw2pcm(const uint8_t* ulawData, int16_t* pcmData, int numSamples);
void pcm2ulaw(const int16_t* pcmData, uint8_t* ulawData,  int numSamples);

