/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdint.h>

typedef struct sphfile_t {
    FILE *fileHandle;
    uint16_t audioFormat; // 1:PCM, 3:float, 7:uLaw
    uint16_t numChannels; // 1, 2
    uint32_t sampleRate;  // 8000 16000 
    uint16_t bitDepth;    // 8, 16, 32
    uint32_t dataSize;    // in bytes
    uint32_t numSamplesPerChannel;
    uint32_t numSamples;  // Across all channels
    char endianess;       // 'l' or 'b'
    char mode;            // 'r' or 'w'
} SPHFILE;

SPHFILE* openSphereFile(const char* filename, char *mode, SPHFILE *sf);
SPHFILE* closeSphereFile(SPHFILE* sf);
size_t readSphereFile(SPHFILE* sf, void *buffer, size_t numSamples);
size_t seekSphereFile(SPHFILE* sf, size_t offsetSamples);
size_t readSphereAudio(SPHFILE* sf, float *fltBuf, size_t numSamples);
