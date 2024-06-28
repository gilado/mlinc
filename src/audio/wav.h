/* Copyright (c) 2023-2024 Gilad Odinak */
typedef struct wavfile_t {
    FILE *fileHandle;
    uint16_t audioFormat; // 1:PCM 7:uLaw
    uint16_t numChannels; // 1, 2
    uint32_t sampleRate;  // 8000 16000 
    uint16_t bitDepth;    // 8, 16, 32
    uint32_t dataSize;    // in bytes
    uint32_t numSamplesPerChannel;
    uint32_t numSamples;  // Across all channels
    char endianess;       // 'l' or 'b'
    char mode;            // 'r' or 'w'
} WAVFILE;

WAVFILE* openWavFile(const char* filename, char *mode, WAVFILE *wf);
WAVFILE* closeWavFile(WAVFILE* wf);
size_t readWavFile(WAVFILE* wf, void *buffer, size_t numSamples);
size_t seekWavFile(WAVFILE* wf, size_t offsetSamples);
size_t writeWavFile(WAVFILE* wf, void *buffer, size_t numSamples);

