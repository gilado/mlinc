/* Copyright (c) 2023-2024 Gilad Odinak */
#define NUMPHN 64
#define SIZEPHN 8
typedef struct phninfo_t {
    uint32_t startPos; // in samples
    uint32_t endPos;
    char phoneme[SIZEPHN];
    int label;
} PHNINFO;

typedef struct phnfile_t {
    FILE *fileHandle;
    char mode;         // 'r'
} PHNFILE;

PHNFILE* openPhonemeFile(const char* filename, char *mode, PHNFILE* pf);
PHNFILE* closePhonemeFile(PHNFILE* pf);
size_t readPhonemeFile(PHNFILE* pf, size_t cnt, PHNINFO *phninfo);

int encodePhoneme(char *phn);
const char *decodePhoneme(int enc);
