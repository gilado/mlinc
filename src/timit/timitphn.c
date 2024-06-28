/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "timitphn.h"

PHNFILE* openPhonemeFile(const char* filename, char *mode, PHNFILE* pf)
{
    FILE* fileHandle = NULL;   
    if (*mode != 'r') {
        fprintf(stderr,"In openPhonemeFile('%s'): invalid mode '%s'; only 'r' supported.\n",filename,mode);
        return NULL;
    }
    //printf("Openning '%s' for read.\n",filename);

    fileHandle = fopen(filename,"rb");
    if (fileHandle == NULL) {
        fprintf(stderr,"In openPhonemeFile('%s'): failed to open the file for read.\n",filename);
        return NULL;
    }
    pf->fileHandle = fileHandle;
    pf->mode = *mode;   
    return pf;
}

PHNFILE* closePhonemeFile(PHNFILE* pf)
{
    FILE *fileHandle = pf->fileHandle;
    int rv;
    pf->fileHandle = NULL;
    if (fileHandle == NULL)
      return pf;
    rv = fclose(fileHandle);
    if (rv != 0) {
        fprintf(stderr,"In closePhonemeFile(): failed to close the file.\n");
        return NULL;
    }
    return pf;
}

size_t readPhonemeFile(PHNFILE* pf, size_t cnt, PHNINFO *phninfo)
{
    size_t rcnt = 0;
    for (rcnt = 0; rcnt < cnt; rcnt++) {
        char line[100];
        char *s = fgets(line,sizeof(line),pf->fileHandle);
        if (s == NULL)
            break;
        uint32_t startPos;
        uint32_t endPos;
        char phoneme[4];
        int e = sscanf(line,"%u %u %4s",&startPos,&endPos,phoneme);
        if (e < 3) {
            fprintf(stderr,"In readPhonemeFile(): malformed line '%s'\n",line);
            break;
        }
        phninfo->startPos = startPos;
        phninfo->endPos = endPos;
        strcpy(phninfo->phoneme,phoneme);
        phninfo->label = encodePhoneme(phoneme);
        phninfo++;
    };
    return rcnt;
}

// phn2vusn[][1]: v-voiced, u-unvoiced, s-silence, n-noise 
static const char *phn2vusn[NUMPHN][2] = {
{"",""},{"aa","v"},{"ae","v"},{"ah","v"},
{"ao","v"},{"aw","v"},{"ax","v"},{"axr","v"},
{"ax-h","u"},{"ay","v"},{"b","v"},{"bcl","v"},
{"ch","u"},{"d","v"},{"dcl","v"},{"dh","v"},
{"dx","v"},{"eh","v"},{"el","v"},{"em","v"},
{"en","v"},{"eng","v"},{"er","v"},{"ey","v"},
{"f","u"},{"g","v"},{"gcl","v"},{"h","v"},
{"hh","u"},{"hv","u"},{"ih","v"},{"ix","v"},
{"iy","v"},{"jh","v"},{"k","u"},{"kcl","u"},
{"l","v"},{"m","v"},{"n","v"},{"ng","v"},
{"nx","v"},{"ow","v"},{"oy","v"},{"p","u"},
{"pcl","u"},{"q","v"},{"r","v"},{"s","u"},
{"sh","u"},{"t","u"},{"tcl","u"},{"th","u"},
{"uh","v"},{"uw","v"},{"ux","v"},{"v","v"},
{"w","v"},{"wh","v"},{"y","v"},{"z","v"},
{"zh","v"},{"pau","s"},{"epi","s"},{"h#","s"}
};

int encodePhoneme(char *phn)
{
    for (int i = 0; i < NUMPHN; i++) {
        if (strcmp(phn2vusn[i][0],phn) == 0)
            return i;
    }
    return -1;
}

const char *decodePhoneme(int enc) 
{
    if (enc < 0 || enc >= NUMPHN)
        return NULL;
    return phn2vusn[enc][0];
}

