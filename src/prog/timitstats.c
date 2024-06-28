/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "timitphn.h"

struct phn_s {
  char name[SIZEPHN];
  double meanDur;
  double sumSqr;
  int count;
} phn[NUMPHN];

void initphn() 
{
    for (int i = 0; i < NUMPHN; i++) {
        const char *name = decodePhoneme(i);
        strncpy(phn[i].name,name,SIZEPHN);
        phn[i].meanDur = 0.0;
        phn[i].sumSqr = 0.0;
        phn[i].count = 0;
    }
}

// https://cpsc.yale.edu/sites/default/files/files/tr222.pdf
// equations 1.3a and 1.3b
inline void onlineSDev(int *count, double *mean, double *sumSqr, double newVal)
{
    (*count)++; 
    double newcnt = ((double) *count);  // j
    double delta = newVal - *mean;      // Xj - M1,j-1
    double incr = delta / newcnt;       // 1/j * (Xj - M1,j-1)
    *mean += incr;                      // M1,j
    double delta2 = newVal - *mean;     // Xj - M1,j
    *sumSqr += delta * delta2;          // S1,j * j
}

int main(int argc, char **argv)
{
    initphn();
    if (argc < 2) {
        fprintf(stderr,"Syntax: timitstats2 <filelist file>\n");
        return 0;
    }

    char *filelist = argv[1];
    FILE *fp = fopen(filelist,"rb");
    if (fp == NULL) {
        fprintf(stderr,"Failed to open '%s' for read\n",filelist);
        return 0;
    }
    int fileno;
    for (fileno = 0;; fileno++) {
        char buffer[100];
        char *fn = fgets(buffer,sizeof(buffer),fp);
        if (fn == NULL || strlen(fn) == 0) {
            fclose(fp);
            break;
        }
        fn[strlen(fn) - 1] = '\0'; // punch out end of line char
        PHNFILE pfin, *pfp;
        pfp = openPhonemeFile(fn,"r",&pfin);
        if (pfp == NULL) {
            fprintf(stderr,"Failed to open '%s' for read - skipping\n",fn);
            continue;
        }
        int lineno;
        for (lineno = 0;; lineno++) {
            PHNINFO phni;
            size_t cnt = readPhonemeFile(pfp,1,&phni);
            if (cnt < 1)
                break;
            double duration = 
                           ((double) (phni.endPos - phni.startPos)) / 16000.0;
            int i = phni.label;
            onlineSDev(&phn[i].count,&phn[i].meanDur,&phn[i].sumSqr,duration);
        }
    }
    printf("Processed %d files\n",fileno);
    printf("phoneme,count  ,mean   ,stddev ,\n");
    for (int i = 0; phn[i].name[0] != '\0'; i++) {
        double sdev = sqrt(phn[i].sumSqr / ((double) phn[i].count));
        printf("%-7s,%7d,%7.3lf,%7.3lf,\n",
               phn[i].name,phn[i].count,phn[i].meanDur,sdev);
    }
}
