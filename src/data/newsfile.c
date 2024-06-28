/* Copyright (c) 2023-2024 Gilad Odinak */
/* Read News Aggregator samples dataset file */
#include <stdio.h>
#include <string.h>
#include "newsfile.h"

/* Reads news headlines file. The file consists of lines, each storing
 * one sample data, with 2 tab separate fields. The first field contains
 * the headline text string, and the second field is a letter denoting
 * the headline's category: (b)usiness, (t)ravel,(e)ntertainment, (m)edical.
 *
 * For each sample, returns the headline's text.
 * For each category, returns the ordinal number of the category specified
 * in yn[].
 */
int read_news_file(char* newsfile, 
                   char text[1200][200], int yc[1200], char* yn[4])
{
    FILE *fp = fopen(newsfile,"rb");
    if (fp == NULL) {
        fprintf(stderr,"%s: failed to open file for read\n",newsfile);
        return 0;
    }
    for (int i = 0; i < 1200; i++) {
        char buffer[256];
        char *line = fgets(buffer,sizeof(buffer),fp);
        if (line == NULL) {
            fprintf(stderr,"%s: at line %d: "
                           "failed to read from file\n",newsfile,i + 1);
            fclose(fp);
            return 0;
        }
        char cname[16];
        /* text is filled with any characters except tab */
        int cnt = sscanf(line,"%199[^\t]\t%15s",text[i],cname);
        if (cnt < 2) {
            fprintf(stderr,"%s: at line %d: "
                    "failed to parse 2 values from file\n",newsfile,i + 1);
            fclose(fp);
            return 0;
        }
        yc[i] = -1;
        for (int j = 0; j < 4; j++) {
            if (cname[0] == yn[j][0]) { /* Compare only the first letter */
                yc[i] = j;
                break;
            }
        }
        if (yc[i] == -1) {
            fprintf(stderr,"%s: at line %d: "
                    "unknown category %s\n",newsfile,i + 1,cname);
            fclose(fp);
            return 0;
        }
    }
    fclose(fp);
    return 1;
}

