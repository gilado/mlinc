/* Copyright (c) 2023-2024 Gilad Odinak */
/* Reads samples from News Aggregator dataset file */
#ifndef NEWSFILE_H
#define NEWSFILE_H

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
                   char text[1200][200], int yc[1200], char* yn[4]);
#endif
