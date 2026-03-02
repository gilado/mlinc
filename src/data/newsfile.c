/* Copyright (c) 2023-2024 Gilad Odinak */
/* Read News Aggregator samples dataset file */
#include <stdio.h>
#include <ctype.h>
#include "mem.h"
#include "hash.h"
#include "newsfile.h"

/* Returns a pointer to the first letter (a-z A-Z) in a string, or NULL */
static inline char* first_letter(char* s)
{
    while (*s != '\0' && !isalpha(*s)) s++;
    return (*s != '\0') ? s : NULL;
}

/* Returns a pointer to the first non letter in a string */
static inline char* first_nonletter(char* s)
{
    while (isalpha(*s)) s++;
    return s;
}

/* Processes a text file to create a word vocabulary, a word frequency table,
 * and/or an array of word tokens.
 *
 * Parameters:
 *   file_name  - Name of file to be processed.
 *   file_dir   - Directory prefix where the file is located (optional).
 *   hmap       - Hashmap that stores the word vocabulary (optional).
 *   add_new    - If non-zero, new words are added to the hashmap.
 *   max_vocab  - Maximum number of words to include in the vocabulary.
 *   word_freq  - An output array of size max_vocab, where each entry is 
 *                incremented by the number of times the corresponding word
 *                appears in the text file (optional, requires hmap).
 *   file_words - An output array of size max_words, which stores the hashmap
 *                index for each word encountered in the file. If add_new is
 *                zero, words not in the vocabulary are ignored (skipped).
 *   max_words  - Size of file_words array
 *
 * Returns:
 *  - If hmap is not NULL: Returns the number of words that were not skipped.
 *  - If hmap is NULL: Returns the total number of words.
 *  - Returns -1 on error.
 *
 * Notes:
 *  - The frq field of word_freq array elements is not updated by this funciton
 *  - Only consider alphabetic character sequences, delimited by
 *    non-alphabetic characters, as words.
 *  - Convert all alphabetic characters to lowercase before further processing.
 *    For example: "King", "king", "king's" => "king"
 *                 "Kings", "kings", "kings'" => "kings".
 */
int process_news_file(const char* file_name,
                      const char* file_dir,
                      HASHMAP* hmap, int add_new,
                      int max_vocab, WRDFRQ* word_freq,
                      int *file_words, int max_words)
{
    char buffer[20000];
    int cnt;                /* Number of characters in buffer        */
    int off = 0;            /* Location in buffer for next file read */
    int file_word_cnt = 0;  /* Number of words in the file           */

    int maxpath = 512;
    char filepath[maxpath * 3];
    filepath[0] = '\0';
    if (file_dir != NULL)
        strncpy(filepath,file_dir,maxpath);
    filepath[maxpath - 1] = '\0';
    int pfxlen = strlen(filepath);
    if (pfxlen > 0 && filepath[pfxlen - 1] != '/') {
        strcat(filepath,"/");
        pfxlen = strlen(filepath);
    }
    strncpy(filepath + pfxlen,file_name,maxpath - pfxlen - 1);
    filepath[maxpath - 1] = '\0';

    FILE* fp = fopen(filepath,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In process_news_file: "
                "failed to open data file '%s' for read\n",file_name);
        return -1;
    }

    for (;;) {
        cnt = fread(buffer + off,1,sizeof(buffer) - off - 1,fp) + off;
        if (cnt < off) cnt = off;
        buffer[cnt] = '\0';
        char* w = first_letter(buffer);
        while (w != NULL) {
            char* e = first_nonletter(w);
            int len = e - w;
            if (e == buffer + cnt) { /* String ends at end of buffer         */
                if (isalpha(buffer[cnt - 1])) { /* Last char part of a word  */
                    if (!feof(fp)) { /* Word may continue past end of buffer */
                        memmove(buffer,w,len);
                        off = len;
                        break;
                    }
                }
            }
            if (len == 0) {
                off = 0;
                break;
            }
            for (int i = 0; i < len; i++)
                w[i] = tolower(w[i]);
            w[len] = '\0'; /* Replace non letter with end of string */
            if (hmap != NULL) {
                int inx = hashmap_str2inx(hmap,w,add_new);
                if (inx >= 0 && inx < max_vocab) {
                    if (word_freq != NULL) {
                        word_freq[inx].inx = inx;
                        word_freq[inx].cnt++;
                    }
                    if (file_words != NULL) {
                        if (file_word_cnt < max_words)
                            file_words[file_word_cnt] = inx;
                        else {
                            fprintf(stderr,
                                "\nFile contains more than %d words\n",
                                max_words);
                            return file_word_cnt;
                        }
                    }
                    /* Count only words that are not skipped */
                    file_word_cnt++;
                }
            }
            else
                file_word_cnt++; /* Count all words */
            if (e + 1 - buffer >= (int) sizeof(buffer))
                break;
            w = first_letter(e + 1); /* Continue past end of prv string */
        }
        if (feof(fp))
            break;
    }
    fclose(fp);
    return file_word_cnt;
}

/* Reads a list file containing file names (one per line), filters to include
 * only those files that end with the ".txt" extension (case-insensitive),
 * verifies that these files exist under the given data directory, and returns
 * an array of the valid file names (without paths).
 *
 * Parameters:
 *   list_file  - Path to a text file containing file names, one per line.
 *   data_dir   - Directory prefix where the data files are located.
 *   num_files  - Pointer to an integer where the function stores the count of
 *                valid files found.
 *
 * Returns an array of pointers to file names, or NULL on failure.
 * The returned array should be freed by calling free_news_file_list().
 *
 * Note that the function only returns file names, not full file paths.
 */
char** read_news_file_list(const char* list_file,
                           const char* data_dir,
                           int* num_files)
{
    char** file_list = NULL;
    FILE* lfp;
    int maxpath = 512;
    char filepath[maxpath * 3];
    strncpy(filepath,data_dir,maxpath);
    filepath[maxpath - 1] = '\0';
    int pfxlen = strlen(filepath);
    if (filepath[pfxlen - 1] != '/') {
        strcat(filepath,"/");
        pfxlen = strlen(filepath);
    }

    lfp = fopen(list_file,"rb");
    if (lfp == NULL) {
        fprintf(stderr,"In read_file_list: "
                "failed to open list file '%s' for read\n",list_file);
        return NULL;
    }

    int max_files = 0;
    for (;;) {
        char* filename = filepath + pfxlen;
        filename = fgets(filename,maxpath - pfxlen,lfp);
        if (filename == NULL || strlen(filename) == 0)
            break; /* End of file list */
        max_files++;
    }
    fclose(lfp);

    lfp = fopen(list_file,"rb");
    if (lfp == NULL) {
        fprintf(stderr,"In read_file_list: "
                "failed to open list file '%s' for read\n",list_file);
        return NULL;
    }

    file_list = allocmem(max_files + 1,1,char*);

    int file_cnt = 0;
    for (;;) {
        char* filename = filepath + pfxlen;
        filename = fgets(filename,maxpath - pfxlen,lfp);
        if (filename == NULL || strlen(filename) == 0)
            break; /* End of file list */

        filename[strcspn(filename, "\r\n")] = '\0';
        const int fnlen = strlen(filename);

        const char* ext = ".txt";
        int off = fnlen - strlen(ext);
        if (off <= 0 || strcasecmp(filename + off, ext) != 0)
            continue; /* Skip files whose name does not end with ext */

        FILE* fp = fopen(filepath, "rb");
        if (fp == NULL) {
            fprintf(stderr,"In read_file_list: "
                    "failed to open data file '%s' for read\n",filename);
            continue;
        }
        fclose(fp);
        file_list[file_cnt] = allocmem(fnlen + 1,1,char);
        strcpy(file_list[file_cnt],filename);
        file_cnt++;
    }
    *num_files = file_cnt;
    return file_list;
}

/* Frees the memory allocated and returned by read_news_file_list()
 *
 * file_list - An array of pointers to file names
 * num_files - The number of entries in file_list array
 */
void free_news_file_list(char** file_list, int num_files)
{
    for (int i = 0; i < num_files; i++)
        freemem(file_list[i]);
    freemem(file_list);
}
