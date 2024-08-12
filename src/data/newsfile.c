/* Copyright (c) 2023-2024 Gilad Odinak */
/* Read News Aggregator samples dataset file */
#include <stdio.h>
#include <ctype.h>
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
 *   fp         - Pointer to a text file opened for reading.
 *   hmap       - Pointer to a hashmap that stores the word vocabulary
 *                (optional).
 *   add_new    - If non-zero, new words are added to the hashmap.
 *   max_vocab  - Maximum number of words to include in the vocabulary.
 *   word_freq  - Pointer to an array of size max_vocab, where each entry is 
 *                incremented by the number of times the corresponding word 
 *                appears in the text file (optional, requires hmap).
 *   file_words - Pointer to an array of size max_file_words, which stores the 
 *                hashmap index for each word encountered in the file.
 *                If add_new is zero, words not in the vocabulary are ignored
 *                (skipped).
 *
 * Returns:
 *   - If hmap is not NULL: Returns the number of words that were not skipped.
 *   - If hmap is NULL: Returns the total number of words.
 *   - Returns -1 on error.
 *
 * Notes:
 *   - Only consider alphabetic character sequences, delimited by 
 *     non-alphabetic characters, as words.
 *   - Convert all alphabetic characters to lowercase before further processing.
 *   - For example: "King", "king", "king's" => "king"
 *                  "Kings", "kings", "kings'" => "kings".
 */
int process_file(FILE* fp, HASHMAP* hmap, int add_new,
                 int max_vocab, WRDFRQ* word_freq,
                 int *file_words, int max_file_words)
{
    char buffer[20000];
    int cnt;                /* Number of characters in buffer        */
    int off = 0;            /* Location in buffer for next file read */
    int file_word_cnt = 0;  /* Number of words in the file           */
    for (;;) {
        cnt = fread(buffer + off,1,sizeof(buffer) - off - 1,fp) + off;
        if (cnt < off) cnt = off;
        buffer[cnt] = '\0';
        char* w = first_letter(buffer);
        while (w != NULL) {
            char* e = first_nonletter(w);
            int len = e - w;
            if (e == buffer + cnt) { /* String ends at end of buffer          */
                if (isalpha(buffer[cnt - 1])) { /* Last char part of a word   */
                    if (!feof(fp)) { /* Word may continue past end of buffer  */
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
                        if (file_word_cnt < max_file_words)
                            file_words[file_word_cnt] = inx;
                        else {
                            fprintf(stderr,
                                    "\nFile contains more than %d words\n",
                                    max_file_words);
                            return file_word_cnt;
                        }
                    }
                    file_word_cnt++; /* Count only words that are not skipped*/
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
    return file_word_cnt;
}
