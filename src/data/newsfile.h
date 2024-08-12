/* Copyright (c) 2023-2024 Gilad Odinak                */
/* Reads text files from Kaggle's News Articles Corpus */
#ifndef NEWSFILE_H
#define NEWSFILE_H
#include "hash.h"

/* Stores frequency of a word in the dataset */
typedef struct wrdfrq_s {
    int inx; /* Word index in hashmap */
    int cnt; /* Number of times word was encountered in the txt */
} WRDFRQ; 

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
 *   - If hmap is NULL: Returns the total number of words processed.
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
                 int *file_words, int max_file_words);

#endif
