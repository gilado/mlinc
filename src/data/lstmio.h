/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to load and store NN LSTM layer  */
#ifndef LSTMIO_H
#define LSTMIO_H
#include <stdio.h>
#include "lstm.h"

/* read_lstm - Read an LSTM layer from a file
 * 
 * Reads an LSTM layer from the file pointed to by fp.
 * 
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 * 
 * Returns:
 *   Pointer to the read LSTM layer if successful, NULL otherwise
 */
LSTM* read_lstm(FILE* fp);

/* write_lstm - Write an LSTM layer to a file
 * 
 * Writes the LSTM layer pointed to by d to the file pointed to by fp. 
 * 
 * Parameters:
 *   d  - Pointer to the LSTM layer to be written
 *   fp - Pointer to a FILE object representing the output file
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_lstm(const LSTM* d, FILE* fp);

/* load_lstm - Load an LSTM layer from a file
 * 
 * Opens the file specified by the filename parameter for reading and 
 * loads an LSTM layer from it.
 * 
 * Parameters:
 *   filename - Name of the file to load the LSTM layer from
 * 
 * Returns:
 *   Pointer to the loaded LSTM layer if successful, NULL otherwise
 */
LSTM* load_lstm(const char* filename);

/* store_lstm - Store an LSTM layer into a file
 * 
 * Opens the file specified by the filename parameter for writing and 
 * stores the LSTM layer pointed to by d into it.
 * 
 * Parameters:
 *   d        - Pointer to the LSTM layer to be stored
 *   filename - Name of the file to store the LSTM layer in
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_lstm(const LSTM* d, const char* filename);

#endif
