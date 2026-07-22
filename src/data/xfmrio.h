/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store NN decoder-only TRANSFORMER layer */
#ifndef XFMRIO_H
#define XFMRIO_H
#include <stdio.h>
#include "transformer.h"

/* read_transformer - Read a TRANSFORMER layer from a file
 *
 * Reads a TRANSFORMER layer from the file pointed to by fp.
 *
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 *
 * Returns:
 *   Pointer to the read TRANSFORMER layer if successful, NULL otherwise
 */
TRANSFORMER* read_transformer(FILE* fp);

/* write_transformer - Write a TRANSFORMER layer to a file
 *
 * Writes the TRANSFORMER layer pointed to by l to the file pointed to by fp.
 *
 * Parameters:
 *   l     - Pointer to the TRANSFORMER layer to be written
 *   final - If not zero, record the layer as inference-only
 *   fp    - Pointer to a FILE object representing the output file
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_transformer(const TRANSFORMER* l, int final, FILE* fp);

/* load_transformer - Load a TRANSFORMER layer from a file
 *
 * Opens the file specified by the filename parameter for reading and
 * loads a TRANSFORMER layer from it.
 *
 * Parameters:
 *   filename - Name of the file to load the TRANSFORMER layer from
 *
 * Returns:
 *   Pointer to the loaded TRANSFORMER layer if successful, NULL otherwise
 */
TRANSFORMER* load_transformer(const char* filename);

/* store_transformer - Store a TRANSFORMER layer into a file
 *
 * Opens the file specified by the filename parameter for writing and
 * stores the TRANSFORMER layer pointed to by l into it.
 *
 * Parameters:
 *   l        - Pointer to the TRANSFORMER layer to be stored
 *   filename - Name of the file to store the TRANSFORMER layer in
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_transformer(const TRANSFORMER* l, const char* filename);

#endif
