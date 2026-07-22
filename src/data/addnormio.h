/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store NN ADDNORM (layer normalization) layer */
#ifndef ADDNORMIO_H
#define ADDNORMIO_H
#include <stdio.h>
#include "addnorm.h"

/* read_addnorm - Read an ADDNORM layer from a file
 *
 * Reads an ADDNORM layer from the file pointed to by fp.
 *
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 *
 * Returns:
 *   Pointer to the read ADDNORM layer if successful, NULL otherwise
 */
ADDNORM* read_addnorm(FILE* fp);

/* write_addnorm - Write an ADDNORM layer to a file
 *
 * Writes the ADDNORM layer pointed to by l to the file pointed to by fp.
 *
 * Parameters:
 *   l  - Pointer to the ADDNORM layer to be written
 *   fp - Pointer to a FILE object representing the output file
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_addnorm(const ADDNORM* l, FILE* fp);

/* load_addnorm - Load an ADDNORM layer from a file
 *
 * Opens the file specified by the filename parameter for reading and
 * loads an ADDNORM layer from it.
 *
 * Parameters:
 *   filename - Name of the file to load the ADDNORM layer from
 *
 * Returns:
 *   Pointer to the loaded ADDNORM layer if successful, NULL otherwise
 */
ADDNORM* load_addnorm(const char* filename);

/* store_addnorm - Store an ADDNORM layer into a file
 *
 * Opens the file specified by the filename parameter for writing and
 * stores the ADDNORM layer pointed to by l into it.
 *
 * Parameters:
 *   l        - Pointer to the ADDNORM layer to be stored
 *   filename - Name of the file to store the ADDNORM layer in
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_addnorm(const ADDNORM* l, const char* filename);

#endif
