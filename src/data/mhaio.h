/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store NN MHA (Multi-Head Attention) layer */
#ifndef MHAIO_H
#define MHAIO_H
#include <stdio.h>
#include "mha.h"

/* read_mha - Read an MHA layer from a file
 *
 * Reads an MHA layer from the file pointed to by fp.
 *
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 *
 * Returns:
 *   Pointer to the read MHA layer if successful, NULL otherwise
 */
MHA* read_mha(FILE* fp);

/* write_mha - Write an MHA layer to a file
 *
 * Writes the MHA layer pointed to by l to the file pointed to by fp.
 *
 * Parameters:
 *   l     - Pointer to the MHA layer to be written
 *   final - If not zero, record the layer as inference-only.
 *   fp    - Pointer to a FILE object representing the output file
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_mha(const MHA* l, int final, FILE* fp);

/* load_mha - Load an MHA layer from a file
 *
 * Opens the file specified by the filename parameter for reading and
 * loads an MHA layer from it.
 *
 * Parameters:
 *   filename - Name of the file to load the MHA layer from
 *
 * Returns:
 *   Pointer to the loaded MHA layer if successful, NULL otherwise
 */
MHA* load_mha(const char* filename);

/* store_mha - Store an MHA layer into a file
 *
 * Opens the file specified by the filename parameter for writing and
 * stores the MHA layer pointed to by l into it.
 *
 * Parameters:
 *   l        - Pointer to the MHA layer to be stored
 *   filename - Name of the file to store the MHA layer in
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_mha(const MHA* l, const char* filename);

#endif
