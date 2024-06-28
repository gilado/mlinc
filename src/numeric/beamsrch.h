/* Copyright (c) 2023-2024 Gilad Odinak */
/* Beam search decoder             */
#ifndef BEAMSRCH_H
#define BEAMSRCH_H



void beam_search(fArr2D probabilities_, 
                 int T, int C, int beam_width, 
                 iArr2D sequences_, fVec scores_);

#endif
