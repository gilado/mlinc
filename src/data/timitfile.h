/* Copyright (c) 2023-2024 Gilad Odinak */
/* Read speech feature files            */
#ifndef TIMITFILE_H
#define TIMITFILE_H

#define TIMIT_CLASS_CNT 64
static const char* phoneme_names[TIMIT_CLASS_CNT] = {
    "","aa","ae","ah","ao","aw","ax","axr",
    "ax-h","ay","b","bcl","ch","d","dcl","dh",
    "dx","eh","el","em","en","eng","er","ey",
    "f","g","gcl","h","hh","hv","ih","ix",
    "iy","jh","k","kcl","l","m","n","ng",
    "nx","ow","oy","p","pcl","q","r","s",
    "sh","t","tcl","th","uh","uw","ux","v",
    "w","wh","y","z","zh","pau","epi","h#"
};

int read_timit_files(const char* file_list,
                     int max_samples, int sample_dim,
                     int max_sequences, int* seq_length,
                     fArr2D x_, iVec y_);
#endif
