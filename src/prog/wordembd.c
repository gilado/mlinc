/* Copyright (c) 2023-2024 Gilad Odinak */

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "hash.h"
#include "cossim.h"

const char* usage ="Usage: wordembd -i <word embedding file>\n";

/* Value type for evaluation */
enum { VAL_VEC, VAL_SCALAR, VAL_OP };
typedef struct {
    int type;    /* one of: VAL_VEC, VAL_SCALAR, VAL_OP */
    union {
        float *v;    /* for VAL_VEC    */
        float s;     /* for VAL_SCALAR */
        char op;     /* for VAL_OP     */
    };
} Val;

/* Operator precedence */
int precedence(char op)
{
    switch (op) {
        case '+': case '-': return 1;
        case '@': case '/': return 2;
    }
    return 0;
}

/* Evaluates an embedding expression and prints the result.
 *
 * The expression is given as a token array. Each token is either:
 *   - A non-negative integer: index into the embeddings array
 *   - A negative integer: ASCII code of an operator or parenthesis
 *
 * Supported operators:
 *   '+'  vector addition
 *   '-'  vector subtraction
 *   '@'  cosine similarity (scalar result)
 *   '(' ')' grouping
 *
 * Operator precedence:
 *   1. Parentheses
 *   2. '+' and '-' (left associative)
 *   3. '/' (division by a scalar)
 *   4. '@' (cosine similarity)
 *
 * Type rules:
 *   - '+' and '-' operate on embeddings only
 *   - '/' operates on embedding and scalar and yields an embedding
 *   - '@' operates on two embeddings and yields a scalar
 *   - Once a scalar is produced, no further operations are allowed
 *
 * Grammar:
 * expression := term { ('+' | '-') term }
 * term       := factor { ('@' | '/') factor }
 * factor     := embedding | number | '(' expression ')'
 *
 * Behavior:
 *   - Prints the expression using vocabulary strings
 *   - Prints either the resulting embedding or scalar value
 *   - If the result is an embedding, prints the nearest vocabulary word
 *
 * Reference: https://en.wikipedia.org/wiki/Shunting_yard_algorithm
 *
 * Parametes:
 * tokens        - tokenized expression
 * tkncnt        - number of tokens
 * hmap          - word <=> index hashmap
 * embeddings    - vocab_size x embedding_dim array
 * vocab_size    - number of embeddings
 * embedding_dim - embedding dimensionality
 */
void eval_embd_expr(
    Val* tokens, int tkncnt,
    HASHMAP* hmap, fArr2D embeddings,
    int vocab_size, int embedding_dim)
{
    typedef float (*ArrWE)[embedding_dim];
    ArrWE E = (ArrWE)embeddings;

    Val out[256]; int outn = 0;
    Val ops[256]; int opsn = 0;

    for (int i = 0; i < tkncnt; i++) {
        Val t = tokens[i];
        if (t.type != VAL_OP) {
            out[outn++] = t;
            continue;
        }

        char op = t.op;
        if (op == '(') {
            ops[opsn++] = t;
            continue;
        }
        if (op == ')') {
            while (opsn > 0 && ops[opsn - 1].op != '(')
                out[outn++] = ops[--opsn];
            if (opsn > 0) 
                opsn--; // pop '('
            continue;
        }

        while (opsn > 0) {
            char top = ops[opsn - 1].op;
            if (top != '(' && precedence(top) >= precedence(op))
                out[outn++] = ops[--opsn];
            else 
                break;
        }
        ops[opsn++] = t;
    }
    while (opsn > 0)
        out[outn++] = ops[--opsn];

    Val st[256];
    int sp = 0;
    float tv[256][embedding_dim];
    int tp = 0;

    for (int i = 0; i < outn; i++) {
        Val t = out[i];
        if (t.type == VAL_VEC) {
            st[sp++] = (Val){ VAL_VEC, .v = t.v };
            continue;
        }
        if (t.type == VAL_SCALAR) {
            st[sp++] = (Val){ VAL_SCALAR, .s = t.s };
            continue;
        }

        /* t.type == VAL_OP */
        Val b = st[--sp];
        Val a = st[--sp];
        char op = t.op;

        if (op == '@') {
            if (a.type != VAL_VEC || b.type != VAL_VEC) {
                printf("Type error: '@' requires two embeddings\n");
                return;
            }
            float r = cosine_similarity(a.v, b.v, embedding_dim);
            printf("= %f\n", r);
            return;
        }
        if (op == '/') {
            if (a.type != VAL_VEC || b.type != VAL_SCALAR) {
                printf("Type error: '/' requires vector / scalar\n");
                return;
            }
            float* r = tv[tp++];
            for (int k = 0; k < embedding_dim; k++)
                r[k] = a.v[k] / b.s;
            st[sp++] = (Val){ VAL_VEC, .v = r };
            continue;
        }
        if (op == '+' || op == '-') {
            if (a.type != VAL_VEC || b.type != VAL_VEC) {
                printf("Type error: '+' and '-' require embeddings\n");
                return;
            }
            float* r = tv[tp++];
            for (int k = 0; k < embedding_dim; k++)
                r[k] = (op == '+') ? a.v[k] + b.v[k]
                                   : a.v[k] - b.v[k];
            st[sp++] = (Val){ VAL_VEC, .v = r };
            continue;
        }
    }

    if (sp != 1 || st[0].type != VAL_VEC) {
        printf("Invalid expression\n");
        return;
    }

    /* find nearest word */
    int best = -1;
    float best_sim = -1e9f;
    for (int i = 0; i < vocab_size; i++) {
        float s = cosine_similarity(st[0].v, E[i], embedding_dim);
        if (s > best_sim) {
            best_sim = s;
            best = i;
        }
    }

    printf("Result: ");
    for (int i = 0; i < 4 && i < embedding_dim; i++)
        printf("%.6f ", st[0].v[i]);
    if (embedding_dim > 4) printf("...");
    printf("\nNearest: %s (%.6f)\n",
           hashmap_inx2str(hmap, best), best_sim);
}

int main(int argc, char** argv)
{
    FILE* fp;
    char* embfile;
    int vocab_size;
    int embedding_dim;
    float learning_rate;
    int epochs;
    int cnt;

    if (argc < 3 || strcmp(argv[1],"-i") != 0 || strlen(argv[2]) == 0) {
        fprintf(stderr,usage);
        exit(1);
    }
    embfile = argv[2];
    fp = fopen(embfile,"rb");
    if (fp == NULL) {
        fprintf(stderr,"Could not open file '%s' for read\n",embfile);
        exit(1);
    }

    cnt = fscanf(fp,
                "#,vocab_size,%d,embedding_dim,%d,learning_rate,%f,epochs,%d",
                &vocab_size,&embedding_dim,&learning_rate,&epochs);
    if (cnt != 4) {
        fprintf(stderr,"'%s': Invalid file header format.\n", embfile);
        fclose(fp);
        exit(1);
    }

    HASHMAP* hmap = hashmap_create(vocab_size * 3,vocab_size * 15);
    typedef float (*ArrWE)[embedding_dim];
    ArrWE word_embeddings = allocmem(vocab_size,embedding_dim,float);

    /* Read embeddings */
    /* Note that file starts at line 1, and first embedding starts at line 2 */
    int wcnt = 0;
    for (int lineno = 2; ; lineno++) {
        int wrdinx;
        char word[256];
        cnt = fscanf(fp,"%d,%255[^,],",&wrdinx,word);
        if (cnt != 2) {
            if (cnt == -1) /* End of file */
                break;
            fprintf(stderr,
                "'%s': Invalid index, word format at line %d\n",
                embfile,lineno);
            hashmap_free(hmap);
            freemem(word_embeddings);
            fclose(fp);
            exit(1);
        }
        hashmap_str2inx(hmap,word,1);

        for (int j = 0; j < embedding_dim; j++) {
            cnt = fscanf(fp,"%f%*[,]", &word_embeddings[wrdinx][j]);
            if (cnt != 1) {
                fprintf(stderr,
                    "'%s': Invalid embedding value at line %d, value #%d\n",
                    embfile,lineno,j + 1);
                hashmap_free(hmap);
                freemem(word_embeddings);
                fclose(fp);
                exit(1);
            }
        }
        if (++wcnt > vocab_size) {
            fprintf(stderr,
                "'%s': Ignoring words beyond declared vocabulary size of %d\n",
                embfile,vocab_size);
            break;
        }
    }
    fclose(fp);
    if (wcnt < vocab_size)
        vocab_size = wcnt;

    int maxtkn = 256;
    Val tokens[maxtkn];
    int tkninx = 0;

    for (;;) {
        char line[256];
        fprintf(stdout,"> ");
        fflush(stdout);
        char* p = fgets(line,sizeof(line),stdin);
        if (p == NULL)
            break;
        for (; *p != '\0'; p++)
            if (*p < 0x20)
                *p = ' ';
        while (--p >= line && *p == ' ') 
            *p = '\0';

        p = line;
        if (strcmp(p,".") == 0)
            break;
        if (strcmp(p,"=") == 0) {
            eval_embd_expr(tokens,tkninx,hmap,(fArr2D) word_embeddings,vocab_size,embedding_dim);
            printf("\n");
            tkninx = 0;
            continue;
        }

        while (*p != '\0' && tkninx < maxtkn) {
            while (*p == ' ') p++;
            if (*p == '\0') break;

            switch (*p) {
                case '+': case '-': case '/': case '@': case '(': case ')':
                    tokens[tkninx++] = (Val){ VAL_OP, .op = *p++ };
                    continue;
            }

            if (isalpha(*p)) {
                char word[256], *w = word;
                while (*p != '\0' && isalpha(*p))
                    if (w < word + sizeof(word) - 1)
                        *w++ = *p++;
                *w = '\0';

                int winx = hashmap_str2inx(hmap, word, 0);
                if (winx < 0 || winx >= vocab_size)
                    continue;

                tokens[tkninx++] = (Val){ VAL_VEC, .v = word_embeddings[winx] };

                printf("%s:", word);
                for (int i = 0; i < 4 && i < embedding_dim; i++)
                    printf(" %.6f", word_embeddings[winx][i]);
                if (embedding_dim > 4) printf(" ...");
                printf("\n");
            } 
            else 
                p++;
        }
    }

    hashmap_free(hmap);
    freemem(word_embeddings);
    return 0;
}
