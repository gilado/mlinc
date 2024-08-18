/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include "mem.h"
#include "etime.h"
#include "array.h"
#include "random.h"
#include "norm.h"
#include "cossim.h"
#include "findsim.h"
#include "annoy.h"

static void print_vector(fVec vec, int dim);

int main()
{
    int num_vec = 3000000;
    int vec_dim = 100;
    int num_trees = 4;
    float search_q = 0.5;
    int topn = 5;

    int seed = 42;
    init_lrng(seed);

    printf("\nCreating %d data point vectors of %d dimensions\n",
                                                              num_vec,vec_dim);

    /* Create random data vectors */
    float (*data)[vec_dim] = allocmem(num_vec,vec_dim,float);
    for (int i = 0; i < num_vec; i++)
        for (int j = 0; j < vec_dim; j++)
            data[i][j] = urand(0,1);

    float query[vec_dim];
    for (int j = 0; j < vec_dim; j++)
        query[j] = urand(0,1);

    /* Build search tree for the data */
    printf("\nBuilding search tree (%d trees) ... ",num_trees);
    fflush(stdout);
    float start_time = current_time();
    ANNOY* annoy = annoy_create(data,num_vec,vec_dim,num_trees);
    printf("%6.3f seconds\n",elapsed_time(start_time));

    /* Find topn most similar vectors using annoy */
    printf("\nFind %d vectors similar to\n",topn);
    print_vector(query,vec_dim);
    printf("\n(annoy search_q = %g)\n",search_q);
    
    int a_similar[topn];
    float a_similarity[topn];
    start_time = current_time();
    int a_cnt = annoy_most_similar(annoy,query,search_q,
                                                  a_similar,a_similarity,topn);
    float a_find_time = elapsed_time(start_time) * 1000;

    /* Find topn most similar vectors using exhaustive search */
    int similar[topn];
    float similarity[topn];
    start_time = current_time();
    int cnt = find_most_similar(data,num_vec,vec_dim,
                                query,similar,similarity,topn);
    float find_time = elapsed_time(start_time) * 1000;

    if (a_cnt < topn)
        topn = a_cnt;
    if (cnt < topn)
        topn = cnt;

    printf("\nAnnoy search results "
           "(%d checks %5.3f milliseconds):\n",annoy->cos_sim_cnt,a_find_time);
    float a_sum_sim = 0;
    for (int i = 0; i < topn; i++) {
        a_sum_sim += a_similarity[i];
        print_vector(data[a_similar[i]],vec_dim);
        printf(", Similarity %6.4f\n",a_similarity[i]);
    }
    printf("Overall similarity %6.4f\n",a_sum_sim / topn);

    printf("\nExhaustive search results "
           "(%d checks, %5.3f milliseconds):\n",num_vec,find_time);
    float sum_sim = 0;
    for (int i = 0; i < topn; i++) {
        sum_sim += similarity[i];
        print_vector(data[similar[i]],vec_dim);
        printf(", Similarity %6.4f, In Annoy: ",similarity[i]);
        int j;
        for (j = 0; j < topn; j++)
            if (similar[i] == a_similar[j])
                break;
        printf("%s\n",((j < topn) ? "Yes" : "No"));
    }
    printf("Overall similarity %6.4f\n",sum_sim / topn);

    annoy_free(annoy);
    freemem(data);
    printf("\n");
    return 0;
}

static void print_vector(fVec vec, int dim)
{
    int edge = 2;
    printf("[");
    if (dim <= 2 * edge) {
        for (int i = 0; i < dim; i++)
            printf("%7.4f ",vec[i]);
    }
    else {
        for (int i = 0; i < edge; i++)
            printf("%7.4f ",vec[i]);
        printf("... ");
        for (int i = dim - edge; i < dim; i++)
            printf("%7.4f ",vec[i]);
    }
    printf("]");
}
