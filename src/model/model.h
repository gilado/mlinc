/* Copyright (c) 2023-2024 Gilad Odinak */
/* Multi layer neural network model data structures and functions */
#ifndef MODEL_H
#define MODEL_H
#include <math.h>
#include "array.h"
#include "loss.h"
#include "ctc.h"
#include "adamw.h"
#include "dense.h"
#include "lstm.h"

typedef struct layer_s {
    char type;      /* (d)ense or (l)stm */
    union {
        DENSE* dense;
        LSTM* lstm;
    };
    fArr2D* grads;  /* Array of gradients and adam momentums    */
    int num_grads;  /* Number of entries in grads[]             */
} LAYER;

typedef struct model_s {
    int num_layers; /* Number of layers                           */
    LAYER *layer;   /* Array of layers with num_layers elements   */ 
    int batch_size; /* Number of input vectors processed together */
    int input_dim;  /* Input vectors dimension (may include bias) */
    int add_bias;   /* Either 1 (add) or 0 (do not add)           */ 
    int output_dim; /* Output vectors dimension (last layer size) */
    char loss_func; /* (m)ean-square-root (c)ross-entropy (C)tc   */
    CTC* ctc;       /* For ctc loss                               */
    char optimizer; /* (l)inear (a)damw                           */
    int update_cnt; /* For adamw optimizer                        */
    int normalize;  /* If not zero, normalize input data          */
    fVec mean;      /* For input normalization                    */
    fVec sdev;      /* For input normalization                    */
    int final;      /* If zero, can be further trained            */
} MODEL;

/* Creates a container for multi layer neural network.
 * num_layers specifies the number of layers
 * batch_size is the number of vector inputs processed together 
 * between model updates.
 * input_dim is the dimension of the model first layer input.
 *
 * If add_bias is zero, input_dim includes a bias dimension 
 * whose value is 1.0; otherwise, a bias dimension is added internally.
 *
 * If normalize is not zero, normalizes input feature vectors by feature,
 * to have mean of zero and standard deviation of one.
 *
 * Use model_add() to add layers to the model up to numlayers.  
 *
 * Note that the output dimension is determined by the size of the last layer.
 */
MODEL* model_create(int num_layers, 
                    int batch_size, int input_dim, int add_bias, int normalize);

/* Frees the memory allocated by model_create() and all added layers */
void model_free(MODEL* m);

/* Adds a layer to a model
 * m points to a model
 * layer points to a neural network (e.g. DENSE) to be added as a layer
 * type is the type of the layer: "dense" or "lstm"
 * 
 * The layer is added after all other layers in the model
 */
void model_add(MODEL* m, void* layer, const char* type);

/* Prepares model for training. 
 *
 * loss_func can be one of mean-square-error cross-entropy ctc
 *
 * optimizer can be one of (l)inear (a)damw
 *
 * both optimizers incorporate weight decay; to disable, set it to 0 when
 * invoking model_fit().
 */
void model_compile(MODEL* m, const char* loss_func, const char* optimizer);

/* Sets a new batch size.
 *
 * This function changes the batch size of an existing, possibly trained,
 * model. Smaller batch size reduces memory requirements and decoding latency
 * but may also reduce accuracy.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 */
void model_set_batch_size(MODEL* m, int batch_size);

/* Sets a new loss function.
 *
 * Changes the loss function of an existing, possibly trained, model. 
 * This can be used to further train a pre-trained model.
 *
 * Currently only switching from cross-entropy to ctc-loss is supported
 *
 * Parameters:
 *   loss_func - ctc
 *
 * Returns: 1 on successful change, 0 otherwise
 */
int model_set_loss_function(MODEL* m, const char* loss_func);

/* Trains model on data xTr and true outputs yTr. The data is organized as
 * a list of data sample sequences of varying lengths and corresponding
 * true outputs. The dimension of the vectors in x sequences is
 * the input dimension of the first layer, and the dimension of the
 * vectors in y sequences is the output dimension of the last layer.
 *
 * This function may be called more than once to train a pre-trained
 * model.
 *
 * xTr is an array of vectors of one or more training data sequences.
 *
 * yTr is an array of vectors of corresponding true outputs.
 *
 * lenTr is an array of integers denoting the lengths of training sequences.
 * if the data consist of one sequence (or is not sequential) it can be NULL.
 *
 * numTr is the number of vectors in xTr, yTr if lenTr is NULL. Otherwise,
 * it is the number of sequences; the total number of vectors is the sum
 * of the values in lenTr.
 *
 * xVd is an array of vectors of one or more training data sequences.
 *
 * yVd is an array of vectors of corresponding true outputs.
 *
 * lenVd is an array of integers denoting the lengths of training sequences.
 * if the data consist of one sequence (or is not sequential) it can be NULL.
 *
 * numVd is the number of vectors in xVd, yVd if lenVd is NULL. Otherwise,
 * it is the number of sequences; in that case, the total number of vectors
 * is the sum of the values in lenVd. Set numVd to 0 to indicate that no
 * validation data is provided. In that case, xVd, yVd and lenVd should
 * be set to NULL.
 *
 * shuffle applies only when data consists of one sequence; if not zero,
 * samples within the sequence are shuffled. Otherwise, sequences are 
 * always shuffled, and samples within sequences are never shuffled.
 *
 * num_epochs is the number of iterations through the entire dataset.
 *
 * learning_rate is a gradient multiplier controling the rate of descent.
 *
 * weight_decay is a multiplier that suppresses weights magnitude.
 * 
 * If losees is not NULL, it points to an array of num_epoch elements
 * that are updated with the loss value of each epoch.
 *
 * If accuracies is not NULL, it points to an array of num_epoch elements
 * that are updated with the accuracy at the end of each epoch.
 * For regression, accuracy is the R-squared coefficient
 * For classification, accuracy is the fraction of predicted labels that
 * matched the true labels.
 *
 * Similarily, if validation data is present and v_losses, v_accuracies 
 * are not NULL, they are updated with the validation loss and accuracy
 * at the end of each epoch.
 *
 * If final is not zero, frees gradients memory at the end of training.
 * Otherwise, memory is retained, allowing further training of the model.
 *
 * If verbose is not zero, prints the loss and accuracy values at the 
 * end of each epoch to standard output; if it is greater then 1, prints
 * each epoch's loss and accuracy on a separate line.
 */ 
void model_fit(MODEL* m, 
    const fArr2D xTr, const fArr2D yTr, const int *lenTr, int numTr, 
    const fArr2D xVd, const fArr2D yVd, const int *lenVd, int numVd, 
    int shuffle, int num_epochs, float learning_rate, float weight_decay,
    float* losses, float* accuracies, float* v_losses, float* v_accuracies,
    int final, int verbose);
    
/* Predicts the outputs of the inputs samples in x, and returns them in y.
 *
 * x is an array of input samples.
 * y is an array to be updated with output predictions.
 * len is number of smaples.
 */
void model_predict(MODEL* m, const fArr2D x, fArr2D y, int len);

#endif
