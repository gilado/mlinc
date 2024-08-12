/* Copyright (c) 2023-2024 Gilad Odinak */
/* Activation functions and their derivatives */
#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "array.h"

/* Applies the sigmoid activation function to each element of a 2D array
 *
 * Parameters:
 *   m  : Pointer to the 2D array to be processed
 *   B  : Number of rows in the matrix
 *   S  : Number of columns in the matrix
 */
static inline void sigmoid(fArr2D m_/*[B][S]*/, int B, int S) 
{
    typedef float (*ArrBS)[S];
    ArrBS m = (ArrBS) m_;

    for (int i = 0; i < B; i++)    
        for (int j = 0; j < S; j++)    
            m[i][j] = 1.0 / (1.0 + exp(-m[i][j]));
}

/* Applies the rectified linear unit (ReLU) activation function to each 
 * element of a 2D array.
 *
 * Parameters:
 *   m  : Pointer to the 2D array to be processed
 *   B  : Number of rows in the matrix
 *   S  : Number of columns in the matrix
 */
static inline void relu(fArr2D m_/*[B][S]*/, int B, int S) 
{
    typedef float (*ArrBS)[S];
    ArrBS m = (ArrBS) m_;

    for (int i = 0; i < B; i++)    
        for (int j = 0; j < S; j++)    
            if (m[i][j] < 0.0)
                m[i][j] = 0.0;
}

/* Applies the softmax activation function to each row of a 2D array
 *
 * Parameters:
 *   a  : Pointer to the 2D array to be processed
 *   B  : Number of rows in the matrix
 *   K  : Number of columns in the matrix
 *
 * Motes:
 * Converts predictions in vectors that are the rows of the passed in array
 * to probabilities.
 *
 * See https://en.wikipedia.org/wiki/Softmax_function
 *
 * This implementation improves numerical stability by normalizing the input.
 * It subtracts the maximum value from each row to prevent overflow.
 */
static inline void softmax(fArr2D a_/*[B][K]*/, int B, int K)
{
    typedef float (*ArrBK)[K];
    ArrBK a = (ArrBK) a_;
    for (int j = 0; j < B; j++) {
        typedef float (*VecK); 
        VecK  p = (VecK) a[j];

        float m = 0.0; /* max(p[]) */
        for (int i = 0; i < K; i++) {
            if (m < p[i])
                m = p[i];
        }
        float s = 0.0; /* sum(exp(p[] - m) */
        for (int i = 0; i < K; i++) {
            p[i] = exp(p[i] - m);
            s += p[i];
        }
        for (int i = 0; i < K; i++)
            p[i] /= s;
    }
}

/* Calculates the derivative of the sigmoid function at point sigmoid_z.
 *
 * Parameters:
 *   z : The output of the sigmoid function
 *
 * Note:
 * The derivative of the sigmoid function, is sigmoid(x) * (1 - sigmoid(x))
 * However the value passed to this function is z = sigmoid(x)
 * sigmoid(x) * (1 - sigmoid(x)) => z * (1 - z)
 */
static inline float d_sigmoid_1(float z)
{
    return z * (1 - z);
}

/* Calculates the derivative of the sigmoid function for a 2D array z, and 
 * updates the input array x, by multiplying its original values by the 
 * derivative. 
 *
 * Parameters:
 *   x : Pointer to the 2D array to be updated
 *   z : Pointer to the 2D array, the output of the sigmoid activation function
 *   B : Number of rows in the matrices
 *   D : Number of columns in the matrices
 */
static inline void d_sigmoid(fArr2D x_/*[B][D]*/, 
                             const fArr2D z_/*[B][D]*/, 
                             int B, int D)
{
    typedef float (*ArrBD)[D];
    ArrBD x = (ArrBD) x_;
    const ArrBD z = (const ArrBD) z_;
    
    for (int i = 0; i < B; i++)    
        for (int j = 0; j < D; j++)    
            x[i][j] *= d_sigmoid_1(z[i][j]);
}

/* Calculates the derivative of the ReLU function at point z.
 *
 * Parameters:
 *   z : The output of the ReLU function
 *
 * Notice that the drivative of relu(x) and the derivative of relu(relu(x))
 * yield the same values.
 */
static inline float d_relu_1(float z)
{
    return (z > 0.0) ? 1.0 : 0.0;
}

/* Calculates the derivative of the sigmoid function for a 2D array z, and 
 * updates the input array x, by multiplying its original values by the 
 * derivative. 
 *
 * Parameters:
 *   x : Pointer to the 2D array to be updated
 *   z : Pointer to the 2D array, the output of the ReLU function
 *   B : Number of rows in the matrices
 *   D : Number of columns in the matrices
 */
static inline void d_relu(fArr2D x_/*[B][D]*/,
                          const fArr2D z_/*[B][D]*/, 
                          int B, int D)
{
    typedef float (*ArrBD)[D];
    ArrBD x = (ArrBD) x_;
    const ArrBD z = (const ArrBD) z_;
    
    for (int i = 0; i < B; i++)    
        for (int j = 0; j < D; j++)    
            x[i][j] *= d_relu_1(z[i][j]);
}

/* Calculates the derivative of the softmax function for a 2D array z, and 
 * updates the input array x, by multiplying its original values by the 
 * derivative. 
 *
 * Parameters:
 *   x  : Pointer to the 2D array to be updated
 *   z  : Pointer to the 2D array, the output of the softmax function
 *   yt : Pointer to the 2D array of the the target values
 *   B  : Number of rows in the matrices
 *   D  : Number of columns in the matrices
 *
 * Reference:
 *   https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative
 */
static inline void d_softmax(fArr2D x_/*[B][D]*/, 
                             const fArr2D z_/*[B][D]*/, 
                             const fArr2D yt_/*[B][D]*/, 
                             int B, int D)
{
    typedef float (*ArrBD)[D];
    ArrBD x = (ArrBD) x_;
    const ArrBD z = (const ArrBD) z_;
    const ArrBD yt = (ArrBD) yt_;
    
    for (int i = 0; i < B; i++)
        for (int j = 0; j < D; j++)
            x[i][j] *= z[i][j] * (yt[i][j] - z[i][j]);
}

/* Calculates the derivative of the hyperbolic tangent (tanh) function
 *
 * Parameters:
 *   x : Input value to the tanh function
 *
 * Note:
 * The canonical definition of d/dx tanh(x) is (1 / cosh(x)^2); this function
 * uses an equivalnet.
 */ 
static inline float d_tanh(float x)
{
    return 1 - pow(tanh(x),2);
}

/* Calculates the derivative of the hyperbolic tangent (tanh) function given
 * its output i.e. z = tanh(x)
 *
 * Parameters:
 *   z : Output value of the tanh function
 *
 * Note:
 * The derivative of the tanh function, is 1 - tanh(x) * tanh(x)
 * However the value passed to this function is z = tanh(x)
 *  1 - tanh(x) * tanh(x) => 1 - z * z
 */
static inline float d_tanh_x(float z)
{
    return 1 - pow(z,2);
}

#endif
