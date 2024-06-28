/* plot_graph: Plots a graph of predictions and true values over epochs.
 * 
 * Displays up to three graphs: x vs y, training and validation lossses,
 * and training and validation accuracies.
 * 
 * Parameters:
 *   x             - Array of x-axis input values.
 *   yp            - Array of predicted y-values.
 *   yt            - Array of true y-values.
 *   len           - Length of the x, yp, and yt arrays.
 *   nepochs       - Number of epochs.
 *   losses        - Array of training losses.
 *   accuracies    - Array of training accuracies.
 *   v_losses      - Array of validation losses.
 *   v_accuracies  - Array of validation accuracies.
 *   title         - Title of the plot.
 */
extern
void plot_graph(float* x_, float* yp_, float* yt_, int len, 
                int nepochs, float* losses_, float* accuracies_, 
                float* v_losses_, float* v_accuracies_, const char* title);

/* plot_cm: Plots a confusion matrix with options for displaying numbers,
 *          circles, or both.
 * 
 * Displays up to three graphs: confusion matrix, training and validation
 * lossses, and training and validation accuracies.
 *
 * Parameters:
 *   cm            - A pointer to the first element of a 2D square array 
 *                   representing the confusion matrix.
 *   nc            - Number of classes (number of rows and columns in cm).
 *   clsnames      - Array of class names, containing nc elements.
 *   nepochs       - Number of epochs.
 *   losses_       - Array of training losses.
 *   accuracies_   - Array of training accuracies.
 *   v_losses_     - Array of validation losses.
 *   v_accuracies_ - Array of validation accuracies.
 *   title         - Title of the plot.
 *   mode          - Display mode for confusion matrix cells ('numbers',
 *                   'circles', 'both').
 */
extern
void plot_cm(const int (*)[]/*[nc][nc]*/, int nc, const char** clsnames/*[nc]*/,
             int nepochs, float* losses_, float* accuracies_, 
             float* v_losses_, float* v_accuracies_,
             const char* title, const char* mode);

/* plot_pca: Plots a PCA graph of the given data.
 * 
 * Parameters:
 *   x             - 2D array of x values, shape [len][2].
 *   y             - Array of class labels.
 *   len           - Number of data points.
 *   n_classes     - Number of unique classes.
 *   class_names   - Array of class names (length of n_classes)
 *   point_size    - Size of the points in the plot.
 *   title         - Title of the plot.
 */
extern 
void plot_pca(float x[][2], int* y, int len, 
              int n_classes, const char** class_names, 
              float point_size, const char* title);
