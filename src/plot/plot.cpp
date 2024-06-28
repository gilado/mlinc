/* Copyright (c) 2023 Gilad Odinak           */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <math.h>
#include "matplotlibcpp.h"

extern "C" 
void ctrlc(int sig) 
{ 
    (void) sig;
    printf("\n"); 
    exit(0); 
}

namespace plt = matplotlibcpp;
#include <vector>

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
extern "C" 
void plot_graph(float* x_, float* yp_, float* yt_, int len, 
                int nepochs, float* losses_, float* accuracies_, 
                float* v_losses_, float* v_accuracies_, const char* title) 
{
    signal(SIGINT, ctrlc); /* matplotcpp hijacks this */
    printf("Plotting results - close plot window to continue\n");
    int num_plots = (accuracies_ != NULL) ? 3 : 2;
    std::vector<double> x(x_,x_ + len);
    std::vector<double> yt(yt_,yt_ + len);
    std::vector<double> yp(yp_,yp_ + len);
    double epochs_[nepochs];
    for (int i = 0; i < nepochs; i++) epochs_[i] = i + 1; 
    std::vector<double> epochs(epochs_,epochs_ + nepochs);
    plt::figure_size(num_plots * 300,400);

    plt::subplot(1,num_plots,1);
    plt::plot(x, yt,{{"label", "Actual"}});
    plt::plot(x, yp,{{"label", "Predicted"}});
    plt::title(title);
    plt::ylabel("$f(x)$");
    plt::xlabel("$x$");
    plt::legend();

    if (losses_ != NULL) {
        plt::subplot(1,num_plots,2);
        std::vector<double> losses(losses_,losses_ + nepochs);
        plt::named_semilogy("Training",epochs,losses,"#1F7784");
        if (v_losses_ != NULL) {
            std::vector<double> v_losses(v_losses_,v_losses_ + nepochs);
            plt::named_semilogy("Validation",epochs,v_losses,"#FF7F0E");
        }
        plt::title("Loss");
        plt::xlabel("Epoch");
        plt::legend();
    }

    if (accuracies_ != NULL) {
        plt::subplot(1,num_plots,3);
        std::vector<double> accuracies(accuracies_,accuracies_ + nepochs);
        plt::named_semilogy("Training",epochs,accuracies,"#1F7784");
        if (v_accuracies_ != NULL) {
            std::vector<double> v_accuracies(v_accuracies_,v_accuracies_ + nepochs);
            plt::named_semilogy("Validation",epochs,v_accuracies,"#FF7F0E");
        }    
        plt::title("Accuracy");
        plt::xlabel("Epoch");
        plt::legend();
    }
    try {
        plt::show();
    }
    catch(...) {
    }
    signal(SIGINT, ctrlc); /* matplotcpp hijacks this */
}

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
extern "C"
void plot_cm(const int* cm_/*[nc][nc]*/, int nc, const char** clsnames/*[nc]*/,
             int nepochs, float* losses_, float* accuracies_, 
             float* v_losses_, float* v_accuracies_,
             const char* title, const char* mode)
{
    int txt = !strcasecmp(mode,"numbers") || !strcasecmp(mode,"both");
    int gfx = !strcasecmp(mode,"circles") || !strcasecmp(mode,"both");
    if (!gfx && !txt) {
        fprintf(stderr,"In plot_cm: invalid mode '%s'\n",mode);
        return;
    }
    signal(SIGINT, ctrlc); /* matplotcpp hijacks this */
    printf("Plotting results - close plot window to continue\n");
    if (nepochs == 0)
        losses_ = accuracies_ = NULL;
    int num_plots = (accuracies_ != NULL) ? 3 : ((losses_ != NULL) ? 2 : 1);
    double epochs_[nepochs];
    for (int i = 0; i < nepochs; i++) epochs_[i] = i + 1; 
    std::vector<double> epochs(epochs_,epochs_ + nepochs);
    plt::figure_size(num_plots * 300,400);
    plt::subplot(1,num_plots,1);
    plt::title(title);
        
    int cls[nc];
    for (int i = 0; i < nc; i++)
        cls[i] = i;
    
    std::vector<int> ticks(cls,cls + nc);
    std::vector<std::string> labels(clsnames,clsnames + nc);
    std::map<std::string, std::string> xticks_keywords;
    xticks_keywords["rotation"] = "vertical"; 
    std::map<std::string, double> margins;
    margins["bottom"] = 0.2;
    plt::subplots_adjust(margins);
    plt::xticks(ticks, labels,xticks_keywords);
    plt::yticks(ticks, labels);

    for (int i = 0; i <= nc; ++i) {
        plt::plot({i - 0.5,i - 0.5},{-0.5,nc - 0.5},"k-"); // Vertical lines
        plt::plot({-0.5,nc - 0.5},{i - 0.5,i - 0.5},"k-"); // Horizontal lines
    }

    typedef int (*CM)[nc];
    CM cm = (CM) cm_;
    int maxcm = 0;
    for (int i = 0; i < nc; i++)
        for (int j = 0; j < nc; j++)
            if (maxcm < cm[i][j])
                maxcm = cm[i][j];
    float scale = log(1 + sqrt(maxcm));        
    for (int i = 0; i < nc; i++) {
        for (int j = 0; j < nc; j++) {
            std::vector<double> xs, ys;
            float size = sqrt(cm[i][j] * 1000) / scale;
            xs.push_back(j);
            ys.push_back(i);
            if (gfx)
                plt::scatter(xs,ys,size,{{"label",""}});
            std::string str = std::to_string(cm[i][j]);
            if (txt)
                plt::text((double)(j - 0.075 * str.size()),(double)(i - 0.05),str);
        }
    }
    if (losses_ != NULL) {
        plt::subplot(1,num_plots,2);
        std::vector<double> losses(losses_,losses_ + nepochs);
        plt::named_semilogy("Trainig",epochs,losses,"#1F7784");
        std::vector<std::string> yticklabels;
        if (v_losses_ != NULL) {
            std::vector<double> v_losses(v_losses_,v_losses_ + nepochs);
            plt::named_semilogy("Validation",epochs,v_losses,"#FF7F0E");
        }
        plt::title("Loss");
        plt::xlabel("Epoch");
        plt::legend();
    }

    if (accuracies_ != NULL) {
        plt::subplot(1,num_plots,3);
        std::vector<double> accuracies(accuracies_,accuracies_ + nepochs);
        plt::named_semilogy("Trainnig",epochs,accuracies,"#1F7784");
        if (v_accuracies_ != NULL) {
            std::vector<double> v_accuracies(v_accuracies_,v_accuracies_ + nepochs);
            plt::named_semilogy("Validation",epochs,v_accuracies,"#FF7F0E");
        }    
        plt::title("Accuracy");
        plt::xlabel("Epoch");
        plt::legend();
    }
    try {
        plt::show();
    }
    catch(...) {
    }
    signal(SIGINT, ctrlc); /* matplotcpp hijacks this */
}

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
extern "C" 
void plot_pca(float x[][2], int* y, int len, 
              int n_classes, const char** class_names, 
              float point_size, const char* title)
{
    signal(SIGINT, ctrlc); /* matplotcpp hijacks this */
    printf("Plotting results - close plot window to continue\n");

    plt::figure_size(800, 600);

    for (int i = 0; i < n_classes; i++) {
        std::vector<double> x1, x2;
        for (int j = 0; j < len; j++) {
            if (y[j] == i) {
                x1.push_back(x[j][0]);
                x2.push_back(x[j][1]);
            }
        }
        plt::scatter(x1, x2, point_size, {{ "label", class_names[i] }});
    }

    plt::xlabel("Principal Component 1");
    plt::ylabel("Principal Component 2");
    plt::title(title);
    plt::legend();
    plt::grid(true);

    try {
        plt::show();
    }
    catch(...) {
    }
    signal(SIGINT, ctrlc); /* matplotcpp hijacks this */
}
