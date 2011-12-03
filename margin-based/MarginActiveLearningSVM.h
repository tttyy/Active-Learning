/*
 *  MarginActiveLearningSVM.h
 *  active-learning-local3
 *
 *  Created by Chen Liu on 11/28/11.
 *  Copyright 2011 University of California, Los Angeles. All rights reserved.
 *
 */

#ifndef MARGIN_ACTIVE_LEARNING_SVM_H
#define MARGIN_ACTIVE_LEARNING_SVM_H

#include"svm.h"
#include <vector>
#include "../DataPoint.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class MarginActiveLearningSVM {
private:
    int dimension;
    int n_label;
    //double* weight;
    int n_iteration, k;
    double epsilon, delta;
    std::vector<DataPoint> working_set;
//    struct svm_model *model;
    struct svm_parameter param;
//    struct svm_problem prob;
//    struct svm_node* xspace;
    int x_space_size;
    double *weight;
    double rho;
    
    void update_weight(bool separable); //train on current working set to update weight
    //In the case of unseparable, train on current working set to find an approximate
    //optimal solution of weight
    
    
    
    
public:
    double C;
    
    //constructor
    MarginActiveLearningSVM(int d, int C, double eps, double delt);
    ~MarginActiveLearningSVM();
    
    //methods
    int classify(DataPoint &point); //classify a data point, +1 or -1
    double margin(DataPoint &point); //compute the margin of current datapoint
    struct svm_node* convert_DataPoint_to_svm_node(DataPoint &x);
    
    /**
     * train the model in separable scenario
     */
    bool build_model_separable_iter(std::vector<DataPoint> &data_vec);
    void build_model_separable(std::vector<DataPoint> &data_vec);
    bool build_model_unseparable_iter(std::vector<DataPoint> &data_vec, double, double);
    void build_model_unseparable(std::vector<DataPoint> &data_vec, double, double);
    int getNumberOfLabel();
};



#endif