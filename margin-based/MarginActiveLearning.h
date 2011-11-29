/*
 *  MarginActiveLearning.h
 *  margin-based-active-learning
 *
 *  Created by Chen Liu on 11/13/11.
 *  Copyright 2011 University of California, Los Angeles. All rights reserved.
 *
 */

#ifndef MARGIN_ACTIVE_LEARNING_H
#define MARGIN_ACTIVE_LEARNING_H

#include <vector>
#include "../DataPoint.h"

class MarginActiveLearning {
private:
    int dimension;
    double C;
    int n_label;
    double* weight;
    int n_iteration, k;
    double epsilon, delta;
    std::vector<DataPoint> working_set;
    
    void update_weight(bool separable); //train on current working set to update weight
     //In the case of unseparable, train on current working set to find an approximate
     //optimal solution of weight
    

public:
    //constructor
    MarginActiveLearning(int d, double C, double eps, double delt);
    ~MarginActiveLearning();
    
    //methods
    int classify(DataPoint point); //classify a data point, +1 or -1
    double margin(DataPoint point); //compute the margin of current datapoint
    
    /**
     * train the model in separable scenario
     */
    bool build_model_separable_iter(std::vector<DataPoint> &data_vec);
    void build_model_separable(std::vector<DataPoint> &data_vec);
    bool build_model_unseparable_iter(std::vector<DataPoint> &data_vec, double, double);
    void build_model_unseparable(std::vector<DataPoint> &data_vec, double, double);
	void set_niter_for_unseparable(double);
    int getNumberOfLabel();
};

#endif
