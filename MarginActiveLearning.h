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
#include "DataPoint.h"

class MarginActiveLearning {
public:
    int dimension;
    int n_label;
    double* weight;
    std::vector<DataPoint> working_sets;
    
    //constructor
    MarginActiveLearning(int d);
    ~MarginActiveLearning();
    
    //methods
    int classify(DataPoint point); //classify a data point, +1 or -1
    double margin(DataPoint point); //compute the margin of current datapoint
    
    /**
     * Try to add a DataPoint point. If the margin of point is less than b,
     * then include the point into working_sets and ask for a label. Otherwise,
     * drop the data point
     */
    bool add_point(DataPoint point, double b);
    
    void update_weight(); //train on current working set to update weight
    
    /**
     * train the model in separable scenario
     */
    void build_model_separable(std::vector<DataPoint> data_vec, double epsilon, double delta, int C);
    
};

#endif