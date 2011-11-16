/*
 *  MarginActiveLearning.cpp
 *  margin-based-active-learning
 *
 *  Created by Chen Liu on 11/13/11.
 *  Copyright 2011 University of California, Los Angeles. All rights reserved.
 *
 */


#define _USE_MATH_DEFINES

#include "MarginActiveLearning.h"
#include "../DataPoint.h"
#include "linear.h"
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iterator>

MarginActiveLearning::MarginActiveLearning(int d, int C, double eps, double delt)
{
    this->dimension = d;
    this->n_label = 0;
    this->weight = new double[d];
    this->n_iteration = (int)ceil(log(1 / eps) / log(2.0));
    this->C = C;
    this->epsilon = eps;
    this->delta = delt;
    this->working_set = std::vector<DataPoint> ();
};

MarginActiveLearning::~MarginActiveLearning()
{
    delete [] this->weight;
};

int MarginActiveLearning::classify(DataPoint point)
{
    if (point.dimension != this->dimension) {
        fprintf(stderr, "Error! Dimension does not match!");
        exit(1);
    }
    double dot_prod = 0.0;
    for (int i = 0; i < this->dimension; i++) {
        dot_prod += this->weight[i] * point.x[i];
    }
    return dot_prod >=0 ? 1 : -1;
};

double MarginActiveLearning::margin(DataPoint point)
{
    if (point.dimension != this->dimension) {
        fprintf(stderr, "Error! Dimension does not match!");
        exit(1);
    }
    double dot_prod = 0.0;
    for (int i =0; i < this->dimension; i++) {
        dot_prod += this->weight[i] * point.x[i];
    }
    return fabs(dot_prod);
};

void MarginActiveLearning::update_weight()
{
    struct problem prob;
    prob.l = this->working_set.size();
    prob.y = Malloc(int, prob.l);
    prob.x = Malloc(struct feature_node*, prob.l);
    feature_node* x_space = Malloc(struct feature_node, prob.l * (dimension+1) );
    
    int j = 0;
    for (int i = 0; i < prob.l; i++) {
        prob.x[i] = &x_space[j];
        prob.y[i] = this->working_set[i].label;
        for (int k = 0; k < dimension; k++) {
            x_space[j].index = k;
            x_space[j].value = this->working_set[i].x[k];
        }
        x_space[dimension].index = -1;
    }
    
    // default values
    struct parameter *param = Malloc(struct parameter, 1);
    param->solver_type = L2R_L2LOSS_SVC_DUAL;
    param->C = 1;
    param->nr_weight = 0;
    param->weight = NULL;
    param->weight_label = NULL;
    
    struct model *model = train(&prob, param);
    for (int i = 0; i < dimension; i++) {
        this->weight[i] = model->w[i];
    }
    free_model_content(model);
    destroy_param(param);
};

/**
 * This function will perform one more iteration of training
 */
void MarginActiveLearning::build_model_separable_iter(std::vector<DataPoint> &data_vec)
{
    if(this->k >= n_iteration)
        return;
    double d = (double) this->dimension;
    int m = (int)(C * sqrt(d) * (d * log(d) + log(this->k / this->delta)));
    double b = M_PI / pow(2.0, this->k - 1);
    this->working_set = std::vector<DataPoint>();

    for(int i = 0; i < data_vec.size(); i++) {
        /**
         * Try to add a DataPoint point. If the margin of point is less than b,
         * then include the point into working_sets and ask for a label. Otherwise,
         * drop the data point
        */
        if(this->margin(data_vec[i]) < b) {
            this->working_set.push_back(data_vec[i]);
        }
        if(this->working_set.size() > m)
            break;
    }
    this->k += 1;
    update_weight();
}

/**
 * This function will train a complete model in one time
 */
void MarginActiveLearning::build_model_separable(std::vector<DataPoint> &data_vec)
{
    while(this->k < n_iteration) {
        std::random_shuffle(data_vec.begin(), data_vec.end());
        this->build_model_separable_iter(data_vec);
    }
}
