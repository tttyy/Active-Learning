/*
 *  MarginActiveLearning.cpp
 *  margin-based-active-learning
 *
 *  Created by Chen Liu on 11/13/11.
 *  Copyright 2011 University of California, Los Angeles. All rights reserved.
 *
 */

#include "MarginActiveLearning.h"
#include "DataPoint.h"
#include "linear.h"
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iterator>

#define _USE_MATH_DEFINES

MarginActiveLearning::MarginActiveLearning(int d)
{
    this->dimension = d;
    this->n_label = 0;
    this->weight = new double[d];
    this->working_set = std::vector<DataPoint>();
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

bool MarginActiveLearning::add_point(DataPoint point, double b)
{
    double margin = this->margin(point);
    if (margin < b) {
        this->working_set.push_back(point);
        return true;
    }
    return false;
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

void MarginActiveLearning::build_model_separable(std::vector<DataPoint> data_vec, double epsilon, double delta, int C)
{
    srand(time(NULL));
    int n = data_vec.size();
    int n_iteration = round(log(1 / epsilon) / log(2.0) + 0.5);
    for (int k = 1; k <= n_iteration; k++) {
        this->update_weight();
        
        double d = (double) this->dimension;
        int m = C * sqrt(d) * (d * log(d) + log(k / delta));
        double b = M_PI / pow(2.0, k-1);

	int n_labeled = 0;
        std::random_shuffle(data_vec.begin(), data_vec.end());
	int j = 0;
        while(1) {
	    if(n_labeled > m)
		    break;
            if (this->add_point(data_vec[j], b))
		    n_labeled++;
	    if (j == data_vec.size() )
		    j = 0;
	    else
		    j++;
	    
        }
        
    }
}
