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
#include "../perceptron/Perceptron.h"
#include "../helpers.h"
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
	for (int i=0;i<d;i++)
		this->weight[i] = 1/sqrt((double)d);
    this->C = C;
    this->epsilon = eps;
    this->delta = delt;
    this->working_set = std::vector<DataPoint> ();
	this->k = 1;
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

void MarginActiveLearning::update_weight(bool separable)
{

	Perceptron *perc = new Perceptron(dimension, this->working_set.size());
    bool converged = false;
    while(converged == false) {
	    for (int i=0;i<this->working_set.size();i++)
	    {
		    perc->read(this->working_set[i].x, this->working_set[i].label);
	    }
	    for (int i = 0; i < dimension; i++) {
		    this->weight[i] = perc->getWeight()[i];
        }
	    normalize(this->weight, dimension);
        converged = true;
        if (separable) {
            for (int i = 0; i < this->working_set.size(); i++)
            {
                if(converged && this->classify(working_set[i]) != this->working_set[i].label) {
                    converged = false;
                    perc->setT(0);
                }
            }
        }
    }
};

/**
 * This function will perform one more iteration of training
 */
bool MarginActiveLearning::build_model_separable_iter(std::vector<DataPoint> &data_vec)
{
    if(this->k >= n_iteration)
        return false;
	std::vector<int> indexVec;
	for (int i=0;i<data_vec.size();i++)
		indexVec.push_back(i);
	std::random_shuffle(indexVec.begin(), indexVec.end());

    double d = (double) this->dimension;
    int m = (int)(C * sqrt(d) * (d * log(d) + log(this->k / this->delta)));
    double b = M_PI / pow(2.0, this->k - 1);

    int n_labeled = 0;
    for(int i = 0; i < data_vec.size(); i++) {
        /**
         * Try to add a DataPoint point. If the margin of point is less than b,
         * then include the point into working_sets and ask for a label. Otherwise,
         * drop the data point
        */
		DataPoint dp = data_vec[indexVec[i]];
        if(this->margin(dp) < b) {
            this->working_set.push_back(dp);
			n_labeled ++;
            n_label++;
        }
        if(n_labeled > m)
            break;
    }
    this->k += 1;
    update_weight(true);

	return true;
}

/**
 * This function will train a complete model in one time
 */
void MarginActiveLearning::build_model_separable(std::vector<DataPoint> &data_vec)
{
    this->n_iteration = (int)ceil(log(1 / epsilon) / log(2.0));
    while(this->k < n_iteration) {
        this->build_model_separable_iter(data_vec);
    }
}

int MarginActiveLearning::getNumberOfLabel()
{
	return n_label;
}

/**
 * This function will perform one more iteration of training
 */
bool MarginActiveLearning::build_model_unseparable_iter(std::vector<DataPoint> &data_vec, double alpha, double beta)
{
    if(this->k >= n_iteration)
        return false;
	std::vector<int> indexVec;
	for (int i=0;i<data_vec.size();i++)
		indexVec.push_back(i);
	std::random_shuffle(indexVec.begin(), indexVec.end());
    
    double d = (double) this->dimension;
    double b = pow(2.0, (alpha - 1) * k) * M_PI * pow(d, -0.5) * sqrt(5+alpha * k * log(beta) + log(2.0 + k));
    double e = pow(2.0, alpha * (1 - k) - 4) * beta / sqrt(5 + alpha * k * log(2.0) - log(beta) + log(1.0 + k));
    double m = C * pow(e, -2.0) * (d + log(k / delta));
    
    int n_labeled = 0;
    for(int i = 0; i < data_vec.size(); i++) {
        /**
         * Try to add a DataPoint point. If the margin of point is less than b,
         * then include the point into working_sets and ask for a label. Otherwise,
         * include the point into working_set with automatic label.
         */
		DataPoint dp = data_vec[indexVec[i]];
        if(this->margin(dp) < b) {
            this->working_set.push_back(dp);
			n_labeled ++;
            n_label++;
        } else {
            dp.label = this->classify(dp);
            this->working_set.push_back(dp);
        }

        if(n_labeled > m)
            break;
    }
    this->k += 1;
    update_weight(false);
    this->working_set.clear();
	return true;
}

void MarginActiveLearning::build_model_unseparable(std::vector<DataPoint> &data_vec, double alpha, double beta)
{
    this->n_iteration = (int)ceil(log(beta / this->epsilon) / log(2.0));
    while(this->k < n_iteration) {
        this->build_model_unseparable_iter(data_vec, alpha, beta);
    }
}
