/*
 *  MarginActiveLearning.cpp
 *  margin-based-active-learning
 *
 *  Created by Chen Liu on 11/13/11.
 *  Copyright 2011 University of California, Los Angeles. All rights reserved.
 *
 */


#define _USE_MATH_DEFINES

#include "MarginActiveLearningSVM.h"
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
#include <cassert>
#include <map>

MarginActiveLearningSVM::MarginActiveLearningSVM(int d, int C, double eps, double delt)
{
    this->dimension = d;
    this->n_label = 0;
    this->C = C;
    this->epsilon = eps;
    this->delta = delt;
	this->k = 1;
    weight = (double*)malloc(dimension * sizeof(double) );
    for (int i = 0; i < dimension; i++) {
        weight[i] = 1.0;
    }
    normalize(weight, dimension);
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.nr_weight = 0;
    param.weight = NULL;
    param.weight_label = NULL;
    param.probability = 0;
    rho = 0.0;
    x_space_size = 0;
    working_set = std::vector<DataPoint>();
};

MarginActiveLearningSVM::~MarginActiveLearningSVM()
{
    svm_destroy_param(&param);
    free(weight);
    
};

struct svm_node* MarginActiveLearningSVM::convert_DataPoint_to_svm_node(DataPoint &pt)
{
    struct svm_node* node = NULL;
    int j = 0;
    if (pt.useMap) {
        node = Malloc(struct svm_node, pt.xMap.size()+1);
        for (std::map<const int, double>::iterator iter=pt.xMap.begin(); iter!=pt.xMap.end(); iter++) {
            node[j].index = iter->first;
            node[j].index = iter->second;
            j++;
        }
    } else {
        for (int i = 0; i < pt.dimension; i++) {
            if(fabs(pt.x[i]) > 1e-6 )
                j++;
        }
        node = Malloc(struct svm_node, j+1);
        j = 0;
        for (int i = 0; i < pt.dimension; i++) {
            if(fabs(pt.x[i]) > 1e-6 ) {
                node[j].index = i;
                node[j].value = pt.x[i];
                j++;
            }
        }
    }
    node[j].index = -1;
    return node;
}


int MarginActiveLearningSVM::classify(DataPoint &point)
{
    double dot_prod = 0.0;
    for (int i =0; i < this->dimension; i++) {
        dot_prod += this->weight[i] * point.x[i];
    }
    dot_prod -= rho;
    return dot_prod >=0 ? 1 : -1;
};

double MarginActiveLearningSVM::margin(DataPoint &point)
{
    double dot_prod = 0.0;
    for (int i =0; i < this->dimension; i++) {
        dot_prod += this->weight[i] * point.x[i];
    }
    return fabs(dot_prod);
    
};

void MarginActiveLearningSVM::update_weight(bool separable)
{
    struct svm_problem prob;
    struct svm_node* xspace;
    
    int nnz = 0;
    for (std::vector<DataPoint>::iterator iter = working_set.begin(); iter != working_set.end() ; iter++) {
        nnz += iter->nnz() + 1;
    }
    
    prob.l = working_set.size();
    prob.x = (struct svm_node**)malloc(prob.l * sizeof(struct svm_node*));
    prob.y = (double *)malloc(prob.l * sizeof(double));
    xspace = (struct svm_node*)malloc(nnz * sizeof(struct svm_node));
    
    int ix = 0, i = 0;
    for (std::vector<DataPoint>::iterator iter = working_set.begin(); iter != working_set.end() ; iter++) {
        prob.y[i] = iter->label;
        prob.x[i] = &(xspace[ix]);
        if (iter->useMap) {
            for (map<const int, double>::iterator it=iter->xMap.begin(); it!=iter->xMap.end(); it++) {
                xspace[ix].index = it->first;
                xspace[ix].value = it->second;
                ix++;
            }
        } else {
            for (int j = 0; j < iter->dimension; j++) {
                xspace[ix].index = j;
                xspace[ix].value = iter->x[j];
                ix++;
            }
        }
        xspace[ix].index = -1;
        ix++;
        i++;
    }
    
    struct svm_model* model = svm_train(&prob, &param);
    double *coef = model->sv_coef[0];
    for (int i = 0; i < dimension; i++) {
        weight[i] = 0.0;
    }
    
    for (int i = 0; i < model->l; i++) {
        svm_node* sv = model->SV[i];
        int j = 0;
        while (sv[j].index != -1) {
            weight[sv[j].index] += coef[i] * sv[j].value;
            j++;
        }
        
    }
    
    double weight_norm = 0.0;
    for (int i = 0; i < dimension; i++) {
        weight_norm += weight[i] * weight[i];
    }
    weight_norm = sqrt(weight_norm);
    for (int i = 0; i < dimension; i++) {
        weight[i] /= weight_norm;
    }
    rho = model->rho[0] / weight_norm;
    free(prob.x);
    free(xspace);
    free(prob.y);
    svm_free_and_destroy_model(&model);
    
};

/**
 * This function will perform one more iteration of training
 */
bool MarginActiveLearningSVM::build_model_separable_iter(std::vector<DataPoint> &data_vec)
{
    this->n_iteration = (int)ceil(log(1 / epsilon) / log(2.0));
    if(this->k > n_iteration)
        return false;
	std::vector<int> indexVec;
	for (int i=0;i<data_vec.size();i++)
		indexVec.push_back(i);
	std::random_shuffle(indexVec.begin(), indexVec.end());
    
    printf("\nIteration %d:", k);
    
    double d = (double) this->dimension;
    int m = (int)(C * sqrt(d) * (d * log(d) + log(this->k / this->delta)));
    if (k == 1) {
        printf("m:%d\t\tk:%d\n", m, k);
        for (int i = 0; i < m; i++) {
            DataPoint pt = data_vec[indexVec[i]];
            working_set.push_back(pt);
            n_label++;
        }
    }
    else {
        double b = M_PI / pow(2.0, this->k - 1);
        printf("m:%d\tb:%lf\tk:%d\n", m,b,k);
        bool sampling_done = false;
        bool get_a_sample = false;
        for(int i = 0; sampling_done == false; i++) {
            /**
             * Try to add a DataPoint point. If the margin of point is less than b,
             * then include the point into working_sets and ask for a label. Otherwise,
             * drop the data point
             */
            DataPoint dp = data_vec[indexVec[i]];
            if(this->margin(dp) < b) {
                working_set.push_back(dp);
                n_label++;
                get_a_sample = true;
            }
            if(n_label == m)
                sampling_done = true;
            if (i == data_vec.size() ) {
                if (get_a_sample) {
                    i = 0;
                    std::random_shuffle(indexVec.begin(), indexVec.end());
                    get_a_sample = false;
                }
                else {
                    sampling_done = true;
                }
                
            }
        }
    }
    this->k += 1;
    update_weight(true);
    double sum = 0.0;
    for (std::vector<DataPoint>::iterator it = working_set.begin(); it != working_set.end(); it++) {
        if (classify(*it) != it->label)
            sum++;
    }
    printf("training error: %lf\n", sum);

	return true;
}

/**
 * This function will train a complete model in one time
 */
void MarginActiveLearningSVM::build_model_separable(std::vector<DataPoint> &data_vec)
{
    this->n_iteration = (int)ceil(log(1 / epsilon) / log(2.0));
    while(this->k < n_iteration) {
        this->build_model_separable_iter(data_vec);
    }
}

int MarginActiveLearningSVM::getNumberOfLabel()
{
	return n_label;
}

/**
 * This function will perform one more iteration of training
 */
bool MarginActiveLearningSVM::build_model_unseparable_iter(std::vector<DataPoint> &data_vec, double alpha, double beta)
{
    
    printf("\nIteration %d:", k);
    
    n_iteration = (int)ceil(log(beta / this->epsilon) / log(2.0));
    if(this->k > n_iteration)
        return false;
	std::vector<int> indexVec;
	for (int i=0;i<data_vec.size();i++)
		indexVec.push_back(i);
	std::random_shuffle(indexVec.begin(), indexVec.end());
    
    working_set.clear();
    
    double d = (double) this->dimension;
    double b = pow(2.0, (alpha - 1) * k) * M_PI * pow(d, -0.5) * sqrt(5+alpha * k * log(beta) + log(2.0 + k));
    double e = pow(2.0, alpha * (1 - k) - 4) * beta / sqrt(5 + alpha * k * log(2.0) - log(beta) + log(1.0 + k));
    double m = ceil(C * pow(e, -2.0) * (d + log(k / delta)));
    printf("m:%lf    b:%lf   k:%d  \n", m, b, k);
    
    if (k == 1) {
        for (int i = 0; i < m && i < data_vec.size(); i++) {
            DataPoint dp = data_vec[indexVec[i]];
            working_set.push_back(dp);
            n_label += 1;
        }
    }
    else{
        
        int n_labeled = 0;
        bool sampling_done = false;
        bool get_a_sample = false;
        int wrong_label = 0;
        for(int i = 0; i < data_vec.size() && sampling_done == false; i++) {
            /**
             * Try to add a DataPoint point. If the margin of point is less than b,
             * then include the point into working_sets and ask for a label. Otherwise,
             * drop the data point
             */
            DataPoint dp = data_vec[indexVec[i]];
            if(this->margin(dp) < b) {
                working_set.push_back(dp);
                n_label++;
                n_labeled++;
                get_a_sample = true;
            } else {
                int label = this->classify(dp);
                if (label != dp.label) {
                    wrong_label++;
                    this->classify(dp);
                }
                dp.label = label;
                working_set.push_back(dp);
            }
            
            if(n_labeled >= m)
                sampling_done = true;
            /*        if (i == data_vec.size() ) {
             if (get_a_sample) {
             i = 0;
             std::random_shuffle(indexVec.begin(), indexVec.end());
             get_a_sample = false;
             }
             else {
             sampling_done = true;
             }
             
             }*/
        }
        printf("wrong label: %d\n", wrong_label);
        printf("actual working set size: %d\n", working_set.size());
    }
    k += 1;
    update_weight(false);
	return true;
}


void MarginActiveLearningSVM::build_model_unseparable(std::vector<DataPoint> &data_vec, double alpha, double beta)
{
    this->n_iteration = (int)ceil(log(beta / this->epsilon) / log(2.0));
    while(this->k < n_iteration) {
        this->build_model_unseparable_iter(data_vec, alpha, beta);
    }
}
/*
 void MarginActiveLearningSVM::set_niter_for_unseparable(double beta)
 {
 this->n_iteration = (int) ceil(log(beta / this->epsilon) / log(2.0));
 }*/
