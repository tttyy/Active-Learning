/*
 *  DataPoint.cpp
 *  margin-based-active-learning
 *
 *  Created by Chen Liu on 11/13/11.
 *  Copyright 2011 University of California, Los Angeles. All rights reserved.
 *
 */

#include "DataPoint.h"
#include <cstdlib>

DataPoint::DataPoint(int d, double* xvec)
{
    this->dimension = d;
    this->x = new double[d];
    for (int i = 0; i < d; i++) {
        this->x[i] = xvec[i];
    }
};

DataPoint::~DataPoint()
{
    delete[] this->x;
    this->x = NULL;
}

DataPoint DataPoint::clone()
{
    return DataPoint(this->dimension, this->x);
}