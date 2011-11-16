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

DataPoint::DataPoint(int d, double* xvec, int l)
{
    this->dimension = d;
    this->x = new double[d];
    for (int i = 0; i < d; i++) {
        this->x[i] = xvec[i];
    }
    this->label = l;
};

DataPoint::DataPoint(const DataPoint &dp)
{
	this->dimension = dp.dimension;
	this->x = new double[dp.dimension];
	for (int i = 0; i < dp.dimension; i++) {
		this->x[i] = dp.x[i];
    }
	this->label = dp.label;
}

DataPoint::~DataPoint()
{
    delete[] this->x;
}

DataPoint DataPoint::clone()
{
    return DataPoint(this->dimension, this->x, this->label);
}
