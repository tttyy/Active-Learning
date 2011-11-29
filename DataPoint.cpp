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
	this->useMap = false;
};

DataPoint::DataPoint(const DataPoint &dp)
{
	this->dimension = dp.dimension;
	if (!(this->useMap = dp.useMap))
	{
		this->x = new double[dp.dimension];
		for (int i = 0; i < dp.dimension; i++) {
			this->x[i] = dp.x[i];
		}
	}
	else
	{
		this->xMap = map<const int, double>(dp.xMap);
	}
	this->label = dp.label;
	
}

DataPoint::DataPoint(int d)
{
	this->dimension = d;
	this->useMap = true;
}

DataPoint::~DataPoint()
{
	if (!this->useMap)
    delete[] this->x;
}

DataPoint DataPoint::clone()
{
    return DataPoint(*this);
}

void DataPoint::addComp(int a, double b)
{
	if (a == -1)
		this->label = b;
	else
		this->xMap[a]=b;
}

int DataPoint::nnz()
{
    if (this->useMap) {
        return xMap.size();
    }
    else
        return this->dimension;
}