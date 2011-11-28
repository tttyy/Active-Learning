/*
 *  DataPoint.h
 *  margin-based-active-learning
 *
 *  Created by Chen Liu on 11/13/11.
 *  Copyright 2011 University of California, Los Angeles. All rights reserved.
 *
 */

#include <map>
using namespace std;

#ifndef DATA_POINT_H
#define DATA_POINT_H

struct DataPoint {
    int dimension;
	int label;
	bool useMap;
    double* x;
	map<const int,double> xMap;
    DataPoint(int d, double* xvec, int l);
	DataPoint(const DataPoint &dp);
	DataPoint(int d);
	void addComp(int, double);
    ~DataPoint();
    DataPoint clone();
};

typedef struct DataPoint DataPoint;

#endif