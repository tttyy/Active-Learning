/*
 *  DataPoint.h
 *  margin-based-active-learning
 *
 *  Created by Chen Liu on 11/13/11.
 *  Copyright 2011 University of California, Los Angeles. All rights reserved.
 *
 */

#ifndef DATA_POINT_H
#define DATA_POINT_H

class DataPoint {
public:
    int dimension;
    double* x;
    int label;
    DataPoint(int d, double* xvec, int l);
    ~DataPoint();
    DataPoint clone();
};

#endif