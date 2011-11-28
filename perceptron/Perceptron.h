/*
 * Perceptron.h
 *
 *  Created on: 2011-11-3
 *      Author: tttyy
 */

#include "../DataPoint.h"

#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

class Perceptron {
protected:
	int d, t, L;
	double *v;

	double dotProduct(DataPoint dp, double[]);

public:
	Perceptron();
	Perceptron(int dim, int label);
	~Perceptron();

	bool read(DataPoint dp);
	bool predict(DataPoint dp);
	void setL(int in);
    void setT(int t);
	double * getWeight();
};

#endif /* PERCEPTRON_H_ */
