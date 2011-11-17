/*
 * Perceptron.h
 *
 *  Created on: 2011-11-3
 *      Author: tttyy
 */

#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

class Perceptron {
protected:
	int d, t, L;
	double *v;

	double dotProduct(double[], double[], int);

public:
	Perceptron();
	Perceptron(int dim, int label);
	~Perceptron();

	bool read(double [], int);
	bool predict(double [], int);
	void setL(int in);
	double * getWeight();
};

#endif /* PERCEPTRON_H_ */
