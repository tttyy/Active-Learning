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
	int d, t, cor;
	double *v;

	double dotProduct(double[], double[], int);

public:
	Perceptron();
	Perceptron(int dim);
	~Perceptron();

	double read(double [], int);
	bool predict(double [], int);
};

#endif /* PERCEPTRON_H_ */
