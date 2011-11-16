/*
 * Perceptron.cpp
 *
 *  Created on: 2011-11-3
 *      Author: tttyy
 */

#include "Perceptron.h"
#include <math.h>
#include <stdio.h>

Perceptron::Perceptron() {}

Perceptron::Perceptron(int dim, int label) {
	// TODO Auto-generated constructor stub
	d = dim;
	L = label;
	v = new double[d];
	for (int i=0;i<d;i++) v[i]=0;
	t = 0;
}

Perceptron::~Perceptron() {
	// TODO Auto-generated destructor stub
	delete(v);
}

double Perceptron::dotProduct(double a[], double b[], int n)
{
	double sum = 0;
	for (int i=0;i<n;i++)
		sum += a[i]*b[i];
	return sum;
}

bool Perceptron::read(double x[], int y) {
	t++;

	double p = dotProduct(x,v,d);
	if ((p>=0 && y<0) || (p<0 && y>0))
	{
		if (t<=L)
			for (int i=0;i<d;i++)
				v[i]+=y*x[i];
		return false;
	}
	else
	{
		return true;
	}
}

bool Perceptron::predict(double x[], int y) {
	return dotProduct(x,v,d)*y>=0;
}

void Perceptron::setL(int in)
{
	this->L = in;
}