/*
 * Perceptron.cpp
 *
 *  Created on: 2011-11-3
 *      Author: tttyy
 */

#include "Perceptron.h"
#include <math.h>
#include <stdio.h>
#include "../helpers.h"

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

double Perceptron::dotProduct(DataPoint dp, double b[])
{
	double sum = 0;
	if (!dp.useMap)
	{
		for (int i=0;i<dp.dimension;i++)
			sum += dp.x[i]*b[i];
	}
	else
	{
		map<const int, double>::iterator iter = dp.xMap.begin();
		while (iter != dp.xMap.end())
		{
			sum += iter->second * b[iter->first-1];
			iter++;
		}
	}
	
	return sum;
}

bool Perceptron::read(DataPoint dp) {
	t++;
	DataPoint dp2(dp);
	normalize(dp2);
	double p = dotProduct(dp2,v);
	if ((p>=0 && dp2.label<0) || (p<0 && dp2.label>0))
	{
		if (t<=L)
		{
			if (!dp2.useMap)
			{
				for (int i=0;i<d;i++)
					v[i]+=dp2.label*dp2.x[i];
			}
			else
			{
				map<const int, double>::iterator iter = dp.xMap.begin();
				while (iter != dp.xMap.end())
				{
					v[iter->first-1]+=dp2.label*iter->second;
					iter++;
				}
			}
		}
		return false;
	}
	else
	{
		return true;
	}
}

bool Perceptron::predict(DataPoint dp) {
	double p = dotProduct(dp,v);
	return (p>=0 && dp.label==1)||(p<0 && dp.label == -1);
}

void Perceptron::setL(int in)
{
	this->L = in;
}

void Perceptron::setT(int t)
{
    this->t = t;
}

double * Perceptron::getWeight()
{
	return v;
}