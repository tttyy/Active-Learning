#pragma once
#include "Perceptron.h"
class ActivePerceptron :
	public Perceptron
{
private:
	int L, R, con, num;
	double s;

public:
	ActivePerceptron(int dim, int label, int pat);
	~ActivePerceptron(void);

	double read(double [], int);
};

