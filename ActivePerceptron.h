#pragma once
#include "Perceptron.h"
class ActivePerceptron :
	public Perceptron
{
private:
	int R, con, num;
	double s;

public:
	ActivePerceptron(int dim, int label, int pat);
	~ActivePerceptron(void);

	bool read(double [], int);
};

