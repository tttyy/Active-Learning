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
	int getNumberOfLabel();

	static int computeL(int d, double delta, double epsilon, double c);
	static int computeR(int d, double delta, double epsilon, double c);
};

