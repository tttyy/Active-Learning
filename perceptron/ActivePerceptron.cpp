#include "ActivePerceptron.h"
#include <math.h>
#include <stdio.h>

ActivePerceptron::ActivePerceptron(int dim, int label, int pat)
{
	d = dim;
	L = label;
	R = pat;
	s = 0;
	v = new double[d];
	t = 0;
	con = 0;	// Number of consistently correct labels
	num = 0;	// Number of requested labels
}


ActivePerceptron::~ActivePerceptron(void)
{
}

bool ActivePerceptron::read(double x[], int y) {
	t++;
	double p;
	if (t==1)
	{
		for (int i=0;i<d;i++) v[i]=x[i]*y;
		s = 1/ sqrt((double)d);
		con ++;
		num++;
		return true;
	}
	else
	{
		p = dotProduct(x,v,d);
		if (num<L && abs(p)<=s)
		{
			num++;
			if (p * y < 0)
			{
				con = 0;
				for (int i=0;i<d;i++)
					v[i]-=2*p*x[i];
				return false;
			}
			else
			{
				con++;
				if (con == R)
				{
					s/=2;
					con = 0;
				}
				return true;
			}
		}
		else
		{
			return predict(x, y);
		}
	}
}

int ActivePerceptron::computeL(int d, double delta, double epsilon, double c)
{
	return c*(double)d*log(1/delta/epsilon)*(log((double)d/delta)+log(log(1/epsilon)));
}

int ActivePerceptron::computeR(int d, double delta, double epsilon, double c)
{
	return c*log((double)d/delta)+log(log(1/epsilon));
}

int ActivePerceptron::getNumberOfLabel()
{
	return num;
}