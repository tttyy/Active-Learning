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
	cor = 0;	// Number of correct predictions
}


ActivePerceptron::~ActivePerceptron(void)
{
}

double ActivePerceptron::read(double x[], int y) {
	t++;
	if (t==1)
	{
		for (int i=0;i<d;i++) v[i]=x[i]*y;
		s = 1/ sqrt((double)d);
		con ++;
		num++;
		cor++;
	}
	else
	{
		double p = dotProduct(x,v,d);
		if (num<L && abs(p)<=s)
		{
			num++;
			if (p * y < 0)
			{
				con = 0;
				for (int i=0;i<d;i++)
					v[i]-=2*p*x[i];
			}
			else
			{
				con++;
				cor++;
				if (con == R)
				{
					s/=2;
					con = 0;
				}
			}
		}
		else
		{
			if (predict(x, y))
				cor++;
		}
	}
	return (double)cor/(double)t;
}