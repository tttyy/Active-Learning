#include "ActivePerceptron.h"
#include <math.h>
#include <stdio.h>
#include "../helpers.h"

ActivePerceptron::ActivePerceptron(int dim, int label, int pat)
{
	d = dim;
	L = label;
	R = pat;
	s = 0;
	v = new double[d];
	for (int i=0;i<d;i++) v[i]=0;
	t = 0;
	con = 0;	// Number of consistently correct labels
	num = 0;	// Number of requested labels
}


ActivePerceptron::~ActivePerceptron(void)
{
}

bool ActivePerceptron::read(DataPoint dp) {
	DataPoint dp2(dp);
	normalize(dp2);
	t++;
	double p;
	if (t==1)
	{
		if (!dp2.useMap)
			for (int i=0;i<d;i++) v[i]=dp2.x[i]*dp2.label;
		else
			for (map<const int, double>::iterator iter=dp2.xMap.begin();iter!=dp2.xMap.end();iter++)
				v[iter->first-1] = iter->second*dp.label;
		s = 200/ sqrt((double)d);
		con ++;
		num++;
		return true;
	}
	else
	{
		p = dotProduct(dp2,v);
		if (num<L && abs(p)<=s)
		{
			num++;
			if ((p>=0 && dp2.label<0) || (p<0 && dp2.label>0))
			{
				con = 0;
				if (!dp2.useMap)
					for (int i=0;i<d;i++)
						v[i]-=2*p*dp2.x[i];
				else
					for (map<const int, double>::iterator iter=dp2.xMap.begin();iter!=dp2.xMap.end();iter++)
						v[iter->first-1] -= 2*p*iter->second;
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
			return predict(dp2);
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