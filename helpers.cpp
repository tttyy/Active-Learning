#include "helpers.h"
#include "stdlib.h"
#include "math.h"
#include "Tokenizer.h"

// Generate an d-dimensional array of random numbers between -0.5 and 0.5
// Assuming that the seed is setup elsewhere
double* arrayGen(int d)
{
	double *a = new double[d];
	for (int i=0;i<d;i++)
	{
		a[i] = rand()/double(RAND_MAX)-0.5;
	}
	return a;
}

// Using a linear threshold function to classify input vector.
// The seperator line is sum(n*x(n))=0
int classify(double x[], int d)
{
	double sum = 0;
	for (int i=0;i<d;i++)
		sum+= (i+1)*x[i];
	if (sum>=0)
		return 1;
	else
		return -1;
}

void normalize(double x[], int d)
{
	double sum = 0;
	for (int i=0;i<d;i++)
		sum+=x[i]*x[i];
	sum = sqrt(sum);
	for (int i=0;i<d;i++)
		x[i]/=sum;
}
void normalize(DataPoint dp)
{
	if (!dp.useMap)
	{
		normalize(dp.x,dp.dimension);
	}
	else
	{
		double sum = 0;
		map<const int, double>::iterator iter = dp.xMap.begin();
		while (iter!= dp.xMap.end())
		{
			sum+=iter->second * iter->second;
			iter++;
		}
		sum = sqrt(sum);
		iter = dp.xMap.begin();
		while (iter!= dp.xMap.end())
		{
			iter->second /= sum;
			iter++;
		}
	}
}

DataPoint readData(string str, int d, bool isMap)
{
	Tokenizer nizer;
	nizer.set(str);
	nizer.setDelimiter(",");
	int i=0;
	if (!isMap)
	{
		double *x = new double[d+1];
		string tmp;
		while ((tmp=nizer.next())!="")
		{
			x[i]=atof(tmp.c_str());
			i++;
		}
		DataPoint dp(d, x, x[d]);
		return dp;
	}
	else
	{
		int dimen;
		double x;
		string tmp;

		DataPoint dp(d);

		while ((tmp=nizer.next())!="")
		{
			if (i%2 == 1)
				x=atof(tmp.c_str());
			else
				dimen = atoi(tmp.c_str());

			if (i%2 == 1)
				dp.addComp(dimen,x);
			i++;
		}

		return d;
	}
}