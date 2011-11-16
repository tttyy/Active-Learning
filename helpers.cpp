#include "helpers.h"
#include "stdlib.h"
#include "math.h"

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

void normalize(double *x, int d)
{
	double sum = 0;
	for (int i=0;i<d;i++)
		sum+=x[i]*x[i];
	sum = sqrt(sum);
	for (int i=0;i<d;i++)
		x[i]/=sum;
}