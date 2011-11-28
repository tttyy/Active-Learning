#include "DataPoint.h"
#include <string>

using namespace std;


double* arrayGen(int d);
int classify(double x[], int d);
void normalize(DataPoint dp);
void normalize(double x[], int d);
DataPoint readData(string, int, bool);