#ifndef qbc_h
#define qbc_h

#include "../Tokenizer.h"
#include <cstdio>
#include <string>
#include <time.h> 
#include <cmath>
#include <fstream>
#include <iostream>


class Qbc
{
private:
	double eps;
	double del;
	double v_s;
	double v_e;
	double target;
	int len;
	int num;
	int** train;

public:
	void readTrain(int length, char* input_file);
	void setValue(double err, double prob);
	void setTarget(double t);
	void start();
	void start2();
	double error();
	void updateVP(double dg);
	double gibbs();
	double output();
	int expnum();
	int limit(int n);
	int judge(double d, int x, int y);
	int label(int x, int y);
	double ptd(double x, double y);
	// bool judge();
	
};


#endif