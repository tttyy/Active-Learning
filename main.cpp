#include "perceptron/ActivePerceptron.h"
#include "margin-based/MarginActiveLearning.h"
#include "helpers.h"
#include "DataPoint.h"
#include "Tokenizer.h"

#include "time.h"
#include "stdio.h"
#include <fstream>
#include <iostream>
#include "string.h"


using namespace std;

#define DIM 10
#define MAXPOINT 1000
#define EPS 0.03
#define DEL 0.03
#define BLOCKSIZE 50

int main(int argc, char** argv)
{
	// 1. Input
	char* train_set = "data/train.txt";
	char* test_set = "data/test.txt";
	string str;
	int train_cnt = 0;

	ifstream input;
	input.open(train_set);
	if(!input)
	{	
		cout << "Unable to open " << train_set << endl;
        exit(1); // terminate with error
	}
	else
	{
		Tokenizer nizer;
		while(input.good())
		{
			input >> str;
			nizer.set(str);
			nizer.setDelimiter(",");
			train[train_cnt][0] = atoi(nizer.next().c_str());
			train[train_cnt][1] = atoi(nizer.next().c_str());
			cnt ++;
		}
	}
	input.close();

	int L=ActivePerceptron::computeL(DIM,DEL,EPS,1);
	int R=ActivePerceptron::computeR(DIM,DEL,EPS,1);
	ActivePerceptron *perca = new ActivePerceptron(DIM, L, R);
	Perceptron *perc = new Perceptron(DIM,L);

	double *x;
	int cor1=0,cor2=0;



	ofstream fs;
	fs.open("out.csv");
	fs << "Num,Acc1,Acc2\n";

	for (int i=0;i<MAXPOINT;i++)
	{
		x = arrayGen(DIM);
		if (perc->read(x,classify(x,DIM))) cor1++;
		if (perca->read(x,classify(x,DIM))) cor2++;
		if ((i+1)%BLOCKSIZE==0)
		{
			fs << i+1<<","<<(double)cor1/BLOCKSIZE<<","<<(double)cor2/BLOCKSIZE<<endl;
			cor1=0;
			cor2=0;
		}
	}

	fs.close();

	return 0;
}