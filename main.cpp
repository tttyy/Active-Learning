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
#include <vector>


using namespace std;

#define PLAIN_DATA 0

#define MAXPOINT 1000
#define EPS 0.03
#define DEL 0.03
#define BLOCK 10
#define TESTBLOCKSIZE 100

int main(int argc, char** argv)
{
	// 1. Input
#if PLAIN_DATA
	char* train_set = "src/data/two-normal.txt";
	char* test_set = "src/data/two-normal-test.txt";
	int DIM = 5;
#else
	char* train_set = "src/data/20news/comp-rec/train-freq.txt";
	char* test_set = "src/data/20news/comp-rec/test-freq.txt";
	int DIM = 61188;
#endif
	string str;
	int train_cnt = 0;
	vector<DataPoint> trainVec;

	ifstream input;
	input.open(train_set);
	if(!input)
	{	
		cout << "Unable to open " << train_set << endl;
        exit(1); // terminate with error
	}
	else
	{
		while(input >> str)
		{
			trainVec.push_back(readData(str,DIM,!PLAIN_DATA).clone());
			train_cnt ++;
		}
	}
	input.close();

	int L=ActivePerceptron::computeL(DIM,DEL,EPS,1);
	int R=ActivePerceptron::computeR(DIM,DEL,EPS,1);
	int cnt = 0;
	double *x;
	int cor = 0;
	
	ofstream fs;
	fs.open("out.csv");

	cout << "Input Done" << endl;

	// Perceptron
	Perceptron *perc = new Perceptron(DIM,L);
	fs << "Perceptron\n";
	fs << "m,Acc\n";

	cnt=0;
	for (int i=0;i<L;i++)
	{
		if (cnt >= trainVec.size())
			break;
		DataPoint dp = trainVec[cnt++];
		perc->read(dp);
		if ((i+1)%(int)(L/BLOCK)==0 || i==L-1)
		{
			cor=0;		
			int j=0;
			input.open(test_set);
			if(!input)
			{	
				cout << "Unable to open " << test_set << endl;
				exit(1); // terminate with error
			}
			while (input >> str)
			{
				DataPoint dpTest = readData(str,DIM,!PLAIN_DATA);
				if (perc->predict(dpTest)) cor++;
				j++;
			}
			input.close();
			fs << i+1 << "," << (double)cor/j << endl;
		}
	}
	cout << "Perceptron Done!" << endl;

	//ActivePerceptron
	ActivePerceptron *perca = new ActivePerceptron(DIM, L, R);
	input.seekg(0);
	fs << "\nActivePerceptron\n";
	fs << "m,Acc\n";

	cnt=0;
	for (int i=0;i<L;i++)
	{
		do
		{
			if (cnt >= trainVec.size())
				break;
			DataPoint dp = trainVec[cnt++];
			perca->read(dp);
		}while(i+1!=perca->getNumberOfLabel());

		if (cnt >= trainVec.size())
			break;
		if ((i+1)%(int)(L/BLOCK)==0 || i==L-1)
		{
			cor=0;
			input.open(test_set);
			if(!input)
			{	
				cout << "Unable to open " << test_set << endl;
				exit(1); // terminate with error
			}
			int j=0;
			while (input >> str)
			{
				DataPoint dpTest = readData(str,DIM,!PLAIN_DATA);
				if (perca->predict(dpTest)) cor++;
				j++;
			}
			input.close();
			fs << i+1 << "," << (double)cor/j << endl;
		}
	}
	cout << "Active Perceptron Done!" << endl;

	// Marginal
	/*
	MarginActiveLearning *margin = new MarginActiveLearning(DIM, 1, EPS, DEL);

	input.seekg(0);
	fs << "\nMarginActiveLearning\n";
	fs << "m,Acc\n";

	while (margin->build_model_separable_iter(trainVec))
	{
		cor=0;
		input.open(test_set);
		if(!input)
		{	
			cout << "Unable to open " << test_set << endl;
			exit(1); // terminate with error
		}
		int j=0;
		while (input >> str)
		{
			DataPoint dpTest = readData(str,DIM,!PLAIN_DATA);
			if (margin->classify(dpTest)==dpTest.label) cor++;
			j++;
		}
		input.close();
		fs << margin->getNumberOfLabel() << "," << (double)cor/j << endl;
	}

	cout << "Margin Done!" << endl;
	*/

	fs.close();
	return 0;
}