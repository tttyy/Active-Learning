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

#define PLAIN_DATA 2

#define PERCEPTRON_OPEN
#define ACTIVE_PERCEPTRON_OPEN
#define MARGIN_OPEN

#define MAXPOINT 1000
#define EPS 0.02
#define DEL 0.03
#define BLOCK 10
#define TESTBLOCKSIZE 100

int main(int argc, char** argv)
{
	// 1. Input
#if PLAIN_DATA == 0
	char* train_set = "src/data/train.txt";
	char* test_set = "src/data/test.txt";
	int DIM = 10;
#else 
#if PLAIN_DATA == 1
	char* train_set = "src/data/2norm/2norm.train";
	char* test_set = "src/data/2norm/2norm.test";
	int DIM = 10;
#else
	char* train_set = "src/data/20news/comp-rec/train-tfidf.txt";
	char* test_set = "src/data/20news/comp-rec/test-tfidf.txt";
	int DIM = 61188;
#endif
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
			DataPoint dp = readData(str,DIM,PLAIN_DATA==2);
			normalize(dp);
			trainVec.push_back(dp.clone());
			train_cnt ++;
		}
	}
	input.close();

	int L=ActivePerceptron::computeL(DIM,DEL,EPS,1);
	if (L>train_cnt) L=train_cnt;
	int R=ActivePerceptron::computeR(DIM,DEL,EPS,1);
	int cnt = 0;
	double *x;
	int cor = 0;
	
	ofstream fs;
	fs.open("out.csv");

	cout << "Input Done" << endl;
	
#ifdef PERCEPTRON_OPEN
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
				DataPoint dpTest = readData(str,DIM,PLAIN_DATA==2);
				if (perc->predict(dpTest)) cor++;
				j++;
			}
			input.close();
			fs << i+1 << "," << (double)cor/j << endl;
			cout << i+1 << "," << (double)cor/j << endl;
		}
	}
	cout << "Perceptron Done!" << endl;
#endif

#ifdef ACTIVE_PERCEPTRON_OPEN
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
				DataPoint dpTest = readData(str,DIM,PLAIN_DATA==2);
				if (perca->predict(dpTest)) cor++;
				j++;
			}
			input.close();
			fs << i+1 << "," << (double)cor/j << endl;
			cout << i+1 << "," << (double)cor/j << endl;
		}
	}
	cout << "Active Perceptron Done!" << endl;
#endif

#ifdef MARGIN_OPEN
	// Marginal
	double C = 1;
	if (PLAIN_DATA == 1)
		C = 0.0003;
	if (PLAIN_DATA == 2)
		C = 0.0000006;
	MarginActiveLearning *margin = new MarginActiveLearning(DIM, C, EPS, DEL);

	input.seekg(0);
	fs << "\nMarginActiveLearning\n";
	fs << "m,Acc\n";
#if PLAIN_DATA == 0
	margin->set_niter_for_separable();
	while (margin->build_model_separable_iter(trainVec))
#else
	margin->set_niter_for_unseparable(0.25);
	while (margin->build_model_unseparable_iter(trainVec,0,0.25))
#endif
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
			DataPoint dpTest = readData(str,DIM,PLAIN_DATA==2);
			if (margin->classify(dpTest)==dpTest.label) cor++;
			j++;
		}
		input.close();
		fs << margin->getNumberOfLabel() << "," << (double)cor/j << endl;
		cout << margin->getNumberOfLabel() << "," << (double)cor/j << endl;
	}

	cout << "Margin Done!" << endl;
#endif

	fs.close();
	return 0;
}