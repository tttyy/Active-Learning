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

#define DIM 10
#define MAXPOINT 1000
#define EPS 0.03
#define DEL 0.03
#define BLOCK 10
#define TESTBLOCKSIZE 100

int main(int argc, char** argv)
{
	// 1. Input
	char* train_set = "src/data/train.txt";
	char* test_set = "src/data/test.txt";
	string str;
	int train_cnt = 0;
	vector<DataPoint> *trainVec = new vector<DataPoint>();

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
			int i=0;
			double x[DIM];
			string tmp;
			while ((tmp=nizer.next())!="")
			{
				x[i]=atoi(tmp.c_str());
				i++;
			}
			normalize(x,DIM);
			DataPoint dp(DIM, x, classify(x, DIM));
			trainVec->push_back(dp);
			train_cnt ++;
		}
	}
	input.close();

	int L=ActivePerceptron::computeL(DIM,DEL,EPS,1);
	int R=ActivePerceptron::computeR(DIM,DEL,EPS,1);
	int cnt = 0;
	double *x;
	int cor = 0;

	input.open(test_set);
	if(!input)
	{	
		cout << "Unable to open " << test_set << endl;
        exit(1); // terminate with error
	}
	
	ofstream fs;
	fs.open("out.csv");

	// Perceptron
	Perceptron *perc = new Perceptron(DIM,L);
	fs << "Perceptron\n";
	fs << "m,Acc\n";

	cnt=0;
	for (int i=0;i<L;i++)
	{
		if (cnt >= trainVec->size())
			break;
		DataPoint dp = trainVec->at(cnt++);
		x = dp.x;
		perc->read(x,classify(x,DIM));
		if ((i+1)%(int)(L/BLOCK)==0 || i==L-1)
		{
			cor=0;
			for (int j=0;j<TESTBLOCKSIZE;j++)
			{
				Tokenizer nizer;
				input >> str;
				nizer.set(str);
				nizer.setDelimiter(",");
				int index=0;
				double x[DIM];
				string tmp;
				while ((tmp=nizer.next())!="")
				{
					x[index]=atoi(tmp.c_str());
					index++;
				}
				normalize(x,DIM);
				if (perc->predict(x,classify(x,DIM))) cor++;
			}
			fs << i+1 << "," << (double)cor/TESTBLOCKSIZE << endl;
		}
	}

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
			if (cnt >= trainVec->size())
				break;
			DataPoint dp = trainVec->at(cnt++);
			x = dp.x;
			perca->read(x,classify(x,DIM));
		}while(i+1!=perca->getNumberOfLabel());

		if (cnt >= trainVec->size())
			break;
		if ((i+1)%(int)(L/BLOCK)==0 || i==L-1)
		{
			cor=0;
			for (int j=0;j<TESTBLOCKSIZE;j++)
			{
				Tokenizer nizer;
				input >> str;
				nizer.set(str);
				nizer.setDelimiter(",");
				int index=0;
				double x[DIM];
				string tmp;
				while ((tmp=nizer.next())!="")
				{
					x[index]=atoi(tmp.c_str());
					index++;
				}
				normalize(x,DIM);
				if (perca->predict(x,classify(x,DIM))) cor++;
			}
			fs << i+1 << "," << (double)cor/TESTBLOCKSIZE << endl;
		}
	}

	// Marginal
	/*MarginActiveLearning *margin = new MarginActiveLearning(DIM, 1, EPS, DEL);

	input.seekg(0);
	fs << "\nMarginActiveLearning\n";
	fs << "m,Acc\n";

	while (margin->build_model_separable_iter(*trainVec))
	{
		cor=0;
		for (int j=0;j<TESTBLOCKSIZE;j++)
		{
			Tokenizer nizer;
			input >> str;
			nizer.set(str);
			nizer.setDelimiter(",");
			int index=0;
			double x[DIM];
			string tmp;
			while ((tmp=nizer.next())!="")
			{
				x[index]=atoi(tmp.c_str());
				index++;
			}

			if (margin->classify(DataPoint(DIM, x,classify(x,DIM)))) cor++;
		}
		fs << margin->getNumberOfLabel() << "," << (double)cor/TESTBLOCKSIZE << endl;
	}*/

	fs.close();
	input.close();
	return 0;
}