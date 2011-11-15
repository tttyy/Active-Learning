#include "qbc.h"
#include "Tokenizer.h"

using namespace std;



void updateVP(double v_s, double v_e, double dg)
{
	double temp, st, en, news, newe;
	temp = dg + 90.000000;
	st = temp;
	en = temp + 180.000000;
	if(st >= 360)
		st = st - 360.000000;
	if(en >= 360)
		en = en - 360.000000;
	if(abs(v_s - st) > 180)
		news = max(v_s, st);
	else
		news = min(v_s, st);
	if(abs(v_e - en) > 180)
		newe = min(v_e, en);
	else
		newe = max(v_e, en);
	cout << news << " " << newe << endl;
}

int main()
{
	int train_length = 10000;
	double eps = 0.03;
	double delta = 0.97;
	double error = 0;
	double number = 0;
	double temp;
	double e[100];
	char* train_set = "train.txt";
	char* test_set = "test.txt";

	srand((unsigned)time(0));
	
	/*
	Qbc query;
	query.readTrain(train_length,train_set);
	cout << "Finish reading!" << endl;
	for(int i = 1; i <= 100; i ++)
	{
		query.setValue(eps, delta);
		query.setTarget(i * 2.00);
		query.start();
		error += query.error();
		number += query.expnum();
		if(query.error() > 0.1)
		{
			cout << query.output() << endl;
		}
	}
	error = error / 100.00;
	number = number / 100.00;
	cout << error << endl;
	cout << number << endl;
	*/

	
	Qbc query;
	query.readTrain(train_length,train_set);
	cout << "Finish reading!" << endl;
	for(int i = 1; i <= 100; i ++)
	{
		query.setValue(eps, delta);
		query.setTarget(i * 2.00);
		query.start2();
		e[i - 1] = query.error();
		error += query.error();
		number += query.expnum();
		if(e[i - 1] > 0.5)
		{
			cout << i * 2.0 << " " << e[i - 1] << endl;
		}
	}
	cout << endl;
	error = error / 100.00;
	number = number / 100.00;
	cout << error << endl;
	cout << number << endl;


	
	
	// Set seed for rand()
	/*
	srand((unsigned)time(0));
	int test, d1, d2, d3, d4;
	double degree;
	d1 = d2 = d3 = d4 = 0;
	for(int i = 0; i < 4000; i ++)
	{
		test = rand()%10001;
		degree = test * 360.00000000 / 10000.00000000;
		//cout << degree << " " << cos(degree) << " " << sin(degree) << endl;
		if(degree <= 90)
			d1 ++;
		else if(degree <= 180)
			d2 ++;
		else if(degree <= 270)
			d3 ++;
		else if(degree <= 360)
			d4 ++;
	}
	cout << d1 << endl;
	cout << d2 << endl;
	cout << d3 << endl;
	cout << d4 << endl;

	updateVP(0.0, 180.0, 225.0);
	updateVP(315.0, 190.0, 270.0);
	updateVP(70.0, 300.0, 135.0);

	double temp = 0.0;
	double degree;

	for(int i = 0; i < 24; i ++)
	{
		degree = temp * 3.1415926 / 180.000000;
		cout << temp << " " << cos(degree) << " " << sin(degree) << endl;
		temp += 15.0;
	}

	*/
	

	/*

	cout << query.limit(0) << endl;
	cout << query.limit(5) << endl;
	cout << query.limit(95) << endl;
	*/

	/*
	input_file.open(train_set);
	if(!input_file)
	{	
		cout << "Unable to open " << train_set << endl;
        exit(1); // terminate with error
	}
	else
	{
		Tokenizer nizer;
		while(input_file.good())
		{
			input_file >> str;
			nizer.set(str);
			nizer.setDelimiter(",");
			train[cnt][0] = atoi(nizer.next().c_str());
			train[cnt][1] = atoi(nizer.next().c_str());
			cnt ++;
		}
	}
	input_file.close();

	//Start the loop
	t_n = 24;
	n = t = 0;
	while(t < t_n)
	{
		s = sample(train_length);
		x = train[s][0];
		y = train[s][1];
		t ++;
	}

	// Test
	cout << pow(3.14, 2) << endl;
	*/

	// Just to see some output
	char click[16];
	gets(click);
	
}
