#include "qbc.h"
using namespace std;

int * newArray(int size)
{
   int * myNewArray=(int *)malloc ( size * sizeof(int) );
   for(int i = 0; i < size; i ++)
   {
        myNewArray[i] = 0;
   }
   return myNewArray;
}

int Qbc::limit(int n)
{
	double res;
	double temp;
	temp = n + 1;
	res = pow(3.1415926, 2) * pow(temp, 2);
	res = log(res / (3 * this->del)) / this->eps;
	return res;
}

void Qbc::setValue(double err, double prob)
{
	this->eps = err;
	this->del = prob;
	this->v_s = -1;
	this->v_e = -1;
	this->num = 0;
}

int Qbc::label(int x, int y)
{
	double res, d;
	d = this->target * 3.141592653 / 180.000000;
	res = x * cos(d) + y * sin(d);
	if(res > 0)
		return 1;
	else if(res < 0)
		return -1;
	else
		return 0;
}

int Qbc::judge(double d, int x, int y)
{
	int res = 0;
	double w_x, w_y, de, r;
	de = d * 3.1415926 / 180.000000;
	w_x = cos(de);
	w_y = sin(de);
	r = x * 1.000000 * w_x + y * 1.000000 * w_y;
	if(r > 0)
		return 1;
	else if(r < 0)
		return -1;
}

void Qbc::updateVP(double dg)
{
	double temp, st, en, news, newe;
	temp = dg + 90.000000;
	st = temp;
	en = temp + 180.000000;
	if(st >= 360)
		st = st - 360.000000;
	if(en >= 360)
		en = en - 360.000000;
	if(this->v_s == -1)
	{
		this->v_s = st;
		this->v_e = en;
	}
	else
	{
		if(abs(this->v_s - st) > 180)
			news = max(this->v_s, st);
		else
			news = min(this->v_s, st);
		if(abs(this->v_e - en) > 180)
			newe = min(this->v_e, en);
		else
			newe = max(this->v_e, en);
		this->v_s = news;
		this->v_e = newe;
	}

}

void Qbc::readTrain(int length, char* input_file)
{
	this->train = (int **) malloc(length * sizeof (int *));
	int cnt = 0;
	string str;
	ifstream input;

	this->len = length;
	for(int i = 0; i < length ; i ++)
	{
		this->train[i] = newArray(2);
	}
	input.open(input_file);
	if(!input)
	{	
		cout << "Unable to open " << input_file << endl;
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
			train[cnt][0] = atoi(nizer.next().c_str());
			train[cnt][1] = atoi(nizer.next().c_str());
			cnt ++;
		}
	}
	input.close();
}


double Qbc::gibbs()
{
	double degree, res;
	double range = 0.00;
	int ranges = 0;
	if(this->v_s != -1 && this->v_e != -1)
	{
		ranges ++;
		if(this->v_e < this->v_s)
		{
			range += this->v_s - this->v_e;
		}
		else
		{
			range += 360 - abs(this->v_e - this->v_s);
		}
	}
	else
	{
		range = 360.000000;
	}

	degree = rand()%10001;
	degree = degree * 1.000000 * range / 10000.00000000;

	// map the degree to real degrees
	if(ranges == 0)
	{
		res = degree;
	}
	else if(ranges == 1)
	{
		if(this->v_e < this->v_s)
			degree += this->v_e;
		else
		{
			if(degree > this->v_s)
				degree += (this->v_e - this->v_s);
		}
		res = degree;
	}
	else
	{
		cout << "Error, wrong number for ranges" << endl;
		exit(0);
	}
	return res;
}

double Qbc::ptd(double x, double y)
{
	double degree;
	degree = atan2(y,x) * 180.0000000 / 3.1415926;
	if(degree < 0)
		degree = 360.000000 - degree * (-1.000000);
	return degree;
}

void Qbc::setTarget(double t)
{
	this->target = t;
}

double Qbc::output()
{
	double res;
	if(this->v_e < this->v_s)
		res = (this->v_s + this->v_e) / 2.000000;
	else
	{
		res = (this->v_s + this->v_e) / 2.000000 + 180.000000;
		if(res > 360.000000)
			res = res - 360.000000;
	}
	return res;
}

void Qbc::start()
{
	int t, n, x, y, s, t_n, r1, r2, flag, lcnt;
	double degree1, degree2, dx, dy, dg;

	srand((unsigned)time(0));
	lcnt = n = t = 0;
	t_n = this->limit(n);

	while(t <= t_n)
	{
		// Sample
		s = rand()%this->len;
		x = train[s][0];
		y = train[s][1];
		//Gibbs
		degree1 = this->gibbs();
		degree2 = this->gibbs();
		while(degree1 == degree2)
		{
			degree2 = this->gibbs();
		}
		r1 = this->judge(degree1,x,y);
		r2 = this->judge(degree2,x,y);
		if(r1 != r2)
		{
			//update version space
			flag = this->label(x, y);
			if(flag < 0)
			{
				x = (-1) * x;
				y = (-1) * y;
			}
			if(flag != 0)
			{
				// convert position to degree
				dx = x * 1.000000;
				dy = y * 1.000000;
				dg = this->ptd(dx, dy);
				this->updateVP(dg);
			}
			n ++;
			t_n = this->limit(n);
			t = 0;
			// cout << this->output() << endl;
		}
		else
		{
			//reject this sample
			t ++;
		}
		lcnt ++;
		// cout << lcnt << " " << t << " " << t_n << endl;
	}
	cout << "Learn over. Example number " << n << endl;
	cout << "Final degree is " << this->output() << endl;
	this->num = n;
}

void Qbc::start2()
{
	int t, n, x, y, s, t_n, r1, r2, flag, lcnt, e;
	double degree1, degree2, dx, dy, dg;

	
	lcnt = n = t = e = 0;
	t_n = this->limit(n);

	while(n < 10)
	{
		// Sample
		s = rand()%this->len;
		x = train[s][0];
		y = train[s][1];
		flag = this->label(x, y);
		if(flag < 0)
		{
			x = (-1) * x;
			y = (-1) * y;
		}
		if(flag != 0)
		{
			// convert position to degree
			dx = x * 1.000000;
			dy = y * 1.000000;
			dg = this->ptd(dx, dy);
			this->updateVP(dg);
		}
		n ++;
		//e = this->error();
	}
	//cout << "Learn over. Example number " << n << endl;
	//cout << "Final degree is " << this->output() << endl;
	this->num = n;
}


double Qbc::error()
{
	double err;
	err = abs(this->target - this->output());
	if(err > 180)
		err = 360.000000 - err;
	err = err / 180.000000;
	cout << err << endl;
	return err;
}

int Qbc::expnum()
{
	return this->num;
}
