#include "stdafx.h"
#include <array>
#include <cstdlib>
#include <cmath>
using std::array;

static double sigmoid(double x) {
	return 1/(1+exp(-x));
}
static double dsigmoid(double x) {
	double k=sigmoid(x);
	return k*(1-k);
}

template<typename T>
void reset(T &x) {
	for(T::iterator i=x.begin();i!=x.end();++i)
		*i=0;
}

static int randSign() {
	return rand()>RAND_MAX/2?1:-1;
}

template<int nInput,int nHidden,int nOutput>
class BPNetwork {
	private:
	array<array<double,nHidden>,nInput> weightIH;
	array<array<double,nOutput>,nHidden> weightHO;
	array<double,nHidden> hiddenBias;
	array<double,nOutput> outputBias;
	public:
	typedef array<double,nInput> input;
	typedef array<double,nOutput> output;
	BPNetwork() {
		for(array<double,nHidden> &i:weightIH)
			for(double &j:i) {
				j=randSign()*1.0*rand()/RAND_MAX;
				j/=nInput;
			}
		for(array<double,nOutput> &i:weightHO)
			for(double &j:i) {
				j=randSign()*1.0*rand()/RAND_MAX;
				j/=nHidden;
			}
		for(double &i:hiddenBias) {
			i=randSign()*1.0*rand()/RAND_MAX;
			i/=nHidden;
		}
		for(double &i:outputBias) {
			i=randSign()*1.0*rand()/RAND_MAX;
			i/=nOutput;
		}
	}
	void train(array<double,nInput> x,array<double,nOutput> y,double rate) {
		//hiddenNet
		array<double,nHidden> hiddenNet;
		reset(hiddenNet);
		for(int i=0;i!=nHidden;++i)
			for(int j=0;j!=nInput;++j)
				hiddenNet[i]+=weightIH[j][i]*x[j];
		for(int i=0;i!=nHidden;++i)
			hiddenNet[i]+=hiddenBias[i];
		//hiddenActivation
		auto hiddenActivation=[&hiddenNet](int h){
			return sigmoid(hiddenNet[h]);
		};
		//outputNet
		array<double,nOutput> outputNet;
		reset(outputNet);
		for(int i=0;i!=nOutput;++i)
			for(int j=0;j!=nHidden;++j)
				outputNet[i]+=weightHO[j][i]*hiddenActivation(j);
		for(int i=0;i!=nOutput;++i)
			outputNet[i]+=outputBias[i];
		//outputActivation
		auto outputActivation=[&outputNet](int o){
			return sigmoid(outputNet[o]);
		};
		//weightIH
		array<double,nHidden> errSum;
		reset(errSum);
		for(int h=0;h!=nHidden;++h)
			for(int k=0;k!=nOutput;++k)
				errSum[h]+=(outputActivation(k)-y[k])*dsigmoid(outputNet[k])*weightHO[h][k];
		for(int i=0;i!=nInput;++i)
			for(int h=0;h!=nHidden;++h) {
				weightIH[i][h]-=rate*2*x[i]*dsigmoid(hiddenNet[h])*errSum[h];
			}
		//weightHO
		for(int h=0;h!=nHidden;++h)
			for(int o=0;o!=nOutput;++o) {
				weightHO[h][o]-=rate*2*hiddenActivation(h)*(outputActivation(o)-y[o])*dsigmoid(outputNet[o]);
			}
		//outputBias
		for(int i=0;i!=nOutput;++i)
			outputBias[i]-=rate*2*(outputActivation(i)-y[i])*dsigmoid(outputNet[i]);
		//hiddenBias
		for(int i=0;i!=nHidden;++i)
			hiddenBias[i]-=rate*2*errSum[i]*dsigmoid(hiddenNet[i]);
	}
	array<double,nOutput> f(array<double,nInput> x) {
		array<double,nHidden> hiddenNet;
		reset(hiddenNet);
		for(int i=0;i!=nInput;++i)
			if(x[i]>100 || x[i]<-100) {
				cerr<<"at "<<i<<endl
					<<x[i]<<endl;
				system("pause");
			}
		for(int i=0;i!=nHidden;++i)
			for(int j=0;j!=nInput;++j) {
				hiddenNet[i]+=weightIH[j][i]*x[j];
			}
		//debug(hiddenNet);
		for(int i=0;i!=nHidden;++i)
			hiddenNet[i]+=hiddenBias[i];
		for(double &i:hiddenNet)
			i=sigmoid(i);
		array<double,nHidden> &hiddenActivation=hiddenNet;
		array<double,nOutput> outputNet;
		reset(outputNet);
		for(int i=0;i!=nOutput;++i)
			for(int j=0;j!=nHidden;++j)
				outputNet[i]+=weightHO[j][i]*hiddenActivation[j];
		for(int i=0;i!=nOutput;++i)
			outputNet[i]+=outputBias[i];
		for(double &i:outputNet)
			i=sigmoid(i);
		array<double,nOutput> &outputActivation=outputNet;
		return outputActivation;
	}
	void save(string path) {
		std::fstream file(path.c_str(),std::ios::out);
		for(auto &i:weightIH)
			for(auto &j:i)
				file<<j<<' ';
		for(auto &i:weightHO)
			for(auto &j:i)
				file<<j<<' ';
		for(auto &i:hiddenBias)
			file<<i<<' ';
		for(auto &i:outputBias)
			file<<i<<' ';
	}
	void load(string &path) {
		std::fstream file(path.c_str(),std::ios::in);
		for(auto &i:weightIH)
			for(auto &j:i)
				file>>j;
		for(auto &i:weightHO)
			for(auto &j:i)
				file>>j;
		for(auto &i:hiddenBias)
			file>>i;
		for(auto &i:outputBias)
			file>>i;
	}
};