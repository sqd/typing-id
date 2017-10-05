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

template<int nInput,int nHidden,int nOutput>
class BPNetwork {
	private:
	array<array<double,nHidden>,nInput> weightIH;
	array<array<double,nOutput>,nHidden> weightHO;
	public:
	BPNetwork() {
		for(array<double,nHidden> &i:weightIH)
			for(double &j:i)
				j=1.0*rand()/RAND_MAX;
		for(array<double,nOutput> &i:weightHO)
			for(double &j:i)
				j=1.0*rand()/RAND_MAX;
	}
	void debug() {
		for(array<double,nHidden> &i:weightIH)
			for(double &j:i)
				cerr<<j<<" ";
		cerr<<endl;
		for(array<double,nOutput> &i:weightHO)
			for(double &j:i)
				cerr<<j<<" ";
	}
	void train(array<double,nInput> x,array<double,nOutput> y,double rate) {
		//hiddenNet
		array<double,nHidden> hiddenNet;
		reset(hiddenNet);
		for(int i=0;i!=nHidden;++i)
			for(int j=0;j!=nInput;++j)
				hiddenNet[i]+=weightIH[j][i]*x[j];
		auto hiddenActivation=[&hiddenNet](int h){
			return sigmoid(hiddenNet[h]);
		};
		//outputNet
		array<double,nOutput> outputNet;
		reset(outputNet);
		for(int i=0;i!=nOutput;++i)
			for(int j=0;j!=nHidden;++j)
				outputNet[i]+=weightHO[j][i]*hiddenActivation(j);
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

	}
	array<double,nOutput> f(array<double,nInput> x) {
		array<double,nHidden> hiddenNet;
		reset(hiddenNet);
		for(int i=0;i!=nHidden;++i)
			for(int j=0;j!=nInput;++j)
				hiddenNet[i]+=weightIH[j][i]*x[j];
		for(double &i:hiddenNet)
			i=sigmoid(i);
		array<double,nHidden> &hiddenActivation=hiddenNet;
		array<double,nOutput> outputNet;
		reset(outputNet);
		for(int i=0;i!=nOutput;++i)
			for(int j=0;j!=nHidden;++j)
				outputNet[i]+=weightHO[j][i]*hiddenActivation[j];
		for(double &i:outputNet)
			i=sigmoid(i);
		array<double,nOutput> &outputActivation=outputNet;
		return outputActivation;
	}
};