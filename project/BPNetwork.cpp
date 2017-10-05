// BPNetwork.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include "ClassBPNetwork.cpp"
#include "file.h"

int l(int x) {
	return x/10==0?1:1+l(x/10);
}

int _tmain(int argc,_TCHAR* argv[]) {
	BPNetwork<inputSize,500,1> network;
	char cmd='n';
	cerr<<"Load weights?(y/n)";
	std::cin>>cmd;
	cmd=tolower(cmd);
	if(cmd=='y') {
		cerr<<"Path:";
		string path;
		std::cin>>path;
		network.load(path);
	}
	else {
		sampleGroup positive=read("positive_sample.txt");
		sampleGroup negative=read("negative_sample.txt");
		cerr<<"Positive size:"<<positive.size()<<endl
			<<"Negative size:"<<negative.size()<<endl;
		int r;
		double rate;
		cerr<<"Training round?(>100 Recommended)";
		std::cin>>r;
		cerr<<"Learning rate?(10/tr<x<0.1 Recommended)";
		std::cin>>rate;
		//cerr<<"Training round  ";
		string progress="======================================";
		cerr<<progress;
		for(int t=1;t<=r;++t) {
			for(sample &s:positive) {
				array<double,1> presult={1};
				network.train(s,presult,rate);
			}
			for(sample &s:negative) {
				array<double,1> nresult={0};
				network.train(s,nresult,rate);
			}
			/*for(int i=0;i!=l(t-1);++i)
				cerr<<'\b';
			cerr<<t;*/
			if(int(progress.length()*(1.0*t/r))!=int(progress.length()*((1.0*t-1)/r))) {
				progress[int(progress.length()*((1.0*t-1)/r))]='+';
				for(int i=0;i!=progress.length();++i)
					cerr<<'\b';
				cerr<<progress;
			}

		}
		cerr<<endl;
	}
	sampleGroup validation=read("H:\\ud\\validation_sample.txt");
	for(int i=0;i!=validation.size();++i){
		cerr<<"Sample "<<i<<":";cout<<network.f(validation[i])[0]<<endl;
	}
	if(cmd=='n') {
		cerr<<"Save weights?(y/n)";
		std::cin>>cmd;
		if(tolower(cmd)=='y') {
			cerr<<"Path:";
			string path;
			std::cin>>path;
			network.save(path);
		}
	}

	system("pause");
	return 0;
}