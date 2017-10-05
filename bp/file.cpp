#include "stdafx.h"

vector<array<double,inputSize> > read(std::string path) {
	vector<array<double,inputSize> > anss;
	
	std::fstream file(path.c_str(),std::ios::in);
	while(!file.eof()) {
		std::string validation="";
		file>>validation;
		if(validation=="") {
			break;
		}
		array<double,inputSize> ans;
		struct k{
			long down,up;
		};
		array<k,43> key;
		for(int i=0;i!=43;++i) {
			file>>key.at(i).down>>key.at(i).up;
		}
		for(int i=0;i!=press;++i) {
			ans[i]=key.at(i).up-key.at(i).down;
		}
		for(int i=0;i!=interval;++i) {
			ans[press+i]=key.at(i+1).down-key.at(i).up;
		}
		file>>ans.back();
		//prehandle
		for(int i=0;i!=press+interval;++i)
			ans[i]/=200;
		ans.back()/=10;
		anss.push_back(ans);
	}
	return anss;
}