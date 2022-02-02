#include<iostream>
#include<string>
#include "MyGraph.h"
#include<map>
#include<set>
#include "TecIndexSB.h"
#include "global.h"
#include <chrono>

using namespace std;


double totalExecutionTime;
double networkReadTime;
double computeTrussTime;
double constructIndexTime;


int main(int argc, char *argv[]) 
{

	string networkfile;
	string supportfile;
	string indexfile;
	int c;

	if (argc < 5) 
	{
		cout << "Call: ./ktruss c networkfile supportfile indexfile"<< endl;
		exit(-1);
	}
	else
	{
		c = atoi(argv[1]);
		networkfile = argv[2];
		supportfile = argv[3];
		indexfile = argv[4];
	}

	totalExecutionTime = 0.0;
	networkReadTime = 0.0;
	computeTrussTime = 0.0;
	constructIndexTime = 0.0;
	
	auto start = std::chrono::high_resolution_clock::now();
	
	MyGraph mygraph;
	
	TecIndexSB tec;
	
	auto t3 = std::chrono::high_resolution_clock::now();
	
	mygraph.readGraphEdgelist(networkfile);
	
	auto t4 = std::chrono::high_resolution_clock::now();
	
	networkReadTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();
	
	if(c == 1)		// do truss decomposition
	{
		map<int, set<Edge>> klistdict;
		map<Edge, int> trussd;
		
		auto t1 = std::chrono::high_resolution_clock::now();
		
		klistdict = mygraph.computeTruss(supportfile, trussd);
		
		auto t2 = std::chrono::high_resolution_clock::now();
		
		computeTrussTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
		
		mygraph.writeSupport(supportfile,trussd);
		
		tec.constructIndex(klistdict, trussd, mygraph);
		
		tec.writeIndex(indexfile);
		
	}
	
	auto end = std::chrono::high_resolution_clock::now();
	
	totalExecutionTime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	
	printf("========totalExecutionTime:%0.9f===========\n", totalExecutionTime*(1e-9));
	printf("========networkReadTime:%0.9f===========\n", networkReadTime*(1e-9));
	printf("========computeTrussTime:%0.9f===========\n", computeTrussTime*(1e-9));
	printf("========constructIndexTime:%0.9f===========\n", constructIndexTime*(1e-9));
	
	return 0;
}
