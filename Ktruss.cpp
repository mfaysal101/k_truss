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
	constructIndexTime = 0.0;
	
	auto start = std::chrono::high_resolution_clock::now();
	
	MyGraph mygraph;
	
	TecIndexSB tec;
	
	mygraph.readGraphEdgelist(networkfile);
	
	if(c == 1)		// do truss decomposition
	{
		map<int, set<Edge>> klistdict;
		map<Edge, int> trussd;
		
		klistdict = mygraph.computeTruss(supportfile, trussd);
		
		mygraph.writeSupport(supportfile,trussd);
		
		tec.constructIndex(klistdict, trussd, mygraph);
		
		tec.writeIndex(indexfile);
		
	}
	
	auto end = std::chrono::high_resolution_clock::now();
	
	totalExecutionTime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	
	printf("========totalExecutionTime:%0.9f===========\n", totalExecutionTime*(1e-9));
	printf("========constructIndexTime:%0.9f===========\n", constructIndexTime*(1e-9));
	
	return 0;
}
