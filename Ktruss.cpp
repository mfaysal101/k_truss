#include<iostream>
#include<string>
#include "MyGraph.h"
#include<map>
#include<set>
#include "TecIndexSB.h"

using namespace std;


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
	return 0;
}
