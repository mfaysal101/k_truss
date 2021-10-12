#include<iostream>
#include<string>
#include "MyGraph.h"
#include<map>

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
	
	mygraph.readGraphEdgelist(networkfile);
	
	if(c == 1)		// do truss decomposition
	{
		std::map<Edge, int> support;
		mygraph.computeSupport(support);
		mygraph.writeSupport(supportfile,support);
	}
	return 0;
}
