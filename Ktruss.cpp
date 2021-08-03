#include<iostream>
#include<string>
#include "MyGraph.h"
#include<map>

using namespace std;


int main(int argc, char *argv[]) 
{

	string networkfile;
	string indexfile;
	int c;

	if (argc < 4) 
	{
		cout << "Call: ./ktruss c networkfile indexfile"<< endl;
		exit(-1);
	}
	else
	{
		c = atoi(argv[1]);
		networkfile = argv[2];
		indexfile = argv[3];
	}

	MyGraph mygraph;
	
	mygraph.readGraphEdgelist(networkfile);
	
	if(c == 1)		// do truss decomposition
	{
		std::map<Edge, int> support;
		mygraph.computeSupport(support);
		mygraph.writeSupport(indexfile,support);
	}
	return 0;
}
