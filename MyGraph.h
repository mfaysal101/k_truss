#ifndef MYGRAPH_H
#define MYGRAPH_H

#include "Edge.h"
#include <map>

using namespace std;


class MyGraph
{
	public:
	
	long long numEdges;
	long long numVertices;
	
	map<int, map<int, Edge>> graph;
	
	void readGraphEdgelist(string filename);
	
	void computeTruss(map<Edge, int>& trussd);
	
	int computeSupport(map<Edge, int>& support);
	
	void writeSupport(string& filename, map<Edge, int>& support);
};


#endif