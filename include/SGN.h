#ifndef SGN_H
#define SGN_H

#include "Edge.h"
#include <vector>

class SGN
{
	public:
	
	int truss;
	int idd;
	std::vector<Edge> edgelist;
	
	SGN()
	{
		
	}
	
	SGN(int truss, int tnid)
	{
		this->truss = truss;
		this->idd = tnid;
	}
	
	void addEdge(Edge e)
	{
		edgelist.push_back(e);
	}
	
};

#endif