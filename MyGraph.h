#ifndef MYGRAPH_H
#define MYGRAPH_H

#include "Edge.h"
#include <map>
#include <vector>
#include <set>



class MyGraph
{
	public:
	
	long long numEdges;
	long long numVertices;
	
	std::map<int, std::map<int, Edge>> graph;
	
	void readGraphEdgelist(std::string filename);
	
	int processEdge(int src, int dst);
	
	std::map<int, std::set<Edge>> computeTruss(std::map<Edge, int>& trussd);
	
	int computeSupport(std::map<Edge, int>& support);
	
	void writeSupport(std::string& filename, std::map<Edge, int>& support);
	
	void bucketSortedEdgelist(int kmax, std::map<Edge, int>& sp, std::vector<Edge>& sorted_elbys, std::map<int, int>& svp, std::map<Edge, int>& sorted_ep);
	
	void reorderEL(std::vector<Edge>& sorted_elbys, std::map<Edge, int>& sorted_ep, std::map<Edge, int>& supd, std::map<int, int>& svp, Edge e1);
	
	Edge removeEdge(int u, int v);
	
	Edge getEdge(int u, int v);
	
};


#endif