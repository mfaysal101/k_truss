#ifndef MYGRAPH_H
#define MYGRAPH_H

#include "Edge.h"
#include <map>
#include <vector>
#include <set>
#include <string>
#include <utility>


class MyGraph
{
	public:
	
	long long numEdges;
	long long numVertices;

	size_t totalEdges;
	size_t totalVertices;
	
	MyGraph()
	{
		numVertices = 0;
		numEdges = 0;
	}

	std::map<int, std::map<int, Edge>> graph;
	std::vector<int> vertexIds;
	
	void readGraphEdgelist(std::string filename);
	
	int processEdge(int src, int dst);
	
	std::map<int, std::set<Edge>> computeTruss(std::string pathtec, std::map<Edge, int>& trussd);
	
	int computeSupport(std::map<Edge, int>& support);
	
	void writeSupport(std::string& filename, std::map<Edge, int>& support);
	
	void bucketSortedEdgelist(int kmax, std::map<Edge, int>& sp, std::vector<Edge>& sorted_elbys, std::map<int, int>& svp, std::map<Edge, int>& sorted_ep);
	
	void reorderEL(std::vector<Edge>& sorted_elbys, std::map<Edge, int>& sorted_ep, std::map<Edge, int>& supd, std::map<int, int>& svp, Edge e1);
	
	Edge removeEdge(int u, int v);
	
	Edge getEdge(int u, int v);

	EdgeList ReadEdgeListFromFile(const char* filename);
	
	size_t getNumVertices(const EdgeList& edges);

	AdjList EdgeToAdjList(const EdgeList& edges);

	void PreProcessAdjList(AdjList& adjlist);
	
	void prepareGPUList(const EdgeList& edgelist, int* u_list, int* v_list, int* gpu_spt_list);

	std::pair<int*, int*> flattenAdjList(AdjList& adjlist);

};


#endif