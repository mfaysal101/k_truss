#ifndef TECINDEXSB_H
#define TECINDEXSB_H

#include <map>
#include <set>
#include "SGN.h"
#include <queue>
#include "MyGraph.h"

class TecIndexSB
{
	public:
	
	std::map<int,std::set<int>> vtoSGN ;/*dictionary for original graph vertices to summary graph nodes. key is original graph vertex , value is the super node set which include this vertex*/
	
	std::map<int, SGN> idSGN;// dictionary for super nodes;  key is id and value is super node objet
	
	std::map<int,std::set<int>> SG;//index summary graph; key is the vertex, value is the edge set of  key vertex
	
	void constructIndex(std::map<int, std::set<Edge>> klistdict, std::map<Edge, int> trussd, MyGraph mygraph);
	
	void addComVertex(int x, int tns, std::map<int, std::set<int>>& vtoSGN);
	
	void addEdgetoTrussCom(Edge e, int tns,std::map<Edge, std::map<int, int>>& edgeigd);
	
	void processTriangleEdge(Edge e1, int t1, std::set<Edge>& proes, std::queue<Edge>& Q, SGN& Vk, std::map<Edge, std::map<int, int>>& edgeigd, std::map<Edge, bool>& activeEdges);
	
	void addEdge4EdgeSpec(Edge e1, SGN Vk, std::map<Edge, std::map<int, int>>& edgeigd);
	
	void writeIndex(std::string filename);
	
	void readIndex(MyGraph mygraph, std::string filename);
	
	std::vector<std::vector<Edge>> findkCommunityForQuery(int query, int k);
};

#endif
