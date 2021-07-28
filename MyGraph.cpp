#include "MyGraph.h"
#include "Edge.h"
#include <stdio.h>
#include <cstring>
#include <iostream>

using namespace std;

typedef map<int, Edge> Edgemap;
typedef map<int, map<int, Edge>> graphmap;

void MyGraph::readGraphEdgelist(string filename)
{
	char FILENAME[filename.length() + 1];
	strcpy(FILENAME, filename.c_str());

	cout << "Reading network " << filename << " file\n" << flush;
	
	FILE* fp = fopen(FILENAME, "r");

	if (fp == NULL) {
		printf("Could not open file\n");
		exit(EXIT_FAILURE);
	}
	printf("File open successful\n");
	
	char* line = NULL;
	size_t len = 0;
	char* token;
	
	numEdges = 0;

	while((getline(&line, &len, fp) != -1)) 
	{
		int src = atoi(strtok(line, " \t"));
		
		int dst = atoi(strtok(NULL, " \t"));
		
		if(src == dst)
		{
			continue;
		}
		
		Edge edge(src, dst);
		
		numEdges++;
		
		graphmap::iterator src_it = graph.find(src);
		
		if(src_it != graph.end())
		{
			if(src_it->second.count(dst) == 0)  // this checking is done to ensure no repeated edge is inserted
			{
				src_it->second.insert(make_pair(dst, edge));
			}
		}
		else
		{
			Edgemap tempmap;
			tempmap.insert(make_pair(dst, edge));
			graph.insert(make_pair(src, tempmap));
		}
		
		graphmap::iterator dst_it = graph.find(dst);
		
		if(dst_it != graph.end())
		{
			if(dst_it->second.count(src) == 0)  // this checking is done to ensure no repeated edge is inserted
			{
				dst_it->second.insert(make_pair(src, edge));
			}
		}
		else
		{
			Edgemap tempmap;
			tempmap.insert(make_pair(src, edge));
			graph.insert(make_pair(dst, tempmap));
		}
		
	}
	
	numVertices = graph.size();
	
	printf("Graph successfully read with %lld vertices and %lld edges\n", numVertices, numEdges);
}

void MyGraph::computeTruss(map<Edge, int>& trussd)
{
	map<Edge, int> sp;
	
	int kmax = computeSupport(sp);
	
	std::printf("maximum support found:%d\n", kmax);
}


int MyGraph::computeSupport(map<Edge, int>& support)
{
	int s = 0;
	int smax = 0;
	
	map<Edge, int> temp;
	
	for(graphmap::iterator it1 = graph.begin(); it1 != graph.end(); it1++)
	{
		int u = it1->first;
		
		Edgemap& u_edgelist = it1->second;		//getting the list of edges that vertex u is connected to
		
		for(Edgemap::iterator it2 = u_edgelist.begin(); it2 != u_edgelist.end(); it2++)
		{
			int v = it2->first;
			Edge edge = it2->second;
		
			if(support.find(edge) == support.end())	// the edge is not already in the support list
			{
				s = 0;
				
				for(Edgemap::iterator it3 = u_edgelist.begin(); it3 != u_edgelist.end(); it3++)
				{
					int w = it3->first;
					
					if(v != w)
					{
						const Edgemap& v_edgelist = graph[v]; //TODO: This line is more appropriate after line 110 and before 113
						if(v_edgelist.count(w))
						{
							s++;
						}
					}
				}
				
				if(s > smax)
				{
					smax = s;
				}
				support.insert(make_pair(edge, s));
			}
			//*/
		}	
	}
	
	return smax;
}