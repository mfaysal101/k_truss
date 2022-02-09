#include "MyGraph.h"
#include "Edge.h"
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>


using namespace std;

typedef map<int, Edge> Edgemap;
typedef map<int, map<int, Edge>> graphmap;


void MyGraph::readGraphEdgelist(string filename)
{
	char FILENAME[filename.length() + 1];
	strcpy(FILENAME, filename.c_str());

	cout << "Reading network " << filename << " file\n" << flush;
	
	FILE* fp = fopen(FILENAME, "r");

	if (fp == NULL) 
	{
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
		int src = atoi(strtok(line, " ,\t"));
		
		int dst = atoi(strtok(NULL, " ,\t"));
		
		numEdges += processEdge(src, dst);
	}
	
	numVertices = graph.size();
	
	printf("Number of vertices:%lld\n",numVertices);
	printf("Number of edges:%lld\n", numEdges);
}

int MyGraph::processEdge(int src, int dst)
{
	if(src == dst)
	{
		return 0;
	}
	
	Edge edge(src, dst);
	
	graphmap::iterator src_it = graph.find(src);
	
	if(src_it == graph.end())
	{
		Edgemap tempmap;
		tempmap.insert(make_pair(dst, edge));
		graph.insert(make_pair(src, tempmap));
	}
	else
	{	
		if(src_it->second.count(dst) > 0)  // this checking is done to ensure no repeated edge is inserted
		{
			return 0;
		}
		src_it->second.insert(make_pair(dst, edge));
	}
	
	graphmap::iterator dst_it = graph.find(dst);
	
	if(dst_it == graph.end())
	{
		Edgemap tempmap;
		tempmap.insert(make_pair(src, edge));
		graph.insert(make_pair(dst, tempmap));
	}
	else
	{
		if(dst_it->second.count(src) > 0)  // this checking is done to ensure no repeated edge is inserted
		{
			return 0;
		}
		dst_it->second.insert(make_pair(src, edge));
	}
	
	return 1;
}

map<int, set<Edge>> MyGraph::computeTruss(string pathtec, map<Edge, int>& trussd)
{
	map<int, set<Edge>> klistdict;
	
	map<Edge, int> sp;

	int kmax = computeSupport(sp);
	
	string supportname = "support.txt";
 
	writeSupport(supportname,sp);

	std::printf("maximum support found:%d\n", kmax);
	
	int k = 2;
	
	vector<Edge> sorted_elbys(sp.size());
	
	map<Edge, int> sorted_ep;
	
	map<int, int> svp;
	
	bucketSortedEdgelist(kmax, sp, sorted_elbys, svp, sorted_ep);

	set<Edge> kedgelist;
	klistdict.insert(make_pair(k, kedgelist));

	for(size_t i = 0; i < sorted_elbys.size(); i++)
	{
		auto e = sorted_elbys[i];
		int val = sp[e];
		if(val > (k - 2))
		{
			set<Edge> kedgelist;
			klistdict.insert(make_pair(k, kedgelist));
			k = val + 2; 
		}
		
		int src, dst;
		
		if(graph[e.s].size() < graph[e.t].size())
		{
			src = e.s;
			dst = e.t;
		}
		else
		{
			src = e.t;
			dst = e.s;
		}
		
		map<int, Edge> nls = graph[src];

		for(Edgemap::iterator it = nls.begin(); it != nls.end(); it++)
		{
			int v = it->first;
			if(graph[v].count(dst))
			{
				Edge e1 = graph[v].at(src);
				Edge e2 = graph[v].at(dst);
				if(!(trussd.find(e1) != trussd.end() || trussd.find(e2) != trussd.end()))
				{
					if(sp[e1] > (k - 2))
					{
						reorderEL(sorted_elbys, sorted_ep, sp, svp, e1);	
					}
					
					if(sp[e2] > (k - 2))
					{
						reorderEL(sorted_elbys, sorted_ep, sp, svp, e2);
					}
				}
			}
		}
		klistdict[k].insert(e);
		trussd.insert(make_pair(e, k));
	}

	return klistdict;
}

void MyGraph::reorderEL(std::vector<Edge>& sorted_elbys, std::map<Edge, int>& sorted_ep, std::map<Edge, int>& supd, std::map<int, int>& svp, Edge e1)
{
	int val = supd[e1];
	int pos1 = sorted_ep[e1];
	int cp = svp[val];
	
	if(cp != pos1)
	{
		Edge tmp2 = sorted_elbys[cp];
		sorted_ep[e1] = cp;
		sorted_ep[tmp2] = pos1;
		sorted_elbys[pos1] = tmp2; //it could be a source of potential array index out of bound error
		svp[val] = cp + 1;
		sorted_elbys[cp] = e1;
	}
	else
	{
		if(sorted_elbys.size() > (cp + 1) && supd[sorted_elbys[cp+1]] == val)
		{
			svp[val] = cp+1;
		}
		else
		{
			svp[val] = -1;
		}
	}
	
	if(svp.find(val-1) == svp.end() || svp[val-1] == -1)
	{
		svp[val - 1] = cp;
	}
	supd[e1] = val - 1;
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


void MyGraph::writeSupport(string& filename, map<Edge, int>& support)
{
	cout<<"support file name:"<<filename<<"\n";
	
	FILE* fp = fopen(filename.c_str(), "w");
	
	if(fp != nullptr)
	{
		for(auto it = support.begin(); it != support.end(); it++)
		{
			Edge edge = it->first;
			fprintf(fp, "%d,%d,%d\n", edge.s, edge.t, it->second);
			//writer<<edge.s<<","<<edge.t<<","<<it->second<<endl;
		}
		fclose(fp);
	}
	else
	{
		cout<<"Could not write support"<<endl;
	}
}

// I commented out the following write support function because the c++ output stream seems to be taking greater time than even Java's BufferedWriter.
// My initial assumption was the map iteration being responsible, but after experimentation, it does not appear to be the case
/*
void MyGraph::writeSupport(string& filename, map<Edge, int>& support)
{
	cout<<"support file name:"<<filename<<"\n";
	
	std::ofstream writer(filename);
	
	if(writer)
	{
		for(auto it = support.begin(); it != support.end(); it++)
		{
			Edge edge = it->first;
			writer<<edge.s<<","<<edge.t<<","<<it->second<<endl;
		}
		writer.close();
	}
	else
	{
		cout<<"Could not write support"<<endl;
	}
}
*/

void MyGraph::bucketSortedEdgelist(int kmax, map<Edge, int>& sp, vector<Edge>& sorted_elbys, map<int, int>& svp, map<Edge, int>& sorted_ep)
{
	vector<int> bucket((kmax + 1), 0);
	
	//#pragma omp parallel for
	for(map<Edge, int>::iterator it = sp.begin(); it != sp.end(); it++)
	{
		bucket[it->second]++;
	}

	int temp;
	
	int p = 0;
	
	for(int i = 0; i < kmax + 1; i++)
	{
		temp = bucket[i];
		bucket[i] = p;
		p = p + temp;
	}
		
	for(map<Edge, int>::iterator it = sp.begin(); it != sp.end(); it++)
	{
		sorted_elbys[bucket[it->second]] = it->first;
		sorted_ep.insert(make_pair(it->first, bucket[it->second]));
		if(svp.find(it->second) == svp.end())
		{
			svp.insert(make_pair(it->second, bucket[it->second]));
		}
		bucket[it->second] = bucket[it->second] + 1;
	}
}

Edge MyGraph::removeEdge(int u, int v)
{
	Edge re;
	
	if(graph[u].find(v) != graph[u].end())
	{
		auto it = graph[u].find(v);
		re = it->second;
		graph[u].erase(v);
		graph[v].erase(u);
	}	
	return re;
}

Edge MyGraph::getEdge(int u, int v)
{
	return graph[u].find(v)->second;
}
