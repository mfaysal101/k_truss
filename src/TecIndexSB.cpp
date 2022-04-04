#include <set>
#include <map>
#include "../include/Edge.h"
#include <queue>
#include <fstream>
#include <sstream>
#include <string>
#include "../include/TecIndexSB.h"
#include <iostream>
#include <string.h>
#include <chrono>
#include "../include/global.h"

using namespace std;


void TecIndexSB::constructIndex(std::map<int, std::set<Edge>> klistdict, std::map<Edge, int> trussd, MyGraph mg)
{
	auto tm1 = std::chrono::high_resolution_clock::now();
	
	map<Edge, map<int, int>> edgeigd;
	
	int t1, t2, tnid = 0; //tree node id
	
	if(klistdict.find(2) != klistdict.end())
	{
		klistdict.erase(2);
	}
	
	Edge e1, e2;
	
	int x, y;
	
	for(std::map<int, std::set<Edge>>::iterator it = klistdict.begin(); it != klistdict.end(); it++)
	{
		set<Edge> kedgelist = it->second;
		map<Edge, bool>activeEdges;
		
		//copying element from set to map for ease in deletion later in the code
		for(set<Edge>::iterator edgeit = kedgelist.begin(); edgeit != kedgelist.end(); edgeit++)
		{
			activeEdges.insert({*edgeit, true});
		}
		
		for(set<Edge>::iterator edgeit = kedgelist.begin(); edgeit != kedgelist.end(); edgeit++)
		{

			set<Edge> proes;
			queue<Edge> Q;

			if (activeEdges[*edgeit])
			{
				auto time1 = std::chrono::high_resolution_clock::now();

				Q.push(*edgeit);
				proes.insert(*edgeit);
				activeEdges[*edgeit] = false;
				
				SGN Vk(it->first, tnid);
				idSGN.insert(make_pair(tnid, Vk));

				set<int> nl;

				SG.insert(make_pair(tnid, nl));
				
				auto time2 = std::chrono::high_resolution_clock::now();
				timediff += std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1).count();

				while (!Q.empty())
				{
					Edge uv = Q.front();
					x = uv.s;
					y = uv.t;

					if (mg.graph[x].size() > mg.graph[y].size())
					{
						y = uv.s;
						x = uv.t;
					}

					idSGN[tnid].addEdge(uv);
					Q.pop();

					addComVertex(x, tnid, vtoSGN);
					addComVertex(y, tnid, vtoSGN);
					addEdgetoTrussCom(uv, tnid, edgeigd);
					mg.removeEdge(x, y);

					for (auto ite = mg.graph[x].begin(); ite != mg.graph[x].end(); ite++)
					{
						int ne = ite->first;
						if (mg.graph[y].find(ne) != mg.graph[y].end())
						{
							e1 = mg.getEdge(x, ne);
							t1 = trussd[e1];
							e2 = mg.getEdge(y, ne);
							t2 = trussd[e2];
							processTriangleEdge(e1, t1, proes, Q, idSGN[tnid], edgeigd, activeEdges);
							processTriangleEdge(e2, t2, proes, Q, idSGN[tnid], edgeigd, activeEdges);	
						}
					}
				}

				tnid++;
			}
		}
	}
	
	printf("Index time:%0.9f\n", timediff*(1e-9));
	auto tm2 = std::chrono::high_resolution_clock::now();

	constructIndexTime += std::chrono::duration_cast<std::chrono::nanoseconds>(tm2 - tm1).count();
	
}

void TecIndexSB::addComVertex(int x, int tns, map<int, set<int>>& vtoSGN)
{
	if(vtoSGN.find(x) != vtoSGN.end())
	{
		vtoSGN[x].insert(tns);
	}
	else
	{
		set<int> cl;
		cl.insert(tns);
		vtoSGN.insert(make_pair(x, cl));
	}
}

void TecIndexSB::addEdgetoTrussCom(Edge e, int tns, map<Edge, map<int, int>>& edgeigd)
{
	if(edgeigd.find(e) != edgeigd.end())
	{
		map<int, int> temp = edgeigd[e];
		
		for(auto it = temp.begin(); it!= temp.end(); it++)
		{
			if(SG[tns].find(it->first)== SG[tns].end())
			{
				SG[tns].insert(it->first);
				SG[it->first].insert(tns);
			}
		}
		
		edgeigd.erase(e);
	}
}


void TecIndexSB::processTriangleEdge(Edge e1, int t1, set<Edge>& proes, queue<Edge>& Q, SGN& Vk, map<Edge, map<int, int>>& edgeigd, map<Edge, bool>& activeEdges)
{
	if(proes.find(e1) == proes.end())
	{
		if(t1 == Vk.truss)
		{
			activeEdges[e1] = false;
			Q.push(e1);
		}
		else
		{
			addEdge4EdgeSpec(e1, Vk, edgeigd);
		}
		proes.insert(e1);
	}
}


void TecIndexSB::addEdge4EdgeSpec(Edge e1, SGN Vk, map<Edge, map<int, int>>& edgeigd)
{
	if(edgeigd.find(e1) == edgeigd.end())
	{
		map<int,int> nl;
		
		nl.insert(make_pair(Vk.idd, Vk.truss));
		
		edgeigd.insert(make_pair(e1, nl));
	}
	else
	{
		if(edgeigd[e1].find(Vk.idd) == edgeigd[e1].end())
		{
			edgeigd[e1].insert(make_pair(Vk.idd, Vk.truss));
		}
	}
}


void TecIndexSB::writeIndex(string filename)
{
	string supernodefile = filename+"_spnode.txt";
	
	string original2Indexfile = filename+"_original2Index.txt";
	
	string summaryIndexfile = filename+"_summaryIndexGraph.txt";
	
	std::ofstream writer(supernodefile);
	
	if(writer)
	{
		for(auto it = idSGN.begin(); it != idSGN.end(); it++)
		{
			SGN sg = it->second;
			writer<<"id,"<<it->first<<",truss,"<<sg.truss<<endl;
			
			for(size_t i = 0; i < sg.edgelist.size(); i++)
			{
				writer<<sg.edgelist[i].s<<","<<sg.edgelist[i].t<<endl;
			}
		}
		writer.close();
	}
	else
	{
		cout<<"Could not write supernode"<<endl;
	}
	
	writer.open(original2Indexfile);
	if(writer)
	{
		writer<<"original_node_id index_graph_node_id"<<endl;
		
		for(auto it = vtoSGN.begin(); it != vtoSGN.end(); it++)
		{
			for(auto it2 = it->second.begin(); it2 != it->second.end(); it2++)
			{
				writer<<it->first<<" "<<*it2<<endl;
			}
		}
		writer.close();
	}
	else
	{
		cout<<"Could not write original2Indexfile"<<endl;
	}
	
	writer.open(summaryIndexfile);
	
	if(writer)
	{
		for(auto it = SG.begin(); it != SG.end(); it++)
		{
			for(auto it2 = it->second.begin(); it2 != it->second.end(); it2++)
			{
				writer<<it->first<<","<<*it2<<endl;
			}
		}
		
		writer.close();
	}
	else
	{
		cout<<"Could not write summaryIndexfile"<<endl;
	}
}

void TecIndexSB::readIndex(MyGraph mygraph, string filename)
{
	string supernodefile = filename+"_spnode.txt";
	
	string original2Indexfile = filename+"_original2Index.txt";
	
	string summaryIndexfile = filename+"_summaryIndexGraph.txt";
	
	FILE* fp = fopen(supernodefile.c_str(), "r");
	
	if(fp == NULL)
	{
		cout<<"Failed to open supernode files"<<endl; 
		exit(EXIT_FAILURE);
	}
	
	char* line = NULL;
	size_t len = 0;
	char* token;
	const char* is_id = "id";
	int id, truss;
	
	while((getline(&line, &len, fp)!= -1))
	{
		token = strtok(line," ,\n\t");
		
		if(strcmp(token, is_id) == 0)
		{
			token = strtok(NULL," ,\n\t");
			id = atoi(token);
			token = strtok(NULL," ,\n\t");
			token = strtok(NULL," ,\n\t");
			truss = atoi(token);
			
			SGN sg(truss, id);
			idSGN.insert(make_pair(id,sg));
			set<int> nl;
			SG.insert(make_pair(id, nl));
		}
		else
		{
			int u =	atoi(token);
			token = strtok(NULL," ,\n\t");
			int v = atoi(token);
			idSGN[id].addEdge(mygraph.getEdge(u, v));
		}			
	}

	for(auto it = idSGN.begin(); it != idSGN.end(); it++)
	{
		SGN sg = it->second;
		vector<Edge> edges = sg.edgelist;
		
		for(unsigned int i = 0; i < edges.size(); i++)
		{
			Edge e = edges[i];
			
			if(vtoSGN.find(e.s) == vtoSGN.end())
			{
				set<int> tempset;
				vtoSGN.insert(make_pair(e.s, tempset));
			}
			vtoSGN[e.s].insert(it->first);
			
			if(vtoSGN.find(e.t) == vtoSGN.end())
			{
				set<int> tempset;
				vtoSGN.insert(make_pair(e.t, tempset));
			}
			
			vtoSGN[e.t].insert(it->first);
		}
		
	}
	
	fp = fopen(summaryIndexfile.c_str(), "r");
	
	if(fp == NULL)
	{
		cout<<"Failed to open summaryIndexfile"<<endl; 
		exit(EXIT_FAILURE);
	}
	
	while((getline(&line, &len, fp)!= -1))
	{
		token = strtok(line," ,\n\t");
		int u = atoi(token);
		token = strtok(NULL," ,\n\t");
		int v = atoi(token);
		
		SG[u].insert(v);
		SG[v].insert(u);
	}
	
	fclose(fp);
	if(line)
	{
		free(line);
	}
}

vector<vector<Edge>> TecIndexSB::findkCommunityForQuery(int query, int k)
{
	set<int> qIn = vtoSGN[query];
	vector<vector<Edge>> cl;
	set<int>ignidl;
	queue<int>ignidq;
	vector<Edge> community;
	
	for(auto it = qIn.begin(); it != qIn.end(); it++)
	{
		int qid = *it;
		
		if(idSGN[qid].truss >= k && (ignidl.find(qid) == ignidl.end()))
		{
			ignidl.insert(qid);
			ignidq.push(qid);
			community.insert(community.end(), idSGN[qid].edgelist.begin(), idSGN[qid].edgelist.end());
			
			while(!ignidq.empty())
			{
				int ig = ignidq.front();
				for(int nid: SG[ig])
				{
					if(idSGN[nid].truss >= k && (ignidl.find(nid) == ignidl.end()))
					{
						ignidq.push(nid);
						ignidl.insert(nid);
						community.insert(community.end(), idSGN[nid].edgelist.begin(), idSGN[nid].edgelist.end());
					}
				}
			}
			
			cl.push_back(community);
			printf("Number of edges in this community:%lu\n",community.size());
		}
	}
	
	return cl;
}



