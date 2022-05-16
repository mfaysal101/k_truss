#include "../include/forward.h"
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace std;


namespace 
{
	template <class InputIterator1, class InputIterator2>
	int IntersectionSize(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2, int n) 
	{
  		int result = 0;
  		while (first1 != last1 && first2 != last2) 
		{
    			if (*first1 >= n || *first2 >= n)
			{
      				break;
			}
   			if (*first1 < *first2)
			{
      				++first1;
			}
    			else if (*first1 > *first2)
			{
      				++first2;
			}
    			else 
			{
      				++result;
      				++first1;
      				++first2;
    			}
  		}
  		return result;
	}

	int IntersectionSizeGPU(int first1, int last1, int first2, int last2, int n, int* flatarray)
	{
		int result = 0;
		
		while(first1 != last1 && first2 != last2)
		{
			if(flatarray[first1] >= n || flatarray[first2] >= n)
			{
				break;
			}
			if(flatarray[first1] < flatarray[first2])
			{
				++first1;
			}
			else if(flatarray[first1] > flatarray[first2])
			{
				++first2;
			}
			else
			{
				++result;
				++first1;
				++first2;
			}	
		}
		return result;
	}
}


map<pair<int,int>, int> Forward(const AdjList& adj_graph)
{
	const int n = adj_graph.size();

	vector<vector<int>> A(n);

	vector<pair<int, int>> deg(n);

	map<pair<int,int>, int>dup_sup;

	for(int i = 0; i < n; i++)
	{
		deg[i] = make_pair(adj_graph[i].size(), i);
	}

	sort(deg.begin(), deg.end(), greater<pair<int, int>>());
	
	vector<int> new_pos(n);

	for(int i = 0; i < n; i++)
	{
		new_pos[deg[i].second] = i;
	}

	for(int i = 0; i < n; i++)
	{
		const int u = deg[i].second;
		
		for(const int v : adj_graph[u])
		{
			if(new_pos[v] <= i)
			{
				continue;
			}
			dup_sup[{u,v}] += IntersectionSize(A[u].begin(), A[u].end(), A[v].begin(), A[v].end(), n);

			A[v].push_back(i);
		}
	}
	
	return dup_sup;	
}



map<pair<int,int>, int> NodeIteratorN(const EdgeList& edges, const AdjList& adjlist)
{
	map<pair<int,int>, int>nodeIt_spt;

	const int n = adjlist.size();

	for(auto edge: edges)
	{
		int u = edge.first;
		int v = edge.second;
		
		nodeIt_spt[edge] = IntersectionSize(adjlist[u].begin(), adjlist[u].end(), adjlist[v].begin(), adjlist[v].end(), n);
	}
	
	return nodeIt_spt;
}


map<pair<int,int>, int> NodeIteratorN4GPU(const EdgeList& edges, int* flatarray, int* indices, int n)
{
	map<pair<int,int>, int>nodeIt_spt;

	for(auto edge: edges)
	{
		int u = edge.first;
		int v = edge.second;
		
		int firstInd1 = indices[u];
		int lastInd1 = indices[u+1];
		int firstInd2 = indices[v];
		int lastInd2 = indices[v+1];

		nodeIt_spt[edge] = IntersectionSizeGPU(firstInd1, lastInd1, firstInd2, lastInd2, n, flatarray);
	}

	return nodeIt_spt;
}

