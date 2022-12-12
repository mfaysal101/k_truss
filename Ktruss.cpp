#include <iostream>
#include <string>
#include "include/MyGraph.h"
#include <map>
#include <set>
#include "include/TecIndexSB.h"
#include "include/global.h"
#include <chrono>
#include "include/cudakernel.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include "include/forward.h"
#include <fstream>

using namespace std;


double totalExecutionTime;
double networkReadTime;
double computeTrussTime;
double constructIndexTime;
double timediff;
double onlyKernelTime;

int main(int argc, char *argv []) 
{

	string networkfile;
	string supportfile;
	string indexfile;
	int c;

	if (argc < 5) 
	{
		cout << "Call: ./ktruss c networkfile supportfile indexfile"<< endl;
		exit(-1);
	}
	else
	{
		c = atoi(argv[1]);
		networkfile = argv[2];
		supportfile = argv[3];
		indexfile = argv[4];
	}

	totalExecutionTime = 0.0;
	networkReadTime = 0.0;
	computeTrussTime = 0.0;
	constructIndexTime = 0.0;
	timediff = 0.0;
	onlyKernelTime = 0.0;
	
	// The following commented out code is to show how to transfer data and do computation in GPU
	
	/*
	// Initialize arrays A, B, and C.
    	double A[3], B[3], C[3];

    	// Populate arrays A and B.
    	A[0] = 1; A[1] = 2; A[2] = 9;
    	B[0] = 1; B[1] = 1; B[2] = 1;

    	// Sum array elements across ( C[0] = A[0] + B[0] ) into array C using CUDA.
    	kernel(A, B, C, 3);
	
	// cudaDeviceSynchronize();
    	// Print out result.
    	std::cout << "C = " << C[0] << ", " << C[1] << ", " << C[2] << std::endl;
	*/
	
	auto start = std::chrono::high_resolution_clock::now();
	
	MyGraph mygraph;
	
	TecIndexSB tec;

	EdgeList edgelist = mygraph.ReadEdgeListFromFile(networkfile.c_str());

	///ToDo: I may need to implement functions to remove duplicate edges and self-loop, for now I am assuming there is no such case

	AdjList adjlist = mygraph.EdgeToAdjList(edgelist);
	
	//this will contain all the u of edge e(u, v)
	int* u_list = new int[edgelist.size()];
	//this will contain all the v of edge e(u, v)
	int* v_list = new int[edgelist.size()];
	
	//this will contain all the support value for edge e(u,v)
	int* gpu_spt_list = new int[edgelist.size()];
	
	mygraph.prepareGPUList(edgelist, u_list, v_list, gpu_spt_list);

	auto dup_sup = Forward(adjlist);

	mygraph.PreProcessAdjList(adjlist);
	
	auto ndItN_start = std::chrono::high_resolution_clock::now();

	auto nodeIt_spt = NodeIteratorN(edgelist, adjlist);

	auto ndItN_end = std::chrono::high_resolution_clock::now();

	auto NodeIteratorN_time = std::chrono::duration_cast<std::chrono::nanoseconds>(ndItN_end - ndItN_start).count();

	ofstream out1("out1.txt"), out2("out2.txt"), out3("out3.txt");

	printf("\n\n");

	for(auto it = nodeIt_spt.begin(); it != nodeIt_spt.end(); it++)
	{
		out1 << it->first.first<< "," << it->first.second << "," << it->second << endl;
	}


	pair<int*, int*> flattenlist = mygraph.flattenAdjList(adjlist);

	int* flatarray = flattenlist.first;
	int* indices = flattenlist.second;

	int len = adjlist.size();

	auto ndItN4GPU_start = std::chrono::high_resolution_clock::now();

	auto nodeIt_spt2 = NodeIteratorN4GPU(edgelist, flatarray, indices, len);

	auto ndItN4GPU_end = std::chrono::high_resolution_clock::now();

	auto NodeIteratorN4GPU_time = std::chrono::duration_cast<std::chrono::nanoseconds>(ndItN4GPU_end - ndItN4GPU_start).count();

	for(auto it = nodeIt_spt2.begin(); it != nodeIt_spt2.end(); it++)
	{
		out2 << it->first.first<< "," << it->first.second << "," << it->second << endl;
	}

	int total_len = 0;

	for(int i = 0; i < len; i++)
	{
		int nested_len = adjlist[i].size();

		total_len += nested_len;

		for(int j = 0; j < nested_len; j++)
		{
			//out1 << adjlist[i][j] << endl;
			//printf("%d\n", adjlist[i][j]);
		}
	}

	//cout<<"\n\n\n\n";

	int edgeCount = edgelist.size();

	auto gKernel_start = std::chrono::high_resolution_clock::now();

	int* results = graphKernel(edgelist, len, flatarray, total_len, indices);

	auto gKernel_end = std::chrono::high_resolution_clock::now();

	auto graphKernel_time = std::chrono::duration_cast<std::chrono::nanoseconds>(gKernel_end - gKernel_start).count();

	for(int i = 0; i < edgeCount; i++)
	{
		out3 << edgelist[i].first <<"," << edgelist[i].second <<"," <<results[i] << endl;
	}

	

	auto t3 = std::chrono::high_resolution_clock::now();

	mygraph.readGraphEdgelist(networkfile);
	
	auto t4 = std::chrono::high_resolution_clock::now();
	
	networkReadTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();

	out1.close();
	out2.close();
	out3.close();
	
	if(c == 1)		// do truss decomposition
	{
		map<int, set<Edge>> klistdict;
		map<Edge, int> trussd;
		
		auto t1 = std::chrono::high_resolution_clock::now();
		
		klistdict = mygraph.computeTruss(supportfile, trussd);
		
		auto t2 = std::chrono::high_resolution_clock::now();
		
		computeTrussTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
		
		mygraph.writeSupport(supportfile,trussd);
		
		tec.constructIndex(klistdict, trussd, mygraph);
		
		tec.writeIndex(indexfile);
		
	}
	
	auto end = std::chrono::high_resolution_clock::now();
	
	totalExecutionTime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	
	printf("========totalExecutionTime:%0.9f===========\n", totalExecutionTime*(1e-9));
	printf("========networkReadTime:%0.9f===========\n", networkReadTime*(1e-9));
	printf("========NodeIteratorN_time:%0.9f==========\n", NodeIteratorN_time*(1e-9));
	printf("========NodeIteratorN4GPU_time:%0.9f==========\n", NodeIteratorN4GPU_time*(1e-9));
	printf("========graphKernel_time:%0.9f==========\n", graphKernel_time*(1e-9));
	printf("========onlyKernelTime:%0.9f==========\n", onlyKernelTime*(1e-9));
	printf("========computeTrussTime:%0.9f===========\n", computeTrussTime*(1e-9));
	printf("========constructIndexTime:%0.9f===========\n", constructIndexTime*(1e-9));
	
	return 0;
}
