#ifndef FORWARD_H
#define FORWARD_H

#include "MyGraph.h"
#include <map>
#include <utility>

std::map<std::pair<int,int>, int> Forward(const AdjList& adj_graph);

std::map<std::pair<int,int>, int> NodeIteratorN(const EdgeList& edges, const AdjList& adjlist);

std::map<std::pair<int,int>, int> NodeIteratorN4GPU(const EdgeList& edges, int* flatarray, int* indices, int n);

#endif
