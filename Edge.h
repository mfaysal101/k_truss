#ifndef EDGE_H
#define EDGE_H

class Edge
{
	public:
	
	int s;
	int t;
	

	Edge()
	{
		
	}
	
	Edge(int source, int target)
	{
		s = source;
		t = target;
	}
	
	// "<" operator overloading required by c++ map for custom object type
	bool operator<(const Edge &ob) const 
	{
        return s < ob.s || (s == ob.s && t < ob.t);
	}

};

#endif