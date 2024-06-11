/************************************************************************
 * Copyright 2008, Strathclyde Planning Group,
 * Department of Computer and Information Sciences,
 * University of Strathclyde, Glasgow, UK
 * http://planning.cis.strath.ac.uk/
 *
 * Maria Fox, Richard Howey and Derek Long - VAL
 * Stephen Cresswell - PDDL Parser
 *
 * This file is part of VAL, the PDDL validator.
 *
 * VAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * VAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with VAL.  If not, see <http://www.gnu.org/licenses/>.
 *
 ************************************************************************/

#ifndef __PARTITIONS
#define __PARTITIONS

/* This code is designed to support a simple problem in managing partitions.
   The idea is that we have a collection of objects that are going to start 
   off as individuals and, over a series of operations, we are going to merge
   them into sets of increasing size. The sets will together form a partition 
   of the union of all the objects at all times. At the end we want each object
   to be mapped to its partition so that we can associate each object with 
   a unique structure that is identical for all the objects in the same partition.
*/
#include <list>
#include <map>

template<class _Key, class _PData, class _PDataCombine>
class Partitioner {
private:
	typedef std::pair<_PData,list< _Key > > _pdata;
	typedef std::map<_Key,_pdata> _pmap;
	
	_pmap partitiondata;
	_PDataCombine combine;
	int partitions;

	struct partitionStruct {
		_Key key;
		partitionStruct * next;

		partitionStruct(_Key k) : key(k), next(0) {};
	};

	typedef map<_Key,partitionStruct *> PElink;
	PElink pelements;

	partitionStruct * trace(partitionStruct * p) const
	{
		while(p->next) p = p->next;
		return p;
	};
	
public:
	Partitioner(_PDataCombine c) : combine(c), partitions(0)
	{};

	void add(_Key k,_PData p)
	{
		if(pelements.find(k) != pelements.end()) return;
		list<_Key> sk;
		sk.push_front(k);
		partitiondata.insert(make_pair(k,make_pair(p,sk)));
		pelements[k] = new partitionStruct(k);
		partitions++;
	};

	bool contains(_Key k) const
	{
		return pelements.find(k) != pelements.end();
	};

	void setData(_Key k,_PData p)
	{
		if(pelements.find(k) == pelements.end()) 
		{
			add(k,p);
		}
		else
		{
			partitiondata[trace(pelements[k])->key].first = p;
		};
	};


 	void connect(_Key k1,_Key k2)
	{
		if(pelements.find(k1) == pelements.end() ||
			pelements.find(k2) == pelements.end()) return;
			
		partitionStruct * e1 = trace(pelements[k1]);
		partitionStruct * e2 = trace(pelements[k2]);
	
		if(e1==e2) return;
		partitiondata[e1->key].second.merge(partitiondata[e2->key].second);
		partitiondata[e1->key].first =
			combine(partitiondata[e1->key].first,partitiondata[e2->key].first);
		partitiondata.erase(e2->key);
		e2->next = e1;
		partitions--;
	};		

	_PData getData(_Key k)
	{
		return partitiondata.find(trace(pelements.find(k)->second)->key)->second.first;
	};

	int count() const
	{
		return partitions;
	};

	_Key partition(_Key k) const
	{
		if(pelements.find(k) == pelements.end()) return k;

		return trace(pelements[k])->key;
	};

	typedef typename _pmap::const_iterator PSI;
	PSI begin() {return partitiondata.begin();};
	PSI end() {return partitiondata.end();};
	typedef const pair<_Key, _pdata > DataSource;
	
	static _PData grabData(DataSource & p)
	{
		return p.second.first;
	};
};


#endif
