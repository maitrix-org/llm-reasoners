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

#ifndef __SASACTION
#define __SASACTION

#include "ToFunction.h"
#include <vector>
#include <map>
#include <iostream>
using std::vector;
using std::map;
using std::ostream;

using VAL::pddl_type;
using VAL::pred_symbol;
using VAL::proposition;

namespace SAS {

class ValueRep {
private:
	const pddl_type * ptp;
	int segment;
	ValueElement * vel;
public:
	ValueRep(const pddl_type * pt,int s,ValueElement * v) : 
		ptp(pt), segment(s), vel(v) {};
	
	int getSegment() const {return segment;};
	const ValueElement * getValue() const {return vel;};
	bool matches(ValueRep * vrep,FastEnvironment * fs);
	ValueRep(ValueRep * vr,FastEnvironment * fenv);
	const string typeName() const {return ptp->getName();};
	const pddl_type * getType() const {return ptp;};
	void write(ostream & o) const 
	{
		o << ptp->getName() << "_" << segment;
	};
};

inline ostream & operator << (ostream & o,const ValueRep & v)
{
	v.write(o);
	return o;
};

class SegmentRep {
private:
	vector<ValueRep *> valuereps;
	vector<int> levelcounts;

public:
	SegmentRep(ValueRep * vr) : valuereps(1,vr), levelcounts(1,0)
	{};
	bool add(ValueRep * vr,FastEnvironment * fe)
	{
// Check whether this is a new ValueRep or not (with this binding)
		for(vector<ValueRep*>::iterator i = valuereps.begin();i != valuereps.end();++i)
		{
			if((*i)->matches(vr,fe)) 
			{
//				cout << "nothing\n";
				return false;
			};
		};
		valuereps.push_back(new ValueRep(vr,fe));
//		cout << "Added: " << *(valuereps[valuereps.size()-1]->getValue()) << "\n";
		return true;
	};
	bool growOneLevel(const pddl_type * pt,const TIMobjectSymbol * tob,
							FunctionStructure * fs);
	void cap() {levelcounts.push_back(valuereps.size());};
};

class RangeRep {
private:
	const pddl_type * group;
	const TIMobjectSymbol * object;
	vector<SegmentRep *> segreps;
public:
	RangeRep(const pddl_type * pt,const TIMobjectSymbol * tob,
				vector<ValueElement *> & vels) :
		group(pt), object(tob)
	{
		for(size_t i = 0;i < vels.size();++i)
		{
			segreps.push_back(new SegmentRep(new ValueRep(pt,i,vels[i])));
		};
	};
	bool add(ValueRep * vr,FastEnvironment * f)
	{
//		cout << "For " << group->getName() << " " << vr->getSegment() << " " <<
//				object->getName() << ": ";
		return segreps[vr->getSegment()]->add(vr,f);
	};
	bool growOneLevel(FunctionStructure * fs)
	{
		bool activated = false;
		for(size_t i = 0;i < segreps.size();++i)
		{
			activated |= segreps[i]->growOneLevel(group,object,fs);
		};
		return activated;
	};
	void cap()
	{
		for(vector<SegmentRep *>::iterator i = segreps.begin();i != segreps.end();++i)
		{
			(*i)->cap();
		};
	};
};

typedef map<const var_symbol *,vector<ValueRep *> > VMap;
typedef map<const pddl_type *,vector<vector<pair<const var_symbol *,SASActionTemplate *> > > >
					PreMap;

class SASActionTemplate {
private:
	static PreMap preMap;
	static map<const pred_symbol *,vector<SASActionTemplate *> > otherprecs;
	
	operator_ * op;
	VMap preconditions;
	VMap postconditions;
	vector<proposition *> statics;
	vector<proposition *> otherpres;
	vector<proposition *> otherposts;
public:
	SASActionTemplate(operator_ * o,const VMap & pres,
						const VMap & posts,
						const vector<proposition *> & sts,
						const vector<proposition *> & opres,
						const vector<proposition *> & oposts) :
		op(o), preconditions(pres), postconditions(posts), statics(sts), 
		otherpres(opres), otherposts(oposts) 
	{
		for(VMap::const_iterator i = pres.begin();i != pres.end();++i)
		{
			for(vector<ValueRep*>::const_iterator j = i->second.begin();
					j != i->second.end();++j)
			{
				if((int) preMap[i->first->type].size() <= (*j)->getSegment())
				{
					preMap[i->first->type].resize((*j)->getSegment()+1);
				};
				preMap[i->first->type][(*j)->getSegment()].
							push_back(make_pair(i->first,this));
			};
		};
		for(vector<proposition *>::const_iterator i = opres.begin();i != opres.end();++i)
		{
			otherprecs[(*i)->head].push_back(this);
		};
	};
	typedef VMap::const_iterator iterator;
	iterator precondsBegin() const {return preconditions.begin();};
	iterator precondsEnd() const {return preconditions.end();};
	iterator postcondsBegin() const {return postconditions.begin();};
	iterator postcondsEnd() const {return postconditions.end();};
	
	void write(ostream & o) const
	{
		o << "(:action " << op->name->getName() << "\n  :parameters (";
		for(VAL::var_symbol_list::const_iterator ps = op->parameters->begin();
									ps != op->parameters->end();++ps)
		{
			o << "?" << (*ps)->getName() << " - " << (*ps)->type->getName() << " ";
		};
		o << ")\n  :precondition (and\n";
		for(VMap::const_iterator i = preconditions.begin();i != preconditions.end();++i)
		{
			for(vector<ValueRep *>::const_iterator j = i->second.begin();j != i->second.end();++j)
			{
				o << "      ((" << (*j)->typeName() << "_" << (*j)->getSegment()
					<< " ?" << i->first->getName() << ") ";
				
				(*j)->getValue()->showValue(o);
				o << ")\n";
			};
		};
		for(vector<proposition*>::const_iterator s = statics.begin();s != statics.end();++s)
		{
			cout << "     (" << (*s)->head->getName();
			for(VAL::parameter_symbol_list::const_iterator pm = (*s)->args->begin();
					pm !=(*s)->args->end();++pm)
			{
				cout << " ?" << (*pm)->getName();
			};
			cout << ")\n";
		};
		for(vector<proposition*>::const_iterator s = otherpres.begin();s != otherpres.end();++s)
		{
			cout << "    (" << (*s)->head->getName();
			for(VAL::parameter_symbol_list::const_iterator pm = (*s)->args->begin();
					pm !=(*s)->args->end();++pm)
			{
				cout << " ?" << (*pm)->getName();
			};
			cout << ")\n";
		};
		o << ")\n  :effect (and\n";
		for(VMap::const_iterator i = postconditions.begin();i != postconditions.end();++i)
		{
			for(vector<ValueRep *>::const_iterator j = i->second.begin();j != i->second.end();++j)
			{
				o << "      ((" << (*j)->typeName() << "_" << (*j)->getSegment()
					<< " ?" << i->first->getName() << ") ";
				
				(*j)->getValue()->showValue(o);
				o << ")\n";
			};
		};
		for(vector<proposition*>::const_iterator s = otherposts.begin();s != otherposts.end();++s)
		{
			cout << "    (" << (*s)->head->getName();
			for(VAL::parameter_symbol_list::const_iterator pm = (*s)->args->begin();
					pm !=(*s)->args->end();++pm)
			{
				cout << " ?" << (*pm)->getName();
			};
			cout << ")\n";
		};
		o << "))\n";
	};
	int preCount() const 
	{
		int c = otherpres.size();
		for(VMap::const_iterator i = preconditions.begin();i != preconditions.end();++i)
		{
			c += i->second.size();
		};
		return c;
	};
	static vector<pair<const var_symbol *,SASActionTemplate*> > & findOps(const pddl_type * pt,int i)
	{
		return preMap[pt][i];
	};
	const operator_ * getOp() const {return op;};
	bool checkPre(FunctionStructure * fs,FastEnvironment * fenv,const var_symbol * v,ValueRep * vrep);
	void enact(FastEnvironment * fenv,Reachables & reachables,
						vector<proposition *> & others);
};

inline ostream & operator<<(ostream & o,const SASActionTemplate & at)
{
	at.write(o);
	return o;
};

/*
class SASAction {
private:
	vector<
*/

};

#endif
