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

#ifndef __TOFUNCTION
#define __TOFUNCTION

#include <iostream>
#include "ptree.h"
#include "TIM.h"
#include <iterator>
#include <set>
#include <map>
#include <vector>
using std::vector;
using std::set;
using std::iterator;

using namespace TIM;
using VAL::pddl_type;
using VAL::FastEnvironment;
using VAL::operator_;
using VAL::var_symbol;

namespace Inst {
class instantiatedOp;
};

using Inst::instantiatedOp;

namespace SAS {

extern bool use_sasoutput;

class ValHolder {
protected:
	PropertySpace * pspace;
	const PropertyState * pstate;
	const pddl_type * forThis;
	Property * prop;
public:
	ValHolder(const pddl_type * pt,const PropertyState * ps,PropertySpace * prpsp,Property * p) : 
		pspace(prpsp), pstate(ps), forThis(pt), prop(p)
	{};
	virtual ~ValHolder() {};
	virtual void write(ostream & o) const = 0;
	const PropertyState * getState() const {return pstate;};
	Property * getProp() const {return prop;};
	const pddl_type * getType() const {return forThis;};
	PropertySpace * getSpace() const {return pspace;};
	virtual bool operator==(const ValHolder & v) const
	{
		return pstate == v.pstate;
	};
	virtual bool equivalent(const ValHolder * v) const
	{
		return prop->equivalent(v->prop);
	};
	virtual void add(const PropertyState * ps) {};
};

class PlaceHolder : public ValHolder {
private:
	static int idGen;
	int id;
	vector<const PropertyState *> allStates;
public:
	PlaceHolder(const pddl_type * pt,const PropertyState * ps,PropertySpace * prpsp) :
		ValHolder(pt,ps,prpsp,0), id(idGen++)
	{
		allStates.push_back(ps);
	};
	void write(ostream & o) const
	{
		o << "{X - ";
		for_each(allStates.begin(),allStates.end(),ptrwriter<const PropertyState>(o," "));
		o << "}";
	};
	bool operator==(const ValHolder & v) const
	{
		const PlaceHolder * p = dynamic_cast<const PlaceHolder *>(&v);
		if(!p) return false;
		return this->ValHolder::operator==(v);
	};
	bool equivalent(const ValHolder * v) const
	{
		const PlaceHolder * p = dynamic_cast<const PlaceHolder *>(v);
		if(!p) return false;
		return this->ValHolder::equivalent(v);
	};
	void add(const PropertyState * ps)
	{
		allStates.push_back(ps);
	};
	typedef vector<const PropertyState*>::const_iterator const_iterator;
	const_iterator begin() const {return allStates.begin();};
	const_iterator end() const {return allStates.end();};
	void showValue(ostream & o) const {o << "X_" << id;};
};

// This holds all the argument types of the "other" arguments in the Property
// ie the arguments of the proposition other than the one defining the properties.
class TypesHolder : public ValHolder {
private:
	vector<pddl_type *> ptypes;
public:
	TypesHolder(const pddl_type * pt,const PropertyState * ps,PropertySpace * psp,Property * p,const vector<pddl_type *> & pts) : 
		ValHolder(pt,ps,psp,p), ptypes(pts) 
	{};
	void write(ostream & o) const 
	{
		vector<pddl_type *>::const_iterator i = ptypes.begin();
		while(true)
		{
			o << (*i)->getName();
			++i;
			if(i==ptypes.end())
			{
				break;
			}
			else
			{
				o << ",";
			};
		};
	};
	bool operator==(const ValHolder & v) const
	{
		const TypesHolder * t = dynamic_cast<const TypesHolder*>(&v);
		if(!t) return false;
		return this->ValHolder::operator==(v);
	};
	bool equivalent(const ValHolder * v) const
	{
		const TypesHolder * t = dynamic_cast<const TypesHolder*>(v);
		if(!t) return false;
		return this->ValHolder::equivalent(v);
	};
};

// This case is used when there are no other arguments.
class NullHolder : public ValHolder {
public:
	NullHolder(const pddl_type * pt,const PropertyState * ps,PropertySpace * psp,Property * p) :
		ValHolder(pt,ps,psp,p)
	{};
	void write(ostream & o) const
	{
		o << "{" << *prop << "}";
	};
	bool operator==(const ValHolder & v) const
	{
		const NullHolder * t = dynamic_cast<const NullHolder*>(&v);
		if(!t) return false;
		return this->ValHolder::operator==(v);
	};
	bool equivalent(const ValHolder * v) const
	{
		const NullHolder * t = dynamic_cast<const NullHolder*>(v);
		if(!t) return false;
		return this->ValHolder::equivalent(v);
	};
};

ostream & operator<<(ostream & o,const ValHolder & vh);

// A ValHolder is just one Property in a PropertyState.
// A Values is the collection of all the things generated for a PropertyState.
// A ValuesUnion is the union of the values for the PropertyStates in one PropertySpace.
// A Range is the product over all PropertySpaces.

class Values {
private:
 	vector<ValHolder *> values;
public:
	Values() : values() {};
	void push_back(ValHolder * v)
	{values.push_back(v);};
	typedef vector<ValHolder *>::const_iterator const_iterator;
	const_iterator begin() const {return values.begin();};
	const_iterator end() const {return values.end();};
	typedef vector<ValHolder *>::iterator iterator;
	iterator begin() {return values.begin();};
	iterator end() {return values.end();};
	size_t size() const {return values.size();};

	bool operator==(const Values & vals) const
	{
		if(values.size() != vals.values.size()) return false;
		for(unsigned int i = 0;i < values.size();++i)
		{
			if(!(*(values[i]) == *(vals.values[i]))) return false;
		};
		return true;
	};
	bool equivalent(const Values & vals) const
	{
		if(values.size() != vals.values.size()) return false;
		for(unsigned int i = 0;i < values.size();++i)
		{
			if(!values[i]->equivalent(vals.values[i])) return false;
		};
		return true;
	};
	ValHolder * & operator[](int i) {return values[i];};
};

class VElement;

class ValuesUnion {
private:
	vector<pair<const PropertyState*,Values> > valuesUnion;
public:
	ValuesUnion(const ValuesUnion & v1,const ValuesUnion & v2);
	ValuesUnion() : valuesUnion() {};
	class const_iterator : public iterator<Values,size_t> {
	private:
		typedef vector<pair<const PropertyState*,Values> >::const_iterator CI;
		CI i;
	public:
		const_iterator(CI j) : i(j) {};
		bool operator==(const const_iterator & x)
		{
			return i == x.i;
		};
		bool operator!=(const const_iterator & x)
		{
			return i != x.i;
		};
		const_iterator & operator++()
		{
			++i;
			return *this;
		};
		const Values & operator *() const
		{
			return i->second;
		};
		const Values * operator->()
		{
			return &(i->second);
		};
		const PropertyState * forState() const {return (i->first);};
	};
	void push_back(const Values & vs,const PropertyState * p) 
	{
		if(find(valuesUnion.begin(),valuesUnion.end(),make_pair(p,vs))==valuesUnion.end())
		{
			valuesUnion.push_back(make_pair(p,vs));
		};
	};
	const_iterator begin() const {return const_iterator(valuesUnion.begin());};
	const_iterator end() const {return const_iterator(valuesUnion.end());};
	size_t size() const {return valuesUnion.size();};
	bool hasPlaceHolder(const PropertyState * ps);
	bool intersectsWith(const ValuesUnion & v) const;
	
};

inline ostream & operator<<(ostream & o,const Values & vls)
{
	o << "(";
	Values::const_iterator i = vls.begin();
	o << **i;
	++i;
	while(i != vls.end())
	{
		o << "," << **i;
		++i;
	};
	o << ")";
	return o;
};

inline ostream & operator<<(ostream & o,const ValuesUnion & vu)
{
	ValuesUnion::const_iterator i = vu.begin();
	o << "(";
	o << *i;
	++i;
	while(i != vu.end())
	{
		o << " U " << *i;
		++i;
	};
	o << ")";
	return o;
};

class Range {
private:
	vector<ValuesUnion> range;
public:
	void push_back(const ValuesUnion & vu) {range.push_back(vu);};
	typedef vector<ValuesUnion>::const_iterator const_iterator;
	const_iterator begin() const {return range.begin();};
	const_iterator end() const {return range.end();};
	size_t size() const {return range.size();};
	void update(const ValuesUnion & oldvu,const ValuesUnion & newvu1,const ValuesUnion & newvu2);
};
	

class VElement {
public:
	virtual ~VElement() {};
	virtual void write(ostream & o) const = 0;
	virtual VElement * copy() const = 0;
	virtual VElement * build(FastEnvironment * fe) {return this;};
	virtual void showValue(ostream & o) const {write(o);};
	virtual bool matches(VElement * vel,FastEnvironment * fenv) = 0;
};

class VarElement : public VElement {
private:
	typedef VAL::parameter_symbol parameter_symbol;
	const parameter_symbol * var;
public:
	VarElement(const parameter_symbol * v) : var(v) {};
	void write(ostream & o) const
	{
		o << var->getName() << "::" << var->type->getName();
	};
	VarElement * copy() const {return new VarElement(*this);};
	VElement * build(FastEnvironment * fe);
	void showValue(ostream & o) const
	{
		o << "?" << var->getName();
	};
	bool matches(VElement * vel,FastEnvironment * fenv)
	{
		if(VarElement * vr = dynamic_cast<VarElement *>(vel))
		{
			return vr->var == var;
		};
		return false;
	};
	const parameter_symbol * getVar() const {return var;};
};

class ObElement : public VElement {
private:
	TIMobjectSymbol * tob;
public:
	ObElement(TIMobjectSymbol * t) : tob(t) {};
	void write(ostream & o) const
	{
		o << *tob;
	};
	ObElement * copy() const {return new ObElement(*this);};
	bool matches(VElement * vel,FastEnvironment * fenv)
	{
		if(VarElement * vr = dynamic_cast<VarElement *>(vel))
		{
			return (*fenv)[vr->getVar()] == tob;
		};
		if(ObElement * ob = dynamic_cast<ObElement *>(vel))
		{
			return ob->tob == tob;
		};
		return false;
	};
};

class PElement : public VElement {
private:
	Property * prop;
public:
	PElement(Property * p) : prop(p) {};
	void write(ostream & o) const
	{
		o << *prop;
	};
	PElement * copy() const {return new PElement(*this);};
	bool matches(VElement * vel,FastEnvironment * fenv)
	{
		if(PElement * pel = dynamic_cast<PElement *>(vel))
		{
			return pel->prop == prop;
		};
		return false;
	};
};

class WildElement : public VElement {
private:
	static int idgen;
	int id;
	ValuesUnion var;
public:
	WildElement(const ValuesUnion & v) : id(idgen++), var(v) {};
	void write(ostream & o) const
	{
		int u = var.size();
		o << "?" << id << " :: (";
		for(ValuesUnion::const_iterator j = var.begin();j != var.end();++j,--u)
		{
			o << "(";
			int ocs = j->size();
			for(Values::const_iterator pp = j->begin();pp != j->end();++pp,--ocs)
			{
				o << **pp;
				if(ocs > 1) o << ",";
			};
			o << ")";
			if(u > 1) o << " U ";
		};
		o << ")";
	};
	WildElement * copy() const {return new WildElement(*this);};
	bool matches(VElement * vel,FastEnvironment * fenv)
	{
		return true;
	};
};

class PlaceHolderElement : public VElement {
private:
	const PlaceHolder * ph;
public:
	PlaceHolderElement(const PlaceHolder * p) : ph(p) {};
	void write(ostream & o) const
	{
		o << *ph;
	};
	PlaceHolderElement * copy() const {return new PlaceHolderElement(ph);};
	void showValue(ostream & o) const
	{
		ph->showValue(o);
	};
	bool matches(VElement * vel,FastEnvironment * fenv)
	{
		if(PlaceHolderElement * pel = dynamic_cast<PlaceHolderElement *>(vel))
		{
			return *ph == *(pel->ph);
		};
		return false;
	};
};

ostream & operator<<(ostream & o,const VElement & v);

class ValueElement {
private:
	const PropertyState * pst;
	vector<VElement *> value;
public:
	ValueElement(const PropertyState * p,const vector<VElement *> & vs) :
		pst(p), value(vs) 
	{};
	ValueElement(ValueElement * vel,FastEnvironment * fe);

	void write(ostream & o) const
	{
		o << "(";
		for_each(value.begin(),value.end(),ptrwriter<VElement>(o," "));
		o << ")";
	};
	const PropertyState * getPS() const {return pst;};
	VElement * operator[](int i) {return value[i];};
	size_t size() const {return value.size();};
	void showValue(ostream & o) const
	{
		o << "(";
		for(vector<VElement*>::const_iterator i = value.begin();i != value.end();)
		{
			(*i)->showValue(o);
			++i;
			if(i != value.end()) o << ",";
		};
		o << ")";
	};
	bool matches(ValueElement * vel,FastEnvironment * fenv)
	{
		if(value.size() != vel->value.size()) return false;
		for(size_t i = 0;i < value.size();++i)
		{
			if(!value[i]->matches(vel->value[i],fenv)) return false;
		};
		return true;
	};
};

ostream & operator << (ostream & o,const ValueElement & ve);

class RangeRep;

typedef map<TIMobjectSymbol *,vector<ValueElement *> > ElementRanges;
typedef map<const pddl_type *,map<const TIMobjectSymbol *,RangeRep *> > Reachables;

class ValueStructure {
private:
	Range range;
	pddl_type * pt;

	ElementRanges rngs;
	
public:
	ValueStructure(pddl_type * p) : pt(p) {};
	
	void add(const ValuesUnion & vu) {range.push_back(vu);};

	void write(ostream & o) const
	{
		int c = range.size();
		for(Range::const_iterator i = range.begin();i != range.end();++i,--c)
		{
			o << "(";
			int u = i->size();
			for(ValuesUnion::const_iterator j = i->begin();j != i->end();++j,--u)
			{
				o << "(";
				int ocs = j->size();
				for(Values::const_iterator pp = j->begin();pp != j->end();++pp,--ocs)
				{
					o << **pp;
					if(ocs > 1) o << ",";
				};
				o << ")";
				if(u > 1) o << " U ";
			};
			o << ")";
			if(c > 1) o << " X ";
		};
	};

	void initialise();
	const Range & getRange() const {return range;};
	const pddl_type * getType() const {return pt;};
	void update(const ValuesUnion & oldvu,const ValuesUnion & newvu1,const ValuesUnion & newvu2);
	void liftFrom(ValueStructure & vs1,ValueStructure & vs2);
	void setUpInitialState(Reachables & reachables);
};

ostream & operator<<(ostream & o,const ValueStructure & vs);



typedef std::map<const pddl_type *,ValueStructure> FunctionRep;

class ValueStruct;
class SASActionTemplate;
class ValueRep;

class FunctionStructure {
private:
	FunctionRep frep;
	vector<const pddl_type *> noStates;
	typedef map<const operator_ *,SASActionTemplate*> SASActionTemplates;
	SASActionTemplates sasActionTemplates;

	void restructure(const vector<ValueStruct> & red,const vector<ValueStruct> & lve,
						const PropertyState * ps);
						
	Reachables reachables;
	map<const operator_ *,pair<int,int> > startOp;
	int levels;
	vector<int> unsatisfiedPrecs;

	vector<VAL::proposition *> others;
	vector<int> othercounts;

public:
	FunctionStructure();

	void initialise();
	void processActions();
	const ValueStructure & forType(const pddl_type * pt) {return frep.find(pt)->second;};
	bool hasFluent(const pddl_type * pt) const {return frep.find(pt) != frep.end();};
	void normalise();
	void restructure(const operator_ * op,const var_symbol * prm,
							const vector<const pddl_type *> & rtps);
	void setUpInitialState();
	bool growOneLevel();
	int startFor(const operator_ * op) const {return startOp.find(op)->second.first;};
	int endFor(const operator_ * op) const {return startOp.find(op)->second.second;};
	bool tryMatchedPre(int k,instantiatedOp * iop,const var_symbol * var,
							SASActionTemplate * sasact,ValueRep * vrep);
	void buildLayers();
	typedef SASActionTemplates::const_iterator iterator;
	iterator begin() const {return sasActionTemplates.begin();};
	iterator end() const {return sasActionTemplates.end();};
};


};

#endif
