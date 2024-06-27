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

#ifndef __TYPECHECK
#define __TYPECHECK

#include "ptree.h"
#include <set>
#include <vector>

using std::set;
using std::vector;

//#define vector std::vector

namespace VAL {

class PTypeRef;
class UTypeRef;
class TypeHierarchy;

struct TypeRef {
	virtual ~TypeRef(){};
	virtual bool operator<(const TypeRef & t) const = 0;
	virtual bool operator>(const PTypeRef & p) const = 0;
	virtual bool operator>(const UTypeRef & u) const = 0;
	virtual bool operator==(const TypeRef & t) const = 0;
	virtual TypeRef * clone() const = 0;
	virtual bool operator==(const PTypeRef & p) const {return false;};
	virtual bool operator==(const UTypeRef & u) const {return false;};
	virtual bool expected() const {return true;};
	virtual void addContents(TypeHierarchy * th) const {};
	virtual void show() const = 0;
	virtual	const pddl_type * operator *() const = 0;

};


class PTypeRef : public TypeRef {
private:
	const pddl_type * pt;
public:
	PTypeRef(const pddl_type * p) : pt(p) {};
	bool operator<(const TypeRef & t) const
	{
		return t > *this;
	};
	bool operator>(const PTypeRef & p) const
	{
		return p.pt < pt;
	};
	bool operator>(const UTypeRef & u) const
	{
		return false;
	};
	bool operator==(const TypeRef & t) const
	{
		return t==*this;
	};
	bool operator==(const PTypeRef & p) const
	{
		return pt == p.pt;
	};
	TypeRef * clone() const 
	{
		return new PTypeRef(*this);
	};
	void show() const
	{
		cout << *pt << "\n";
	};
	const pddl_type * operator *() const
	{
		return pt;
	};
};

class UTypeRef : public TypeRef {
private:
	//const 
		set<const pddl_type *> pts;
public:
	UTypeRef(const pddl_type_list* ps) //: pts(ps->begin(),ps->end())
	{
		for(pddl_type_list::const_iterator i = ps->begin();i != ps->end();++i)
		{
			pts.insert(*i);
		};	
	};

	UTypeRef() {};

	bool operator<(const TypeRef &t) const
	{
		return t > *this;
	};
	bool operator>(const PTypeRef & p) const
	{
		return true;
	};
	bool operator>(const UTypeRef & u) const
	{
		return u.pts < pts;
	};
	bool operator==(const TypeRef & t) const
	{
		return t == *this;
	};
	bool operator==(const UTypeRef & u) const
	{
		return pts == u.pts;
	};
	TypeRef * clone() const 
	{
		return new UTypeRef(*this);
	};
	bool expected() const {return false;};
	void addContents(TypeHierarchy*) const;
	void show() const 
	{
		cout << "UType\n";
	};
	const pddl_type * operator *() const
	{
		return 0;
	};

};

struct TRcompare : public std::binary_function<const TypeRef*,const TypeRef*,bool> {

	bool operator()(const TypeRef * t1,const TypeRef * t2) const
	{
		return *t1 < *t2;
	};
};

class TypeHierarchy  {
private:
	typedef set<const TypeRef *> Nodes;
	typedef map<const TypeRef *, Nodes, TRcompare> Graph;
	typedef Graph::iterator GI;
	typedef Graph::const_iterator GIC;
	
	Graph graph;
	bool closure(Graph & gr,GI & gi,Nodes & vs,GI & gs,const TypeRef * t);
	TypeHierarchy(const TypeHierarchy & th);

	void addDown(const PTypeRef & t1,const PTypeRef & t2);
	Graph downGraph;
	Graph leafNodes;
	
public:
	TypeHierarchy(const analysis * a);
	~TypeHierarchy();
	bool reachable(const TypeRef & t1,const TypeRef & t2);
	void add(const PTypeRef & t,const TypeRef & u);

	const Nodes & leaves(PTypeRef & t);
	vector<const pddl_type *> accumulateAll(const pddl_type * t);
};


class TypeChecker {
private:
	const analysis *thea;
	TypeHierarchy th;
	const bool isTyped;

public:
	TypeChecker(const analysis * a) : thea(a), th(a), isTyped(a->the_domain->types) {};
	bool typecheckDomain();
	bool typecheckAction(const operator_ * act);
	bool typecheckProblem();
	bool typecheckPlan(const plan * p);
	bool typecheckGoal(const goal * g);
	bool typecheckProposition(const proposition * g);
	bool typecheckActionInstance(const plan_step * p);
	bool typecheckDerivationRule(const derivation_rule * d);
	bool typecheckEffect(const effect * e);
	bool typecheckEffects(const effect_lists * e);
	bool typecheckFuncTerm(const func_term * f);
	bool typecheckExpression(const expression * e);
	bool subType(const pddl_typed_symbol *,const pddl_typed_symbol *);
	bool subType(const pddl_type *,const pddl_typed_symbol *);
	bool subType(const pddl_type *,const pddl_type *);

	vector<const_symbol *> range(const var_symbol * v);
	vector<const_symbol *> range(const parameter_symbol * v);
	vector<const_symbol *> range(const pddl_type * t);
	vector<const pddl_type *> leaves(const pddl_type * t);
	vector<const pddl_type *> accumulateAll(const pddl_type * t);
	bool isLeafType(const pddl_type * t);
};

};

#endif
