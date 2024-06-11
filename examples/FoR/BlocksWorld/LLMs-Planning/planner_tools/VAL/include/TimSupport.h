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

#ifndef __TIMSUPPORT
#define __TIMSUPPORT

#include <algorithm>
#include <iostream>
#include <set>
#include <memory>
#include <iterator>
#include <string.h>

#include "TIMUtilities.h"
#include "TypedAnalyser.h"

#define OUTPUT if(getenv("TIMOUT")) 
#define OUTPUT1 if(getenv("TIMOUT") && !strcmp(getenv("TIMOUT"),"Hi"))

using std::multiset;
using std::make_pair;
using std::max;
using std::min;
using std::for_each;
using std::bind2nd;
using std::not1;
using std::mem_fun;


namespace TIM {

class PropertySpace;
class TIMobjectSymbol;
class TIMAnalyser;

class Property {
private:
	VAL::pred_symbol * predicate;
	int posn;

	vector<PropertySpace*> belongTo;
	vector<TIMobjectSymbol*> exhibitors;

	bool isSV;
	bool isReq;
	
public:
	Property() : predicate(0), isSV(false), isReq(false) {};
	
	Property(VAL::pred_symbol * p,int a) : predicate(p), posn(a), isSV(false), isReq(false) {};
	void setSV(bool sv,bool rq)
	{	
		isSV = sv;
		isReq = rq;
	};
	bool isSingleValued() const {return isSV;};
	bool isRequired() const {return isReq;};
	void write(ostream & o) const 
	{
		o << predicate->getName() << "_" << posn;

/*
 		o << "[";
		
		for(vector<pddl_type *>::iterator i = EPS(predicate)->tBegin();
			i != EPS(predicate)->tEnd();++i)
		{
			o << (*i)->getName() << " ";
		};
		o << "]";
*/
	};

	void addIn(PropertySpace * p)
	{
		belongTo.push_back(p);
	};

	typedef vector<PropertySpace *>::iterator SpaceIt;
	SpaceIt begin() {return belongTo.begin();};
	SpaceIt end() {return belongTo.end();};

	void add(TIMobjectSymbol * t) 
	{
		exhibitors.push_back(t);
	};

	typedef vector<TIMobjectSymbol *>::iterator ObjectIt;
	ObjectIt oBegin() {return exhibitors.begin();};
	ObjectIt oEnd() {return exhibitors.end();};

	Property * getBaseProperty(const VAL::pddl_type * pt) const;
	vector<Property *> matchers();
	bool matches(const VAL::extended_pred_symbol * prop,VAL::pddl_type * pt);

	bool applicableTo(VAL::TypeChecker & tc,const VAL::pddl_type * tp) const;
	int familySize() const {return EPS(predicate)->arity();};
	int aPosn() const {return posn;};
	const VAL::extended_pred_symbol * root() const {return EPS(predicate);};
	bool equivalent(const Property * p) const;
};

ostream & operator<<(ostream & o, const Property & p);

struct setUpProps {	
	unsigned int a;
	VAL::pred_symbol * pred;
	
	setUpProps(VAL::pred_symbol * p) : 
		a(0), pred(p) {};

	void operator()(Property & p) 
	{
		p = Property(pred,a);
		++a;
	};
};

class TIMpredSymbol : public VAL::extended_pred_symbol {
private:
	vector<Property> props; 
	typedef map<TIMpredSymbol *,vector<pair<int,int> > > MutexRecords;
	MutexRecords mutexes;
	
public:
	TIMpredSymbol(VAL::pred_symbol * p,VAL::proposition * q) : 
		extended_pred_symbol(p,q), props(q->args->size()) 
	{
		for_each(props.begin(),props.end(),setUpProps(this));
	};
	template<class TI>
	TIMpredSymbol(pred_symbol * p,TI s,TI e) : 
		extended_pred_symbol(p,s,e), props(e-s) 
	{
		for_each(props.begin(),props.end(),setUpProps(this));
		EPS(p)->getParent()->add(this);
	};
	Property * property(int a) {return &(props[a]);}; 
	void setMutex(int p1,TIMpredSymbol * tps,int p2)
	{
		mutexes[tps].push_back(make_pair(p1,p2));
	};
	template<typename TI>
	bool checkMutex(TI sa,TI ea,TIMpredSymbol * pb,TI sb,TI eb) 
	{
		MutexRecords::iterator i = mutexes.find(pb);
		if(i == mutexes.end()) return false;
		if(pb == this)
		{
			for(vector<pair<int,int> >::const_iterator p = i->second.begin();
					p != i->second.end();++p)
			{
				OUTPUT cout << "Examining " << p->first << " " << p->second << "\n";
				if(*(sa+p->first) == *(sb+p->second))
				{
					// Same object has two copies of the same property.
					// Now check whether they are identical propositions.
					for(;sa != ea;++sa,++sb)
					{
						if(*sa != *sb) return true;
					};
				};
			};
		}
		else
		{
			for(vector<pair<int,int> >::const_iterator p = i->second.begin();
					p != i->second.end();++p)
			{
				OUTPUT cout << "Examining " << p->first << " " << p->second << "\n";
			// Same value has two mutex properties...
				if(*(sa+p->first) == *(sb+p->second)) return true;
			};
		};
		return false;
	};
};

#define TPS(x) static_cast<TIMpredSymbol*>(x)
#define cTPS(x) const_cast<TIMpredSymbol *>(static_cast<const TIMpredSymbol*>(x))

class TIMobjectSymbol : public VAL::const_symbol {
private:
	vector<Property*> initial;
	vector<VAL::proposition *> initialps;
	vector<Property*> final;
	vector<PropertySpace *> spaces;

public:
	TIMobjectSymbol(const string & s) : const_symbol(s) {};
	void addInitial(Property*p,VAL::proposition * prp) 
	{
		initial.push_back(p);
		initialps.push_back(prp);
	};
	void addFinal(Property*p) {final.push_back(p);};
	void addIn(PropertySpace * p) {spaces.push_back(p);};
	void distributeStates(TIMAnalyser * tan);
	void write(ostream & o) const
	{
		o << getName();
	};

	const vector<VAL::proposition*> & getInits() const {return initialps;};
	VAL::proposition * find(const Property * p) const
	{
		for(vector<Property*>::const_iterator i = initial.begin();i != initial.end();++i)
		{
			if(p && p->equivalent(*i))
			{
				return initialps[i-initial.begin()];
			};
		};
		return 0;
/*		
		vector<Property*>::const_iterator i = std::find(initial.begin(),initial.end(),p);
		if(i == initial.end())
		{
			for(vector<Property*>::const_iterator ii = initial.begin();ii != initial.end();++ii)
			{
				vector<Property*> ps = (*ii)->matchers();
				vector<Property*>::const_iterator j = std::find(ps.begin(),ps.end(),p);
				if(j != ps.end())
				{
					return initialps[ii-initial.begin()];
				};
			};
			return 0;
		};
		return initialps[i-initial.begin()];
*/
	};
};

ostream & operator <<(ostream & o,const TIMobjectSymbol & t);

#define TOB(x) static_cast<TIM::TIMobjectSymbol*>(x)

class PropertyState {
private:
	typedef multiset<Property *> Properties;
	typedef CascadeMap<Property *,PropertyState> PMap;
	static PMap pmap;

	TIMAnalyser * tan;
	
	Properties properties;

	template<class TI>
	PropertyState(TIMAnalyser * t,TI s,TI e) : tan(t), properties(s,e) {};

	template<class TI>
	static PropertyState * retrieve(TIMAnalyser * tan,TI s,TI e)
	{
		PropertyState * & ps = pmap.forceGet(s,e);
		if(ps==0)
		{
			ps = new PropertyState(tan,s,e);
		};
		return ps;
	};
	
public:
	template<class TI>
	static PropertyState * getPS(TIMAnalyser * tan,const VAL::pddl_type * pt,TI s,TI e)
	{
		vector<Property *> props;
		transform(s,e,inserter(props,props.begin()),
					bind2nd(mem_fun(&Property::getBaseProperty),pt));
		return retrieve(tan,props.begin(),props.end());
	};
	
	void write(ostream & o) const
	{
		o << "{";
		for_each(properties.begin(),properties.end(),
					ptrwriter<Property>(o," "));
		o << "}";
	};
	int count(Property * p) const
	{
		return std::count(properties.begin(),properties.end(),p);
	};

	bool contains(Property * p) const
	{
		return std::find(properties.begin(),properties.end(),p) != properties.end();
	};
	
	bool empty() const
	{
		return properties.empty();
	};

	size_t size() const
	{
		return properties.size();
	};

	typedef Properties::const_iterator PSIterator;

	PSIterator begin() const {return properties.begin();};
	PSIterator end() const {return properties.end();};
	PropertyState * adjust(const PropertyState * del,const PropertyState * add)
	{
// Simple implementation is to remove del from this and then check that the result
// found all entries in del. If so, union add and return a new PropertyState, else
// return 0.
// 
// There is probably an issue over types though. For example, if dels are for a more
// specialised type than the entries in this what should happen? 

		vector<Property *> ps;
		set_difference(properties.begin(),properties.end(),
						del->properties.begin(),del->properties.end(),
						inserter(ps,ps.begin()));
		if(ps.size() + del->properties.size() == properties.size())
		{
			vector<Property *> qs;
			merge(ps.begin(),ps.end(),add->properties.begin(),add->properties.end(),
						inserter(qs,qs.begin()));
			return retrieve(tan,qs.begin(),qs.end());
		}
		else
		{
			return 0;
		};
	};
	pair<PropertyState *,PropertyState *> split(Property *);
	template<class TI>
	PropertyState * add(TI s,TI e)
	{
		if(s==e) return this;
		vector<Property *> qs;
		merge(properties.begin(),properties.end(),s,e,inserter(qs,qs.begin()));
		return retrieve(tan,qs.begin(),qs.end());
	};
};

ostream & operator<<(ostream & o,const PropertyState & p);

class TransitionRule;
class mRec;

struct recordIn {
	PropertySpace * ps;

	recordIn(PropertySpace * p) : ps(p) {};

	void operator()(Property * p)
	{
		p->addIn(ps);
	};
};

struct countInState {
	Property * prop;

	countInState(Property * p) : prop(p) {};
	int operator()(PropertyState* ps) 
	{
		return ps->count(prop);
	};
};

struct recordSV {
	PropertySpace * ps;
	vector<Property *> & sv;
	
	recordSV(PropertySpace * p,vector<Property *> & s) : ps(p), sv(s) {};

	void operator()(Property * p);
};

// Two facts will be mutex if they define properties that appear in different
// states of the same (SV) property space, or if they define properties that are 
// different instantiations for the same object when only one instance of the
// property appears in a state in a (SV) property space.

class PropertySpace {
private:
	set<PropertyState *> states;
	set<TransitionRule *> rules;
	vector<Property *> properties;
	vector<TIMobjectSymbol *> objects;

	bool isStateValued;

	bool isLS;
	bool LSchecked;
	
public:
	PropertySpace(Property * p,TransitionRule * t);
	PropertySpace(Property * p) : 
		states(), rules(), properties(1,p), objects(), isStateValued(true) {};
	void checkStateValued();
	void merge(PropertySpace * ps)
	{
		copy(ps->states.begin(),ps->states.end(),
					inserter(states,states.end()));
		copy(ps->rules.begin(),ps->rules.end(),
					inserter(rules,rules.end()));
		copy(ps->properties.begin(),ps->properties.end(),
					inserter(properties,properties.end()));
		copy(ps->objects.begin(),ps->objects.end(),
					inserter(objects,objects.end()));
		isStateValued &= ps->isStateValued;
		
		delete ps;
	};
	vector<int> countsFor(Property * p)
	{
		vector<int> cs;
		transform(states.begin(),states.end(),inserter(cs,cs.begin()),countInState(p));
		return cs;
	};
	void checkSV(vector<Property *> & sv)
	{
		for_each(properties.begin(),properties.end(),recordSV(this,sv));
	};
	void add(TransitionRule * t);
	PropertySpace * finalise()
	{
		for_each(properties.begin(),properties.end(),recordIn(this));
		return this;
	};
	void assembleMutexes();
	void assembleMutexes(TransitionRule *);
//	void assembleMutexes(TransitionRule *,Property *);
	void assembleMutexes(Property *);
	void assembleMutexes(Property *,Property *);
	void assembleMutexes(VAL::operator_ *,const mRec &);
	void recordRulesInActions();
	void add(PropertyState * ps) {states.insert(ps);};
	void add(TIMobjectSymbol * t) {objects.push_back(t);};

	void write(ostream & o) const;	
	bool isState() const {return isStateValued;};
	bool isStatic() const {return rules.empty();};
	void sortObjects() {sort(objects.begin(),objects.end());};
	bool contains(TIMobjectSymbol * to) const;
	typedef vector<TIMobjectSymbol *>::const_iterator OIterator;
	OIterator obegin() const {return objects.begin();};
	OIterator oend() const {return objects.end();};
	bool extend();

	bool examine(vector<PropertySpace*> &);
	PropertySpace * slice(Property * p);

	bool applicableTo(VAL::TypeChecker & tc,const VAL::pddl_type * tp) const;

	typedef set<PropertyState*>::const_iterator SIterator;
	SIterator begin() const {return states.begin();};
	SIterator end() const {return states.end();};
	int numStates() const {return states.size();};
	bool isLockingSpace();
};

ostream & operator<<(ostream & o,const PropertySpace & p);

class rulePartitioner;
class RuleObjectIterator;

enum opType {INSTANT = 0,START = 1,MIDDLE = 2,END = 3};

class TransitionRule {
private:
	TIMAnalyser * tan;
	VAL::operator_ * op;
	VAL::derivation_rule * drv;
	opType opt;
	int var;
	PropertyState * enablers;
	PropertyState * lhs;
	PropertyState * rhs;

	vector<VAL::const_symbol*> objects;
	
	friend class rulePartitioner;
	friend class RuleObjectIterator;

	TransitionRule(TransitionRule * t,PropertyState * e,PropertyState * l,PropertyState * r);
	
public:
	TransitionRule(TIMAnalyser * t,VAL::operator_ * o,int v,
					PropertyState * e,PropertyState * l,PropertyState * r,
					opType ty = INSTANT);

	TransitionRule(TIMAnalyser * t,VAL::derivation_rule * o,int v,
					PropertyState * e,PropertyState * l,PropertyState * r,
					opType ty = INSTANT);
	
	bool isTrivial() const
	{
		return lhs->empty() && rhs->empty();
	};
	bool isAttribute() const
	{
		return lhs->empty() || rhs->empty();
	};
	bool isIncreasing() const
	{
		return lhs->empty() && !rhs->empty();
	};
	bool isDecreasing() const
	{
		return rhs->empty() && !lhs->empty();
	};

	void distributeEnablers();

	void write(ostream & o) const
	{
		o << (*enablers) << " => " << (*lhs) << " -> " << (*rhs) <<
			(isAttribute()?" attribute rule: ":"") <<
			(isIncreasing()?"increasing":"") <<
			(isDecreasing()?"decreasing":"");
	};

	RuleObjectIterator beginEnabledObjects();
	RuleObjectIterator endEnabledObjects();
	PropertyState * tryRule(PropertyState * p)
	{
		return p->adjust(lhs,rhs);
	};
	void assembleMutex(TransitionRule *);
	void assembleMutex(VAL::operator_*,const mRec & pr);
	Property * candidateSplit();
	void recordInAction(PropertySpace * p);
	int paramNum() const {return var;};
	TransitionRule * splitRule(Property * p);
	const PropertyState * getLHS() const {return lhs;};
	const PropertyState * getRHS() const {return rhs;};
	const PropertyState * getEnablers() const {return enablers;};
	const VAL::operator_ * byWhat() const {return op;};
	bool applicableIn(const PropertyState * p) const;
};

ostream & operator<<(ostream & o,const TransitionRule & tr);

typedef vector<TransitionRule *> TRules;


struct ProtoRule {

	TIMAnalyser * tan;
	VAL::operator_ * op;
	VAL::derivation_rule * drv;
	opType opt;
	int var;
	
	vector<Property *> enablers;
	vector<Property *> adds;
	vector<Property *> dels;

	ProtoRule(TIMAnalyser * t,VAL::operator_ * o,int v,opType ty = INSTANT) : 
		tan(t), op(o), drv(0), opt(ty), var(v) {};

	ProtoRule(TIMAnalyser * t,VAL::derivation_rule * o,int v,opType ty = INSTANT) : 
		tan(t), op(0), drv(o), opt(ty), var(v) {};
	
	void insertPre(Property * p)
	{
		enablers.push_back(p);
	};
	void insertAdd(Property * p)
	{
		adds.push_back(p);
	};
	void insertDel(Property * p)
	{
		dels.push_back(p);
	};

	void addRules(TRules & trules);

};


struct processRule {
	TRules & trules;

	processRule(TRules & tr) : trules(tr) {};

	void operator()(ProtoRule * pr)
	{
		if(!pr) return;
		pr->addRules(trules);
		delete pr;
	};
};

struct doExtension {
	bool again;

	doExtension() : again(false) {};

	void operator()(PropertySpace * p)
	{
		again |= p->extend();
	};

	operator bool() {return again;};
};

struct doExamine {
	TIMAnalyser * tan;
	vector<PropertySpace *> newas;

	doExamine(TIMAnalyser * t) : tan(t) {};

	void operator()(PropertySpace * p);

	operator vector<PropertySpace *>() {return newas;};
};

// Need this because mem_fun won't work with non-const methods like sortObjects.
inline void sortObjects(PropertySpace * p)
{
	p->sortObjects();
};

class TIMpred_decl : public VAL::pred_decl {
public:
	TIMpred_decl() : pred_decl(0,0,0) {};
	TIMpred_decl(VAL::pred_symbol * h,VAL::var_symbol_list * a,VAL::var_symbol_table * vt) :
		pred_decl(h,a,vt) {};
	~TIMpred_decl() 
	{
		args = 0;
		var_tab = 0;
	};
};

class DurativeActionPredicateBuilder : public VAL::VisitController {
private:
	bool inserting;	
	vector<VAL::pred_symbol *> toIgnore;
	VAL::durative_action * replacePreconditionsOf;
public:
	DurativeActionPredicateBuilder() : VisitController(), inserting(true) {};
	const vector<VAL::pred_symbol *> & getIgnores() const {return toIgnore;};
	
	void reverse() {inserting = false;};
	virtual void visit_conj_goal(VAL::conj_goal * cg) {        
        using namespace VAL;

        replacePreconditionsOf->precondition = cg->getGoals()->front();
        const_cast<goal_list*>(cg->getGoals())->pop_front();
    }

    virtual void visit_timed_goal(VAL::timed_goal* ) {
        replacePreconditionsOf->precondition = 0;
    } 
	
	virtual void visit_durative_action(VAL::durative_action * p)
	{
		using namespace VAL;
//		cout << "Treating " << p->name->getName() << "\n";
		if(inserting) {
			pred_symbol * nm = current_analysis->pred_tab.symbol_put(p->name->getName());
			toIgnore.push_back(nm);
			pred_decl * pd = new TIMpred_decl(nm,p->parameters,p->symtab);
			current_analysis->the_domain->predicates->push_front(pd);
			effect_lists * es = new effect_lists;
			effect_lists * ee = new effect_lists;
			timed_effect * ts = new timed_effect(es,E_AT_START);
			es->add_effects.push_front(new simple_effect(new proposition(nm,p->parameters)));
			timed_effect * te = new timed_effect(ee,E_AT_END);
			ee->del_effects.push_front(new simple_effect(new proposition(nm,p->parameters)));
			p->effects->timed_effects.push_front(ts);
			p->effects->timed_effects.push_front(te);
			timed_goal * tg = new timed_goal(new simple_goal(new proposition(nm,p->parameters),E_POS),E_OVER_ALL);
			if(p->precondition) 
			{
				goal_list * gs = new goal_list;
				gs->push_front(tg);
				gs->push_front(p->precondition);
				conj_goal * cg = new conj_goal(gs);
				p->precondition = cg;
			} else {
				p->precondition = tg;
			}
		}
		else
		{
			timed_effect * t = p->effects->timed_effects.front();
			p->effects->timed_effects.pop_front();
			delete t;
			t = p->effects->timed_effects.front();
			p->effects->timed_effects.pop_front();
			delete t;
			replacePreconditionsOf = p;
			goal * oldprecondition = p->precondition;
			p->precondition->visit(this);
			delete oldprecondition;
		};
	};

	virtual void visit_domain(VAL::domain * p) 
	{
		visit_operator_list(p->ops);
	};
};

struct CheckSV {
	vector<Property *> & sv;

	CheckSV(vector<Property*> & s) : sv(s) {};

	void operator()(PropertySpace * ps)
	{
		ps->checkSV(sv);
	};
};

class TIMactionSymbol : public VAL::operator_symbol {
private:
	vector<PropertySpace*> stateChanger;
	vector<TransitionRule*> rules;
	bool fixedDuration;

public:
	TIMactionSymbol(const string & nm) : operator_symbol(nm), fixedDuration(false) {};
	void addStateChanger(PropertySpace * ps,TransitionRule * tr)
	{
		stateChanger.push_back(ps);
		rules.push_back(tr);
	};
	void write(ostream & o) const
	{
		o << name;
		if(fixedDuration) o << "!";
	};
	typedef vector<TransitionRule*>::const_iterator RCiterator;
	RCiterator begin() const {return rules.begin();};
	RCiterator end() const {return rules.end();};
	bool hasRuleFor(int prm) const;
	bool isFixedDuration() const
	{
		return fixedDuration;
	};
	void assertFixedDuration()
	{
		fixedDuration = true;
	};
};

inline ostream & operator<<(ostream & o,const TIMactionSymbol & a)
{
	a.write(o);
	return o;
};

#define TAS(x) static_cast<TIM::TIMactionSymbol*>(x)
#define TASc(x) static_cast<const TIM::TIMactionSymbol* const>(x)

class TIMAnalyser : public VAL::VisitController {
private:
	VAL::TypeChecker & tcheck;
	VAL::analysis * an;
	VAL::FuncAnalysis fan;
	
	bool adding;
	bool initially;
	bool finally;

	bool isDurative;
	bool atStart;
	bool overall;
	
	VAL::operator_ * op;
	VAL::derivation_rule * drv;
	vector<ProtoRule *> rules;
	TRules trules;
	vector<PropertySpace *> propspaces;
	vector<PropertySpace *> attrspaces;
	vector<PropertySpace *> staticspaces;

	vector<Property*> singleValued;
	
	void setUpSpaces();

	static void assembleMutexes(PropertySpace *);
	static void recordRulesInActions(PropertySpace *);

	friend class doExamine;	
	
public:
	TIMAnalyser(VAL::TypeChecker & tc,VAL::analysis * a) :
	    tcheck(tc), an(a), fan(a->func_tab), 
	    adding(true), initially(false), finally(false), 
	    isDurative(false), overall(false), op(0) ,drv(0)
	{};
	VAL::TypeChecker & getTC() {return tcheck;};
	
	void insertPre(int v,Property * p);
	void insertEff(int v,Property * p);
	void insertGoal(VAL::parameter_symbol * c,Property * p);
	void insertInitial(VAL::parameter_symbol * c,Property * p,VAL::proposition * prp);

	virtual void visit_simple_goal(VAL::simple_goal * p);
	virtual void visit_qfied_goal(VAL::qfied_goal * p) 
	{OUTPUT cout << "Quantified goal\n";};
	virtual void visit_conj_goal(VAL::conj_goal * p) 
	{p->getGoals()->visit(this);};
	virtual void visit_disj_goal(VAL::disj_goal * p) 
	{OUTPUT cout << "Disjunctive goal\n";};
	virtual void visit_timed_goal(VAL::timed_goal * p) 
	{
		using namespace VAL;
		if(p->getTime() == (atStart?E_AT_START:E_AT_END) || (overall && p->getTime()==E_OVER_ALL))
			p->getGoal()->visit(this);
	};	
	virtual void visit_imply_goal(VAL::imply_goal * p) 
	{
		OUTPUT cout << "Implication goal\n";
	};
	virtual void visit_neg_goal(VAL::neg_goal * p) 
	{
		OUTPUT cout << "Negative goal\n";
	};
	virtual void visit_simple_effect(VAL::simple_effect * p);
	virtual void visit_simple_derivation_effect(VAL::derivation_rule * p);
	virtual void visit_forall_effect(VAL::forall_effect * p) 
	{
		OUTPUT cout << "Quantified effect\n";
	};
	virtual void visit_cond_effect(VAL::cond_effect * p) 
	{
		OUTPUT cout << "Conditional effect\n";
	};
	virtual void visit_timed_effect(VAL::timed_effect * p) 
	{
		using namespace VAL;
		if(p->ts==(atStart?E_AT_START:E_AT_END))
			p->effs->visit(this);
	};
	virtual void visit_effect_lists(VAL::effect_lists * p) 
	{
		using namespace VAL;
		p->add_effects.pc_list<simple_effect*>::visit(this);
		p->forall_effects.pc_list<forall_effect*>::visit(this);
		p->cond_effects.pc_list<cond_effect*>::visit(this);
		p->timed_effects.pc_list<timed_effect*>::visit(this);
		bool whatwas = adding;
		adding = !adding;
		p->del_effects.pc_list<simple_effect*>::visit(this);
		adding = whatwas;
	};
	virtual void visit_derivation_rule(VAL::derivation_rule * p) 
	{
		drv = p;
		adding = true;
		rules = vector<ProtoRule*>(p->get_head()->args->size(),0);
		p->get_body()->visit(this);
		visit_simple_derivation_effect(p);
		for_each(rules.begin(),rules.end(),processRule(trules));
		drv = 0;
	};
	virtual void visit_operator_(VAL::operator_ * p) 
	{
		op = p;
		adding = true;
		rules = vector<ProtoRule*>(p->parameters->size(),0);
		p->precondition->visit(this);
		p->effects->visit(this);
		for_each(rules.begin(),rules.end(),processRule(trules));
		op = 0;
	};
	virtual void visit_action(VAL::action * p)
	{
		visit_operator_(p);
	}
	virtual void visit_durative_action(VAL::durative_action * p) 
	{
// I think that we can do this in two stages - the at start and the at end.
// We can have a filter on timed goals and effects that decides whether it
// is relevant. Might need to store an optype flag with op, so that we can
// tell whether we generated from a start or end point.
// 
// The tricky bit is the linkage: we need record invariants and we also need
// to ensure that we don't lose state change across durative actions. Toni's
// idea is to have a dummy add effect at start that is preconditioned and deleted
// at the end. That should work, but needs a couple of tweaks:
// 1: If the action has several state change effects then this technique will leave
//    them linked together in one state space. Not entirely clear how these could be 
//    split, but maybe we can use a technique that generalises the state splitting
//    idea (ab->c, c->ab becomes a->c, c->a and b->c', c'->b).
// 2: The mechanism artificially creates increasing/decreasing effects if there was
//    actually no property that was unlinked before introducing the dummies. We can
//    filter these cases out, I think.
//    
		
		isDurative = true;
		atStart = true;
		overall = false;
		visit_operator_(p);
		atStart = false;
		visit_operator_(p);
		overall = true;
		visit_operator_(p);
		overall = false;
		isDurative = false;
	};
	virtual void visit_domain(VAL::domain * p) 
	{
		visit_operator_list(p->ops);
		if (p->drvs) visit_derivations_list(p->drvs);
		setUpSpaces();
	};
	virtual void visit_problem(VAL::problem * p)
	{
		initially = true;
		p->initial_state->visit(this);
		initially = false;
		finally = true;
		if(p->the_goal) p->the_goal->visit(this);
		finally = false;
		if(p->objects) p->objects->visit(this);
		for_each(propspaces.begin(),propspaces.end(),&sortObjects);
		vector<PropertySpace*>::iterator a = 
			partition(propspaces.begin(),propspaces.end(),
					mem_fun(&PropertySpace::isState));		
		copy(a,propspaces.end(),inserter(attrspaces,attrspaces.begin()));
		propspaces.erase(a,propspaces.end());
		a = partition(propspaces.begin(),propspaces.end(),
					not1(mem_fun(&PropertySpace::isStatic)));
		copy(a,propspaces.end(),inserter(staticspaces,staticspaces.end()));
		propspaces.erase(a,propspaces.end());

		while(for_each(attrspaces.begin(),attrspaces.end(),doExtension()));
		while(for_each(propspaces.begin(),propspaces.end(),doExtension()));

		OUTPUT1 {
			for_each(trules.begin(),trules.end(),ptrwriter<TransitionRule>(cout,"\n"));
			for_each(propspaces.begin(),propspaces.end(),
						ptrwriter<PropertySpace>(cout,"\n"));
		};
		
		for_each(propspaces.begin(),propspaces.end(),assembleMutexes);
		for_each(propspaces.begin(),propspaces.end(),recordRulesInActions);
		attrspaces = for_each(attrspaces.begin(),attrspaces.end(),doExamine(this));

		OUTPUT1 {
			cout << "Spaces now look like this:\n";
			for_each(propspaces.begin(),propspaces.end(),
						ptrwriter<PropertySpace>(cout,"\n"));
		};
	};
	virtual void visit_const_symbol(VAL::const_symbol * p)
	{
		TIMobjectSymbol * t = dynamic_cast<TIMobjectSymbol*>(p);
		t->distributeStates(this);
	};
	void checkSV()
	{
		for_each(propspaces.begin(),propspaces.end(),CheckSV(singleValued));
	};

	set<PropertySpace *> relevant(VAL::pddl_type * tp);
	void close(set<Property*> & seed,const VAL::pddl_type * pt);

	typedef vector<PropertySpace *>::const_iterator const_iterator;
	const_iterator pbegin() const {return propspaces.begin();};
	const_iterator pend() const {return propspaces.end();};
	const_iterator abegin() const {return attrspaces.begin();};
	const_iterator aend() const {return attrspaces.end();};
	const_iterator sbegin() const {return staticspaces.begin();};
	const_iterator send() const {return staticspaces.end();};	
};

class mutex;

struct mRec {
	Property * first;
	int second;
	opType opt;

	mRec(Property * x,int y,opType z) : first(x), second(y), opt(z) {};

	bool operator<(const mRec & m) const
	{
		return (first < m.first || (first == m.first &&
					second < m.second) || opt < m.opt);
	};
};

struct pairWith {
	int v;
	opType oo;
	pairWith(int x,opType o) : v(x), oo(o) {};

	mRec operator() (Property * o)
	{
		return mRec(o,v,oo);
	};
};

class MutexStore {
private:
	typedef map<VAL::operator_ *,mutex *> MutexRecord;
	MutexRecord mutexes;

// These are the enablers
	set<mRec> enablers;
	set<mRec> conditions;
// These are the invariants
	set<mRec> invariants;

public:
	virtual ~MutexStore() {};
	  
	mutex * getMutex(VAL::operator_ * o);
	void showMutexes();

	template<class TI>
	void add(TI s,TI e,int v,opType o) 
	{
		transform(s,e,inserter(enablers,enablers.begin()),pairWith(v,o));
	};
	void addCondition(Property * p,int v,opType o)
	{
		conditions.insert(mRec(p,v,o));
	};
	void addInvariant(Property * p,int v)
	{
		invariants.insert(mRec(p,v,MIDDLE));
	};
	void additionalMutexes();
};

#define MEX(x) dynamic_cast<TIM::MutexStore *>(x)

class TIMaction : public VAL::action, public MutexStore {
public:
	TIMaction(VAL::operator_symbol* nm,
	    VAL::var_symbol_list* ps,
	    VAL::goal* pre,
	    VAL::effect_lists* effs,
	    VAL::var_symbol_table* st) : action(nm,ps,pre,effs,st) 
	  {};
};

#define TAc(x) static_cast<TIM::TIMaction*>(x)

class TIMdurativeAction : public VAL::durative_action, public MutexStore {
public:
	bool isFixedDuration() const
	{
		return static_cast<TIMactionSymbol*>(name)->isFixedDuration();
	};
};

#define TDA(x) dynamic_cast<TIM::TIMdurativeAction*>(x)

void showMutex(VAL::operator_ * op);
void completeMutexes(VAL::operator_ * op);

/* MUTEX RELATIONSHIPS:
 * 
 * 	As, Am, Ae, Bs, Bm, Be:
 * 	
 * 		AsxBs, AsxBm, AsxBe    1  2  3
 * 		AmxBs, AmxBm, AmxBe    4  5  6
 * 		AexBs, AexBm, AexBe    7  8  9
 *
 */

enum MutexTypes { NONE = 0,START_START = 1, START_MID = 2, START_END = 4,
                  MID_START = 8, MID_MID = 16, MID_END = 32,
                  END_START = 64, END_MID = 128, END_END = 256};



struct mutRec {
	int first;
	int second;

	opType one;
	opType other;
	
	mutRec(int a,int b,opType x = INSTANT,opType y = INSTANT) : 
		first(a), second(b), one(x), other(y) {};

	bool operator==(const mutRec & m) const
	{
		return first == m.first && second == m.second && one == m.one && other == m.other;
	};

	bool operator!=(const mutRec & m) const
	{
		return !(*this == m);
	};

	bool operator<(const mutRec & m) const
	{
		return (first < m.first || 
					(first == m.first && (second < m.second ||
						(second == m.second && (one < m.one ||
						    (one == m.one && other < m.other))))));
	};
		
};

template<class TI>
TI getIx(TI s,int x)
{
	while(x > 0)
	{
		++s;
		--x;
	};
	return s;
};

class mutex {
private:
	VAL::operator_ * op1;
	VAL::operator_ * op2;
	set<mutRec> argPairs;

public:
	mutex(VAL::operator_ * o1,VAL::operator_ * o2) :
		op1(o1), op2(o2)
	{};
	
	static void constructMutex(VAL::operator_ * o1,int a1,VAL::operator_ * o2,int a2,
					opType t1 = INSTANT,opType t2 = INSTANT)
	{
		OUTPUT cout << "Adding a mutex between " << o1->name->getName() << ":" << a1 << " and " 
			<< o2->name->getName() << ":" << a2 << "\n";
		mutex * m = MEX(o1)->getMutex(o2);
// This condition ensures that we don't store the same mutex pair in both orders
// when they are arguments for the same operator. This is good if we are going to
// use the pairs for checking, but it could be less good if we want to check whether
// a specific pair of op-args are mutex.
		if(o1==o2) 
		{
			int a = min(a1,a2);
			if(a2 == a)
			{
				opType t3 = t1;
				t1 = t2;
				t2 = t3;
			};
			a2 = max(a1,a2);
			a1 = a;
		}
		else if(m->op2==o1)
		{
			int a = a1;
			a1 = a2;
			a2 = a;
			opType t = t1;
			t1 = t2;
			t2 = t;
		};
		if(m->argPairs.find(mutRec(a1,a2,t1,t2)) == m->argPairs.end())
		{
//		cout << "Inserting " << a1 << ":" << a2 << "\n";
			m->argPairs.insert(mutRec(a1,a2,t1,t2));
		};
	};

	void write(ostream & o) const;

	template<class TI>
	unsigned int getMutexes(VAL::operator_* A,TI sa,TI ea,TI sb,TI eb)
	{
		unsigned int ms = NONE;
		if(A==op2)
		{
			TI x = sa;
			sa = sb;
			sb = x;
			x = ea;
			ea = eb;
			eb = x;
		};

		for(set<mutRec>::const_iterator i = argPairs.begin();i != argPairs.end();++i)
		{
			if(*(getIx(sa,i->first))==*(getIx(sb,i->second)))
			{
//			cout << "Mutex for " << i->first << " " << **(getIx(sa,i->first)) << " and " << i->second
//					<< " " << **(getIx(sb,i->second)) << " " << i->one << " " << i->other << " " << 3*(i->one?i->one:1)+(i->other?i->other:1)
//					<< "\n";
				ms |= 1 << 3*(i->one?i->one-1:0)+(i->other?i->other-1:0);
			};
		};
		return ms;
	};
	template<class TI>
	bool selfMutex(TI sa,TI ea)
	{
		for(set<mutRec>::const_iterator i = argPairs.begin();i != argPairs.end();++i)
		{
			if(i->first != i->second && 
					*(getIx(sa,i->first))==*(getIx(sa,i->second)))
			{
				return true;
			};
		};
		return false;
	};
};

inline ostream & operator<<(ostream & o,const mutex & m)
{
	m.write(o);
	return o;
};

template<class TI>
unsigned int getMutexes(VAL::operator_ * A,TI sa,TI ea,VAL::operator_ * B,TI sb,TI eb)
{
	return MEX(A)->getMutex(B)->getMutexes(A,sa,ea,sb,eb);
};

template<class TI>
bool isMutex(const VAL::pred_symbol * pa,TI sa,TI ea,const VAL::pred_symbol * pb,TI sb,TI eb)
{
	return cTPS(pa)->checkMutex(sa,ea,cTPS(pb),sb,eb);
};

template<class TI>
bool selfMutex(const VAL::operator_ * op,TI sa,TI ea)
{
	TIM::MutexStore * tm = MEX(const_cast<VAL::operator_ *>(op));
    if(tm)
    {
            return tm->getMutex(const_cast<VAL::operator_*>(op))->selfMutex(sa,ea);
    }
    else
    {
            return false;
    };
};

class TIMfactory : public VAL::StructureFactory {
public:
	virtual VAL::action * buildAction(VAL::operator_symbol* nm,
	    VAL::var_symbol_list* ps,
	    VAL::goal* pre,
	    VAL::effect_lists* effs,
	    VAL::var_symbol_table* st) {return new TIMaction(nm,ps,pre,effs,st);};
	virtual VAL::durative_action * buildDurativeAction()
	{
		return new TIMdurativeAction();
	};
};

};

#endif
