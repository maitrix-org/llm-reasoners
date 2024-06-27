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

#include "FastEnvironment.h"
#include "TimSupport.h"
#include "ptree.h"
#include "Partitions.h"
#include <numeric>
#include <set>

using std::greater;
using std::max_element;
using std::min_element;
using std::set;
using std::copy;
using std::inserter;

#include <assert.h>

namespace TIM {

using namespace VAL;


ostream & operator<<(ostream & o, const Property & p)
{
	p.write(o);
	return o;
};

int getId(parameter_symbol* s)
{
	const IDsymbol<var_symbol> * i = dynamic_cast<const IDsymbol<var_symbol>*>(s);
	if(!i) return -1;
	return i->getId();
};

template<class T>
TIMpredSymbol * findPred(T * g);

template<>
TIMpredSymbol * findPred(simple_effect * g)
{
	return static_cast<TIMpredSymbol*>(const_cast<pred_symbol *>(g->prop->head));
};

template<>
TIMpredSymbol * findPred(derivation_rule * g)
{
	return static_cast<TIMpredSymbol*>(const_cast<pred_symbol *>(g->get_head()->head));
};

bool PropertySpace::contains(TIMobjectSymbol * t) const
{
	return binary_search(objects.begin(),objects.end(),t);
};

struct recordObjectIn {
	PropertySpace * ps;
	bool added;
	recordObjectIn(PropertySpace * p) : ps(p), added(false) {};

	void operator()(TIMobjectSymbol * o)
	{
		if(!ps->contains(o))
		{
			o->addIn(ps);
			ps->add(o);
			added = true;
		};
	};

	operator bool() {return added;};
};

var_symbol * getAt(var_symbol_list * ps,int v)
{
	var_symbol_list::iterator i = ps->begin();
	for(;v > 0;--v,++i);
	return *i;
};

parameter_symbol * getAt(parameter_symbol_list * ps,int v)
{
	parameter_symbol_list::iterator i = ps->begin();
	for(;v > 0;--v,++i);
	return *i;
};

	
TransitionRule::TransitionRule(TIMAnalyser * t,operator_ * o,int v,
					PropertyState * e,PropertyState * l,PropertyState * r,
					opType ty) : 
	tan(t), op(o), drv(0), opt(ty), var(v), enablers(e), lhs(l), rhs(r),
	objects(var>=0?tan->getTC().range(getAt(op->parameters,var)):vector<const_symbol*>())
{};

TransitionRule::TransitionRule(TIMAnalyser * t,derivation_rule * o,int v,
					PropertyState * e,PropertyState * l,PropertyState * r,
					opType ty) : 
	tan(t), op(0), drv(o), opt(ty), var(v), enablers(e), lhs(l), rhs(r),
	objects(var>=0?tan->getTC().range(getAt(drv->get_head()->args,var)):vector<const_symbol*>())
{};


bool TransitionRule::applicableIn(const PropertyState * ps) const
{
	return std::includes(ps->begin(),ps->end(),lhs->begin(),lhs->end());
};

struct checkNotApplicable {
	TIMobjectSymbol * tos;

	checkNotApplicable(const_symbol * c) :
		 tos(static_cast<TIMobjectSymbol*>(c))
	{};

	bool operator()(Property * p)
	{
		return find_if(p->begin(),p->end(),
				not1(bind2nd(mem_fun(&PropertySpace::contains),tos)))
						!= p->end();
	};
};

class RuleObjectIterator {
private:
	TransitionRule * trule;
	vector<const_symbol *>::iterator obit;

	void findValid()
	{
		while(obit != trule->objects.end())
		{
			if(find_if(trule->enablers->begin(),trule->enablers->end(),
						checkNotApplicable(*obit)) == trule->enablers->end()) break;
			++obit;
		};
	};

public:
	RuleObjectIterator(TransitionRule * tr) : 
		trule(tr), obit(trule->objects.begin())
	{
		findValid();
	};

	void toEnd() 
	{
		obit = trule->objects.end();
	};

	RuleObjectIterator & operator++()
	{
		++obit;
		findValid();
		return *this;
	};

	TIMobjectSymbol * operator*() 
	{
		return static_cast<TIMobjectSymbol*>(*obit);
	};

	bool operator==(const RuleObjectIterator & roi) const
	{
		return trule == roi.trule && obit == roi.obit;
	};

	bool operator!=(const RuleObjectIterator & roi) const
	{
		return trule != roi.trule || obit != roi.obit;
	};

};

RuleObjectIterator TransitionRule::endEnabledObjects()
{
	RuleObjectIterator i(this);
	i.toEnd();
	return i;
};

RuleObjectIterator TransitionRule::beginEnabledObjects()
{
	return RuleObjectIterator(this);
};


struct extendWithIncrRule {
	PropertySpace * ps;
	bool extended;
	extendWithIncrRule(PropertySpace * p) : ps(p), extended(false) {};

	void operator()(TransitionRule * tr)
	{
		if(tr->isIncreasing())
		{
			extended = for_each(tr->beginEnabledObjects(),tr->endEnabledObjects(),
							recordObjectIn(ps));
		};
	};

	operator bool() {return extended;};
};


template<class T>
T isSuper(const PropertyState * p,T b,T e)
{
	while(b != e)
	{
		if(std::includes(p->begin(),p->end(),(*b)->begin(),(*b)->end())) return b;
		++b;
	};
	return b;
};

struct extendWithStateRule {
	set<PropertyState *> & got;
	list<PropertyState *> & toExtend;
	PropertyState * prop;
	
	extendWithStateRule(set<PropertyState*> & s,list<PropertyState*> & l) : 
		got(s), toExtend(l), prop(toExtend.empty()?0:*(toExtend.begin())) {};

	void operator()(TransitionRule * tr) 
	{
		if(!prop) return;
		PropertyState * p = tr->tryRule(prop);
		if(p && got.find(p) == got.end())
		{
			set<PropertyState *>::const_iterator i = isSuper(p,got.begin(),got.end());
			if(i != got.end())
			{
				OUTPUT cout << *p << " is a superset of a state we already have: " << 
					**i << "\n";
			}
			else
			{
				got.insert(p);
				toExtend.push_back(p);
			};
		};
	};

	void next() 
	{
		toExtend.pop_front();
		prop = toExtend.empty()?0:*(toExtend.begin());
	};

	operator bool() {return !toExtend.empty();};

};

bool PropertySpace::extend()
{
	if(!isStateValued)
	{
//		cout << "\nExtending attribute space...\n";
//		write(cout);
// We add to this space every object that is currently in all of the spaces for
// enabling properties of any increasing rule in this space.
		bool b = for_each(rules.begin(),rules.end(),extendWithIncrRule(this));
		if(b) sort(objects.begin(),objects.end());
		return b;
	}
	else
	{
//		cout << "\nExtending state space...\n";
// For each state in the space we need to apply each rule (if it can be applied) and 
// add the corresponding new state to the space. We really should confirm that we only
// use states for enabled objects, but maybe that's a refinement we can save until 
// later.
// 
// Basic pattern: for each state and rule pair, if rule applies then produce new state.
// Key issue: avoid retesting states with rules. One way to do this would be to take
// each state in turn and extend it with all rules, adding new states to a list as they
// are generated. As they are added to the list they can also be added to the internal 
// set of states, so that membership can be checked fast.
		list<PropertyState *> toExtend;
		copy(states.begin(),states.end(),inserter(toExtend,toExtend.begin()));
		extendWithStateRule ewsr(this->states,toExtend);
		while(for_each(rules.begin(),rules.end(),ewsr))
			ewsr.next();
//		write(cout);
		return false;
	};
};

template<>
TIMpredSymbol * findPred(simple_goal * g)
{
	return static_cast<TIMpredSymbol*>(const_cast<pred_symbol *>(g->getProp()->head));
};

struct process_argument {
	TIMAnalyser * ta;
	TIMpredSymbol * timps;
	int arg;
	proposition * prp;

	virtual ~process_argument() {};
	
	template<class T>
	process_argument(TIMAnalyser * t,T * g,proposition * p) :
		ta(t), timps(findPred<T>(g)), arg(0), prp(p) {};

	template<class T>
	process_argument(TIMAnalyser * t,T * g) :
		ta(t), timps(findPred<T>(g)), arg(0), prp(0) {};

	virtual void operator()(parameter_symbol * p) = 0;
};

struct process_argument_in_goal : public process_argument {
	process_argument_in_goal(TIMAnalyser * t,simple_goal * g) : 
		process_argument(t,g)
	{};

	virtual ~process_argument_in_goal() {};
	
	virtual void operator()(parameter_symbol * p)
	{
		Property * prop = timps->property(arg);
		++arg;
		ta->insertPre(getId(p),prop);
	};
};

struct process_argument_in_effect : public process_argument {
	process_argument_in_effect(TIMAnalyser * t,simple_effect * g) : 
		process_argument(t,g)
	{};

	virtual ~process_argument_in_effect() {};
	
	virtual void operator()(parameter_symbol * p)
	{
		Property * prop = timps->property(arg);
		++arg;
		ta->insertEff(getId(p),prop);
	};
};

struct process_argument_in_derivation_effect : public process_argument {
	process_argument_in_derivation_effect(TIMAnalyser * t,derivation_rule * g) : 
		process_argument(t,g)
	{};

	virtual ~process_argument_in_derivation_effect() {};
	
	virtual void operator()(parameter_symbol * p)
	{
		Property * prop = timps->property(arg);
		++arg;
		ta->insertEff(getId(p),prop);
	};
};


struct process_constant_in_goal : public process_argument {
	process_constant_in_goal(TIMAnalyser * t,simple_goal * g) : 
		process_argument(t,g)
	{};

	virtual ~process_constant_in_goal() {};
	
	virtual void operator()(parameter_symbol * p)
	{
		Property * prop = timps->property(arg);
		++arg;
		ta->insertGoal(p,prop);
	};
};

struct process_constant_in_initial : public process_argument {
	process_constant_in_initial(TIMAnalyser * t,simple_effect * g) : 
		process_argument(t,g,g->prop)
	{};

	virtual ~process_constant_in_initial() {};
	
	virtual void operator()(parameter_symbol * p)
	{
		Property * prop = timps->property(arg);
		++arg;
		ta->insertInitial(p,prop,prp);
	};
};

void TIMAnalyser::visit_simple_goal(simple_goal * p) 
{
	if(finally)
	{
		for_each(p->getProp()->args->begin(),p->getProp()->args->end(),
				process_constant_in_goal(this,p));
	}
	else
	{
		for_each(p->getProp()->args->begin(),p->getProp()->args->end(),
				process_argument_in_goal(this,p));
	};
};

struct notIn {
	PropertySpace * ps;

	notIn(PropertySpace * p) : ps(p) {};

	bool operator()(Property * p)
	{
		return (p->begin()==p->end()) || *(p->begin()) != ps;
	};
};

bool Property::matches(const extended_pred_symbol * eps,pddl_type * pt)
{
	if(EPS(predicate)->getParent() != eps->getParent()) return false;
	//cout << "A: " << *pt << "\n";
	//cout << "B: " << *eps << "\n";
	Types::const_iterator tcItr = eps->tcBegin()+posn;
        if (tcItr == eps->tcEnd()) {
		std::cerr << "A problem has been encountered with your domain/problem file.\n";
		std::cerr << "-------------------------------------------------------------\n";
		std::cerr << "Unfortunately, a bug has been encountered in your domain and problem file,\n";
		std::cerr << "and the planner has to terminate.  The predicate:\n\n";
		std::cerr << "\t" << eps->getName() << "\n\n";
		int realArgs = 0;
		{
			Types::const_iterator tcsItr = eps->tcBegin();
			Types::const_iterator tcsEnd = eps->tcEnd();
			for (; tcsItr != tcsEnd; ++tcsItr) ++realArgs;
		}
		std::cerr << "...takes " << realArgs << " argument";
		if (realArgs != 1) std::cerr << "s";
		std::cerr << ", but has been given at least " << posn + 1 << ".\n";
		exit(0);

        }
	if(*tcItr)
	{
	//	cout << "C: " << **(eps->tcBegin()+posn) << "\n";
	}
	else 
	{
	//	cout << "C: ***\n";
		return false;
	};
	return (pt == (*tcItr)->type);
};

bool Property::equivalent(const Property * p) const
{
	if(this==p) return true;
	if(posn != p->posn || 
			EPS(predicate)->getParent() != EPS(p->predicate)->getParent())
			return false;
	return true;
};

struct notMatchFor {
	pddl_type * pt;
	Property * prop;

	notMatchFor(pddl_type * p,Property * pr) : pt(p), prop(pr) 
	{};
	bool operator()(extended_pred_symbol * eps)
	{
		return !(prop->matches(eps,pt));
	};
};

struct toProp {
	int arg;

	toProp(int a) : arg(a) {};

	Property * operator()(extended_pred_symbol * eps)
	{
		return static_cast<TIMpredSymbol*>(eps)->property(arg);
	};
};

void TIMAnalyser::close(set<Property*> & seed,const pddl_type * pt)
{
	bool adding = true;
	while(adding)
	{
		adding = false;
		PropertyState * ps = PropertyState::getPS(this,pt,seed.begin(),seed.end());
		for(TRules::const_iterator r = trules.begin();r != trules.end();++r)
		{
			if((*r)->applicableIn(ps))
			{
				copy((*r)->getRHS()->begin(),(*r)->getRHS()->end(),
								inserter(seed,seed.begin()));
				adding |= (seed.size() > ps->size());
			};
		};
	};
};

vector<Property *> Property::matchers()
{
	vector<extended_pred_symbol *> v;
	holding_pred_symbol * h = EPS(predicate)->getParent();

	std::remove_copy_if(h->pBegin(),h->pEnd(),inserter(v,v.begin()),
				notMatchFor((*((EPS(predicate)->tBegin())+posn))->type,this));
	vector<Property *> ps;
	transform(v.begin(),v.end(),inserter(ps,ps.begin()),toProp(posn));
	return ps;
};

struct setMatchers {
	vector<Property *> & ps;

	setMatchers(vector<Property *> & p) : ps(p) {};

	void operator()(Property * p)
	{
		vector<Property *> props = p->matchers();
		copy(props.begin(),props.end(),inserter(ps,ps.end()));
	};

};

void TIMobjectSymbol::distributeStates(TIMAnalyser * tan)
{
	vector<Property *>::iterator s,e,m;
	vector<Property *> matchers;
	for_each(initial.begin(),initial.end(),setMatchers(matchers));
	s = matchers.begin();
	e = matchers.end();
	
	while(s != e)
	{
		if((*s)->begin() == (*s)->end()) 
		{
			// State belonging to no property space.
			// This should not happen...but, see notes.
			++s;
			continue;
		};
		PropertySpace * p = *((*s)->begin()); // Assumes only one space at this stage
		p->add(this);
		m = std::partition(s,e,notIn(p));
		std::sort(m,e);
		PropertyState * ps = PropertyState::getPS(tan,type,m,e);
		p->add(ps);
		e = m;
	};
};

Property * Property::getBaseProperty(const pddl_type * pt) const
{
	if(!EPS(predicate)->getParent()) return const_cast<Property*>(this);
	TIMpredSymbol * tps = 
		static_cast<TIMpredSymbol*>(EPS(predicate)->getParent()->
			find(predicate,makeTT(EPS(predicate)->tBegin(),posn,pt),
										makeTT(EPS(predicate)->tEnd(),0,0)));
	return tps->property(posn);
};

ostream & operator <<(ostream & o,const TIMobjectSymbol & t)
{
	t.write(o);
	return o;
};

void TIMAnalyser::visit_simple_effect(simple_effect * p) 
{
	if(initially)
	{
		for_each(p->prop->args->begin(),p->prop->args->end(),
				process_constant_in_initial(this,p));
	}
	else
	{
		for_each(p->prop->args->begin(),p->prop->args->end(),
					process_argument_in_effect(this,p));
	};
};

void TIMAnalyser::visit_simple_derivation_effect(derivation_rule * p) 
{
	for_each(p->get_head()->args->begin(),p->get_head()->args->end(),
				process_argument_in_derivation_effect(this,p));
};

void TIMAnalyser::insertPre(int v,Property * p)
{
	if(v<0) 
	{
		OUTPUT cout << "Property for a constant\n";
		return;
	};
	if(overall) 
	{
		MEX(op)->addInvariant(p,v);
		return;
	}
	else
	{
		if (op) MEX(op)->addCondition(p,v,isDurative?(atStart?START:END):INSTANT);
	};
	if(!rules[v]) {
		if (op) rules[v] = new ProtoRule(this,op,v,isDurative?(atStart?START:END):INSTANT);
		if (drv) rules[v] = new ProtoRule(this,drv,v,isDurative?(atStart?START:END):INSTANT);
	}
	rules[v]->insertPre(p);
};

void TIMAnalyser::insertEff(int v,Property * p)
{
	if(v<0) 
	{
		OUTPUT cout << "Property for a constant\n";
		return;
	};
	if(!rules[v]) {
		if (op) rules[v] = new ProtoRule(this,op,v,isDurative?(atStart?START:END):INSTANT);
		if (drv) rules[v] = new ProtoRule(this,drv,v,isDurative?(atStart?START:END):INSTANT);
	}
	if(adding)
	{
		rules[v]->insertAdd(p);
	}
	else
	{
		rules[v]->insertDel(p);
	};
};

void TIMAnalyser::insertGoal(parameter_symbol * c,Property * p)
{
	TIMobjectSymbol * cc = dynamic_cast<TIMobjectSymbol *>(c);
	
	cc->addFinal(p);
};

void TIMAnalyser::insertInitial(parameter_symbol * c,Property * p,proposition * prp)
{
	TIMobjectSymbol * cc = dynamic_cast<TIMobjectSymbol *>(c);

	cc->addInitial(p,prp);
};


ostream & operator<<(ostream & o,const PropertyState & p)
{
	p.write(o);
	return o;
};

ostream & operator<<(ostream & o,const TransitionRule & tr)
{
	tr.write(o);
	return o;
};

void ProtoRule::addRules(TRules & trules)
{
	sort(enablers.begin(),enablers.end());
	sort(adds.begin(),adds.end());
	sort(dels.begin(),dels.end());
	vector<Property *> es;
	set_difference(enablers.begin(),enablers.end(),dels.begin(),dels.end(),
						inserter(es,es.begin()));
	vector<Property *> is;
	set_intersection(dels.begin(),dels.end(),adds.begin(),adds.end(),
						inserter(is,is.begin()));
// Problem here: if we have a constant we still need to know what it is...
	parameter_symbol * v = (var>=0?(op ? getAt(op->parameters,var) : getAt(drv->get_head()->args,var)):0);
	vector<const pddl_type *> types(tan->getTC().leaves(v->type));
	if(types.empty()) types.push_back(v->type);
	if(is.size()>1 || ((is.size()==1) && (dels.size()>1 || adds.size()>1))) 
	{
		assert(op);
		//don't worry, cannot ever be true for derivation rules: have no delete effects, so is empty, dels empty....
		for(vector<Property*>::iterator i = is.begin();i != is.end();++i)
		{
			Property * p = *i;
			if(find(enablers.begin(),enablers.end(),p)!=enablers.end())
			{
				enablers.erase(find(enablers.begin(),enablers.end(),p));
			};
			// Else we are deleting a non-precondition - which could be OK if we
			// are looking at an action end point.
			for(vector<const pddl_type *>::const_iterator pt = types.begin();
							pt != types.end();++pt)
			{
				PropertyState * x = PropertyState::getPS(tan,*pt,i,i+1);
				PropertyState * en = PropertyState::getPS(tan,*pt,enablers.begin(),enablers.end());
		
				trules.push_back(new TransitionRule(tan,op,var,en,x,x,opt));
			};
			enablers.insert(find_if(enablers.begin(),enablers.end(),
										bind2nd(greater<Property*>(),p)),p);
		};
			
		if(adds.size() > is.size() || dels.size() > is.size())
		{
			vector<Property *> nadds, ndels;
			set_difference(adds.begin(),adds.end(),is.begin(),is.end(),
						inserter(nadds,nadds.begin()));
			set_difference(dels.begin(),dels.end(),is.begin(),is.end(),
						inserter(ndels,ndels.begin()));
			enablers.clear();
			merge(es.begin(),es.end(),is.begin(),is.end(),
						inserter(enablers,enablers.begin()));
			es.swap(enablers);
		}
		else
		{
// No other rule to construct (all in split forms).
			return;
		};
	};
	for(vector<const pddl_type *>::const_iterator pt = types.begin();
					pt != types.end();++pt)
	{
		PropertyState * e = PropertyState::getPS(tan,*pt,es.begin(),es.end());
		PropertyState * l = PropertyState::getPS(tan,*pt,dels.begin(),dels.end());
		PropertyState * r = PropertyState::getPS(tan,*pt,adds.begin(),adds.end());
		if (op) {
			trules.push_back(new TransitionRule(tan,op,var,e,l,r,opt));
		} else {
			trules.push_back(new TransitionRule(tan,drv,var,e,l,r,opt));
		}
	};
};

PropertyState::PMap PropertyState::pmap;

PropertySpace * PSCombiner(PropertySpace * p1,PropertySpace * p2)
{
	p1->merge(p2);
	return p1;
};

typedef PropertySpace *(*PSC)(PropertySpace *,PropertySpace *);

typedef Partitioner<Property*,PropertySpace *,PSC> PM;

struct unifyWith {
	PM & pm;
	Property * pr;
	TransitionRule * trl;
	
	unifyWith(PM & p,Property * pp,TransitionRule * tr) : pm(p), pr(pp), trl(tr) {};

	void operator()(Property * p)
	{
		if(!pm.contains(p))
		{
			pm.add(p,new PropertySpace(p,trl));
		}
		else
		{
			//cout << "Adding rule " << *trl << " to " << *(pm.getData(p)) << "\n";
			pm.getData(p)->add(trl);
		};
		pm.connect(pr,p);
	};
};

struct makeSpace {
	PM & pm;

	makeSpace(PM & p) : pm(p) {};

	void operator()(Property * p) 
	{
		if(!pm.contains(p))
		{
			pm.add(p,new PropertySpace(p));
		};
	};
};

struct rulePartitioner {
	PM & pm;
	rulePartitioner(PM & p) : pm(p) {};

	void operator()(TransitionRule * tr)
	{
		if(tr->isTrivial())
		{
			tr->distributeEnablers();
			return;
		};
		Property * p = tr->isIncreasing()?*(tr->rhs->begin()):*(tr->lhs->begin());
		
		for_each(tr->lhs->begin(),tr->lhs->end(),unifyWith(pm,p,tr));
		for_each(tr->rhs->begin(),tr->rhs->end(),unifyWith(pm,p,tr));
		for_each(tr->enablers->begin(),tr->enablers->end(),makeSpace(pm));
	};
};

void TransitionRule::distributeEnablers()
{
	if (op) {
		MutexStore *m = MEX(op);
// The enabler mRecs in an op record the enabling property, the variable affected
// by the rule that is enabled and the part of the operator the rule comes from:
// START, MIDDLE or END
		m->add(enablers->begin(),enablers->end(),var,opt);
	}
};

PropertySpace::PropertySpace(Property * p,TransitionRule * t) :
	properties(1,p), isStateValued(!t->isAttribute()), LSchecked(false)
{
	rules.insert(t);
};

void PropertySpace::add(TransitionRule * t)
{
	rules.insert(t);
	isStateValued &= !t->isAttribute();
};

void PropertySpace::write(ostream & o) const
{
	o << "\nState space states:\n";
	for_each(states.begin(),states.end(),ptrwriter<PropertyState>(o,"\n"));
	o << "\nSpace properties: ";
	for_each(properties.begin(),properties.end(),ptrwriter<Property>(o," "));
	o << "\nSpace objects: ";
	for_each(objects.begin(),objects.end(),ptrwriter<const_symbol>(o," "));
	o << "\nSpace rules:\n";
	for_each(rules.begin(),rules.end(),ptrwriter<TransitionRule>(o,"\n"));
	o << "Space is: " << (isStateValued?"state valued":"attribute valued");
};

ostream & operator<<(ostream & o,const PropertySpace & p)
{
	p.write(o);
	return o;
};

PropertySpace * spaceSet(PM::DataSource & p)
{
	return PM::grabData(p)->finalise();
};

bool PropertySpace::isLockingSpace()
{
	if(LSchecked) return isLS;
	LSchecked = true;

	cout << "Complete check on locking spaces\n";
	return false;
};

void TIMAnalyser::setUpSpaces()
{
	PM pts(&PSCombiner);

	for_each(trules.begin(),trules.end(),rulePartitioner(pts));

	transform(pts.begin(),pts.end(),inserter(propspaces,propspaces.begin()),
				spaceSet);
//	for_each(propspaces.begin(),propspaces.end(),ptrwriter<PropertySpace>(cout,"\n"));
};

// Need this because assembleMutexes is not const.
struct aMxs {
	PropertySpace * p;

	aMxs(PropertySpace * ps) : p(ps) {};

	void operator()(TransitionRule * tr)
	{
		p->assembleMutexes(tr);
	};

	void operator()(Property * ps)
	{
		p->assembleMutexes(ps);
	};
};

struct aMxs1 {
	TransitionRule * tr;

	aMxs1(TransitionRule * t) : tr(t) {};

	void operator()(TransitionRule * t)
	{
		t->assembleMutex(tr);
	};
};

struct aMxs2 {
	PropertySpace * ps;
	Property * p;

	aMxs2(PropertySpace * tps,Property * tp) : ps(tps), p(tp) {};

	void operator()(Property * tp)
	{
		if(p <= tp) ps->assembleMutexes(p,tp);
	};
};

/*
struct aMxs3 {
	TransitionRule * tr;

	aMxs3(TransitionRule * t) : tr(t) {};

	void operator()(Property * p) 
	{
		for(Property::SpaceIt i = p->begin();i != p->end();++i)
		{
			if((*i)->isState())
			{
				cout << *p << " is candidate enabler for action " << tr->byWhat()->name->getName() << "\n";
				(*i)->assembleMutexes(tr,p);
			};
		};	
	};
};


// tr is enabled by property p which appears in this PropertySpace
void PropertySpace::assembleMutexes(TransitionRule * tr,Property * p)
{
	for_each(rules.begin(),rules.end(),aMxs1(tr));
};
*/


// This machinery might need to be reviewed to ensure it handles subtypes
// properly.
void PropertySpace::assembleMutexes(Property * p1,Property * p2)
{
	if(p1==p2)
	{
		if(p1->root()->arity() == 1) return;
// Same property - special case
		SIterator s = begin();
		for(;s != end();++s)
		{
			if((*s)->count(p1)>1)
			{
				break;
			};
		};
		if(s==end())
		{
// Got one...
			OUTPUT cout << "Mutex between " << *p1 << " and " << *p2 << "\n";
			cTPS(p1->root())->setMutex(p1->aPosn(),cTPS(p2->root()),p2->aPosn());
		};
	}
	else
	{
// Different properties
// If they only ever appear in different states then they are mutex...
		SIterator s = begin();
		for(;s != end();++s)
		{
			if((*s)->contains(p1) && (*s)->contains(p2))
			{
				break;
			};
		};
		if(s==end())
		{
// Got a mutex...
			OUTPUT cout << "Mutex between " << *p1 << " and " << *p2 << "\n";
			cTPS(p1->root())->setMutex(p1->aPosn(),cTPS(p2->root()),p2->aPosn());
			cTPS(p2->root())->setMutex(p2->aPosn(),cTPS(p1->root()),p1->aPosn());
		};
	};
};

// This handles the mutexes that arise because an object cannot simultaneously
// make two different state transitions.
void TransitionRule::assembleMutex(TransitionRule * tr)
{
	if (op) {
		OUTPUT cout << "Mutex caused by rules: " << *this 
				<< " (" << this->byWhat()->name->getName() << ") and " << *tr 
				<< " (" << tr->byWhat()->name->getName() << ")\n";

		mutex::constructMutex(op,var,tr->op,tr->var,opt,tr->opt);
		mutex::constructMutex(tr->op,tr->var,op,var,tr->opt,opt);
	}
};

ostream & operator<<(ostream & o,opType opt)
{
	switch(opt)
	{
		case INSTANT:
			break;
		case START:
			o << "[start]";
			break;
		case END:
			o << "[end]";
			break;
		case MIDDLE:
			o << "[middle]";
			break;
		default:
			break;
	};
	return o;
};

void mutex::write(ostream & o) const
{
	for(set<mutRec>::const_iterator i = argPairs.begin();i != argPairs.end();++i)
	{
		if(op1==op2)
		{
			o << "Cannot perform two concurrent '" << op1->name->getName() << "'s for same ";
			if(getAt(op1->parameters,i->first)->type)
			{
				o << getAt(op1->parameters,i->first)->type->getName();
			}
			else
			{
				o << i->first << "th argument";
			};					
			o << " " << i->one << "-" << i->other << "\n";
		}
		else
		{

			o << "Mutex for '" << op1->name->getName() << "' and '" << 
						op2->name->getName() << " args: " << i->first << " " << i->second << "\n";
				o << "Mutex for '" << op1->name->getName() << "' and '" << 
						op2->name->getName()
					<< "' when using same ";
			if(getAt(op1->parameters,i->first)->type)
			{
				o << getAt(op1->parameters,i->first)->type->getName();
			}
			else
			{
				o << i->first << "th argument";
			};
			o << " " << i->one << "-" << i->other << "\n";
		};
	};
};



mutex * MutexStore::getMutex(operator_ * o)
{
	MutexRecord::iterator m = mutexes.find(o);
	if(m == mutexes.end())
	{
		mutex * mx = new mutex(dynamic_cast<operator_*>(this),o);
		mutexes[o] = mx;
		MEX(o)->mutexes[dynamic_cast<operator_*>(this)] = mx;
		return mx;
	}
	else
	{
		return m->second;
	};
};

void MutexStore::showMutexes()
{
	operator_ * o = dynamic_cast<operator_ *>(this);
	for(MutexRecord::iterator i = mutexes.begin();i != mutexes.end();++i)
	{
		if(i->first >= o)
		{
			cout << *(i->second);
		};
	};
};

void showMutex(operator_ * op)
{
	 TIM::MutexStore * tm = MEX(op);
     if(tm)
     {
            tm->showMutexes();
     }
     else
     {
            cout << "Not an action\n";
     };
};

void completeMutexes(operator_ * op)
{
	TIM::MutexStore * tm = MEX(op);
    if(tm) tm->additionalMutexes();
};

void PropertySpace::assembleMutexes(Property * p)
{
	for_each(properties.begin(),properties.end(),aMxs2(this,p));
};


// This is going to consider every rule against every rule in a space.
void PropertySpace::assembleMutexes(TransitionRule * tr)
{
	for_each(rules.begin(),rules.end(),aMxs1(tr));
// This was introduced as a way to handle the additional mutexes, but it is
// unnecessary.
//	for_each(tr->getEnablers()->begin(),tr->getEnablers()->end(),
//					aMxs3(tr));
};


void PropertySpace::assembleMutexes()
{
	for_each(rules.begin(),rules.end(),aMxs(this));
	for_each(properties.begin(),properties.end(),aMxs(this));
};

void TIMAnalyser::assembleMutexes(PropertySpace * p)
{
	p->assembleMutexes();
};

void TIMAnalyser::recordRulesInActions(PropertySpace * p)
{
	p->recordRulesInActions();
};

void TransitionRule::recordInAction(PropertySpace * p)
{
	TAS(op->name)->addStateChanger(p,this);
};

bool TIMactionSymbol::hasRuleFor(int prm) const
{
	for(RCiterator i = begin();i != end();++i)
	{
		if((*i)->paramNum() == prm) return true;
	};
	return false;
};

struct recRiA
{
	PropertySpace * ps;
	recRiA(PropertySpace * p) : ps(p) {};
	void operator()(TransitionRule * tr)
	{
		tr->recordInAction(ps);
	};
};

void PropertySpace::recordRulesInActions()
{
	for_each(rules.begin(),rules.end(),recRiA(this));
};

struct addMx1 {
	operator_ * op;
	const mRec & pr;

	addMx1(operator_ * o,const mRec & p) : op(o), pr(p) {};

	void operator() (PropertySpace * ps)
	{
		//cout << "Property space: " << *ps << "\n";
		//if(pr.opt == MIDDLE && !ps->isState()) return;
		if(!ps->isState()) return;
		
		ps->assembleMutexes(op,pr);
	};

	void operator() (TransitionRule * tr)
	{
		tr->assembleMutex(op,pr);
	};
};

struct addMx {
	operator_ * mx;

	addMx(operator_ * m) : mx(m) {};

	void operator() (const mRec & p)
	{
		//cout << "For enabler " << *(p.first) << "\n";
// Work through all the PropertySpaces that p.first belongs to. Recall p.first
// is an enabler for the operator mx.
		vector<Property *> ps((p.first)->matchers());
		for(vector<Property*>::iterator i = ps.begin();i != ps.end();++i)
		{
			for_each((*i)->begin(),(*i)->end(),addMx1(mx,p));
		};
	};
};

void TransitionRule::assembleMutex(operator_ * o,const mRec & p)
{
	mutex::constructMutex(op,var,o,p.second,opt,p.opt);
	mutex::constructMutex(o,p.second,op,var,p.opt,opt);
};

void PropertySpace::assembleMutexes(operator_ * op,const mRec & p)
{
// Work through all the rules in the space
	for_each(rules.begin(),rules.end(),addMx1(op,p));
};

void MutexStore::additionalMutexes()
{
// Work through all the enabler mRecs of this action
	//cout << "Considering enablers for action: " << dynamic_cast<operator_*>(this)->name->getName() << "\n";
	for_each(conditions.begin(),conditions.end(),
				addMx(dynamic_cast<operator_*>(this)));
// Work through all the invariant mRecs of this action
	for_each(invariants.begin(),invariants.end(),
				addMx(dynamic_cast<operator_*>(this)));
};

bool checkRule(bool x,TransitionRule * tr)
{
	return x && !tr->isIncreasing();
};

Property * TransitionRule::candidateSplit()
{
	if(isIncreasing())
	{
		return *(rhs->begin());
	};
	if(isDecreasing())
	{
		return *(lhs->begin());
	};
	return 0;
};


void recordSV::operator()(Property * p)
{
	vector<int> cnts(ps->countsFor(p));
	int mx = cnts.empty()?0:*max_element(cnts.begin(),cnts.end());
	int mn = cnts.empty()?0:*min_element(cnts.begin(),cnts.end());
	
	p->setSV(mx==1,mn>0);
	if(mx==1)
	{
		sv.push_back(p);
#ifdef VERBOSE
		cout << *p << " is single valued ";
		if(mn>0) cout << "and required";
		cout << "\n";
#endif
	};
};

bool PropertySpace::examine(vector<PropertySpace *> & as)
{
// Leave this on to remind us to fix it!
//#ifdef VERBOSE
	if(std::accumulate(rules.begin(),rules.end(),true,
			checkRule))
	{
		OUTPUT cout << "\nPotential pseudo space...\n" << 
		"This will cause problems in several uses of TIM - tell Derek to get on with fixing it!\n" 
		<< *this << "\n";
	};
//#endif

	while(!isStateValued && properties.size() > 1)
	{
//		cout << "\nMultiple properties...looking for a splitter...\n";
//		cout << "Space is: " << *this << "\n";
		for(set<TransitionRule*>::iterator i = rules.begin(); i != rules.end();++i)
		{
			Property * p = (*i)->candidateSplit();
			if(p) 
			{
//				cout << "Splitter is " << *p << "\n";
				PropertySpace * ps = slice(p);
//				cout << "Split into: " << *ps << "\nand: " << *this << "\n";
				while(ps->extend());
				as.push_back(ps);
				break;
			};
		};
	};
	if(isStateValued)
	{
		if(!isStatic())
		{
			while(extend());
			assembleMutexes();
		};
		return true;
	};
	return false;
};

typedef set<PropertyState*> PStates;

void doExamine::operator()(PropertySpace * p) 
{
	if(p->examine(newas))
	{
		tan->propspaces.push_back(p);
	}
	else
	{
		newas.push_back(p);
	};
};

template<class TI>
struct getConditionally {
	bool cond;
	Property * prop;
	TI pit;
	TI terminus;
	
	getConditionally(bool c,Property * p,TI pt,TI term) : 
		cond(c), prop(p), pit(pt), terminus(term) 
	{
		while(pit != terminus && (c?(*pit == prop):(*pit != prop))) ++pit;
	};

	Property * operator*() 
	{
		return *pit;
	};

	getConditionally<TI> & operator++()
	{
		++pit;
		while(pit != terminus && (cond?(*pit == prop):(*pit != prop))) ++pit;
		return *this;
	};

	bool operator==(const getConditionally<TI> & x) const
	{
		return pit==x.pit;
	};

	bool operator!=(const getConditionally<TI> & x) const
	{
		return pit != x.pit;
	};

};

template<class TI>
getConditionally<TI> getIt(bool c,Property * p,TI x1,TI x2)
{
	return getConditionally<TI>(c,p,x1,x2);
};

pair<PropertyState *,PropertyState *> PropertyState::split(Property * p)
{
	PropertyState * p1 = retrieve(tan,getIt(false,p,begin(),end()),
									getIt(false,p,end(),end()));
	PropertyState * p2 = retrieve(tan,getIt(true,p,begin(),end()),
									getIt(true,p,end(),end()));
	return make_pair(p1,p2);
};

TransitionRule * TransitionRule::splitRule(Property * p)
{
	if(find(lhs->begin(),lhs->end(),p)==lhs->end() && 
			find(rhs->begin(),rhs->end(),p)==rhs->end())
	{
		return 0;
	};
//	cout << "This rule splits: " << *this << "\n";
	
	pair<PropertyState *,PropertyState *> lhss, rhss;
	PropertyState * ens;

// Need to remove all instances of p from the lhs/rhs of rule. Note that since 
// we split rules that had a property in both sides the only way p can appear 
// in both lhs and rhs is if one or other side contains nothing except p (possibly
// several copies). 
	lhss = lhs->split(p);
//	cout << "Left splits to " << *(lhss.first) << " and " << *(lhss.second) << "\n";
	ens = enablers;
	enablers = enablers->add(lhss.first->begin(),lhss.first->end());
	lhs = lhss.second;
	ens = ens->add(lhss.second->begin(),lhss.second->end());
	rhss = rhs->split(p);
	rhs = rhss.second;
//	cout << "Right splits to " << *(rhss.first) << " and " << *(rhss.second) << "\n";

//	cout << "Enablers are now: " << *ens << " and " << *enablers << "\n";
	TransitionRule * trnew = new TransitionRule(tan,op,var,ens,lhss.first,rhss.first,opt);
//	cout << "Objects are: ";
//	for_each(objects.begin(),objects.end(),ptrwriter<const_symbol>(cout," "));
//	cout << "\n";
	return trnew;
};

TransitionRule::TransitionRule(TransitionRule * tr,PropertyState * e,PropertyState * l,PropertyState * r) :
	tan(tr->tan), op(tr->op), opt(tr->opt), var(tr->var), enablers(e), lhs(l), rhs(r),
	objects() {};

struct ruleSplitOn {
	set<TransitionRule *> & xrules;
	set<TransitionRule *> rules;
	Property * prop;

	ruleSplitOn(Property * p,set<TransitionRule *> & rs) : xrules(rs), prop(p) {};

	void operator()(TransitionRule * tr) 
	{
		TransitionRule * t = tr->splitRule(prop);
		if(t) xrules.insert(t);
		if(!tr->isTrivial()) 
		{
			rules.insert(tr);
		};
	};

	operator set<TransitionRule *>() {
		return rules;
	}; 
};

void splitRules(set<TransitionRule *> & rules,Property * p,set<TransitionRule *> & xrules)
{
	rules = for_each(rules.begin(),rules.end(),ruleSplitOn(p,xrules));
};

void splitObjects(vector<TIMobjectSymbol *> & tobs,Property * p,vector<TIMobjectSymbol *> & xtobs)
{
// I think we should be checking whether the objects are really valid for both
// spaces - in particular if the new space is a state space we should be filtering it.
// 
//	cout << "SYMBOLS: ";
//	for_each(tobs.begin(),tobs.end(),ptrwriter<const_symbol>(cout," "));
//	cout << "\n";
};

void splitStates(PStates & states,Property * p,PStates & xstates)
{
	PStates::iterator e = states.end();
	PStates newStates;
	for(PStates::iterator i = states.begin();i != e;++i)
	{
		pair<PropertyState *,PropertyState *> pp = (*i)->split(p);
		if(!pp.first->empty()) xstates.insert(pp.first);
		if(!pp.second->empty()) newStates.insert(pp.second);
	};
	states.swap(newStates);
};

bool ruleCheck(bool t,TransitionRule * tr)
{
	return t && !tr->isAttribute();
};

void PropertySpace::checkStateValued()
{
	isStateValued = accumulate(rules.begin(),rules.end(),true,ruleCheck);
};

PropertySpace * PropertySpace::slice(Property * p)
{
// Also need to remove split property from this space and correct the relevant
// references to membership of property spaces.
	PropertySpace * ps = new PropertySpace(p);
	ps->isStateValued = false;
	splitStates(states,p,ps->states);
//cout << "About to split rules in " << *this << "\n";
	splitRules(rules,p,ps->rules);
//cout << "OBJECTS: ";
//for_each(objects.begin(),objects.end(),ptrwriter<const_symbol>(cout," "));
//cout << "\n";
	splitObjects(objects,p,ps->objects);
	properties.erase(remove(properties.begin(),properties.end(),p),properties.end());
	checkStateValued();
	return ps;
};

bool Property::applicableTo(TypeChecker & tc,const pddl_type * tp) const
{
// Caution: won't work if predicate has an either type spec.
	pddl_type * x = (*(EPS(predicate)->tcBegin() + posn))->type;
	return tc.subType(tp,x);
};

bool PropertySpace::applicableTo(TypeChecker & tc,const pddl_type * tp) const
{
	for(vector<Property*>::const_iterator i = properties.begin();i != properties.end();++i)
	{
		if(!(*i)->applicableTo(tc,tp)) return false;
	};
	return true;
};

set<PropertySpace *> TIMAnalyser::relevant(pddl_type * tp)
{
// Really only want to use this on leaf types. Other types are more of a problem
// at both this end and the caller's end. This is only a temporary management of
// the situation.
	if(!tcheck.isLeafType(tp))
	{
		return set<PropertySpace*>();
	};
	set<PropertySpace*> st;
	for(vector<PropertySpace*>::iterator i = propspaces.begin();
			i != propspaces.end();++i)
	{
		if((*i)->applicableTo(tcheck,tp))
		{
			st.insert(*i);
		};
	};
	return st;
};

};

