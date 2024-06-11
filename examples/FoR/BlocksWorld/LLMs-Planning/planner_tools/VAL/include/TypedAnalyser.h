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

#ifndef __TYPEDANALYSER
#define __TYPEDANALYSER

#include "FuncAnalysis.h"

#include <vector>
#include <set>
#include <iostream>
#include <algorithm>

using std::find;
using std::vector;
using std::ostream;
using std::set;
using std::pair;

#include "ptree.h"
#include "VisitController.h"
#include "typecheck.h"
#include "Cascader.h"
#include "FastEnvironment.h"
#include "Environment.h"

/* Note that it is possible that there is still a problem with the construction 
 * of PropStores for timed initial literals. In particular, notice that the
 * construction of PropStores for standard cases is managed by the TypedPredSubstituter
 * which visits all the pred_decls after the end of the domain visit, causing them
 * to be set up. The same sort of thing might be needed to handle the case of timed
 * initial literals, by recording the times they occur in the TypedPredSubstituter
 * at the outset and then creating them before going on to the Analyser phase.
 */
 
namespace VAL {

extern TypeChecker * theTC;

class holding_pred_symbol;
class extended_pred_symbol;
struct Environment;
class Associater;

void cwrite(const parameter_symbol * p,ostream & o);

class PropInfo {
private: 
	static int x;
	int id;
public:
	PropInfo() : id(++x) {};
	int getId() const {return id;};
	virtual ~PropInfo() {};
};

inline ostream & operator<<(ostream & o,const PropInfo & pi)
{
	o << "--" << pi.getId();
	return o;
};



class PropStore {
public:
	virtual ~PropStore() {};
	virtual PropInfo * get(const proposition * p) const = 0;
	virtual void add(proposition * p,PropInfo * pi) = 0;
	virtual void write(ostream & o) const = 0;
	virtual void notify(void(extended_pred_symbol::*f)(operator_ *,const proposition *),
							operator_ * o,const proposition * p) {};
	virtual void notify(void(extended_pred_symbol::*f)(derivation_rule *,const proposition *),
							derivation_rule * o,const proposition * p) {};
	virtual PropInfo * get(FastEnvironment * f,const proposition * p) const = 0;
	virtual PropInfo * get(Environment * f,const proposition * p) const = 0;
	virtual PropInfo * partialGet(FastEnvironment * f,const proposition * p) const = 0;

	virtual extended_pred_symbol * getEP(FastEnvironment * f,const proposition * p) const = 0;
};


class SimplePropStore : public PropStore {
private:
	extended_pred_symbol * ep;
	typedef CascadeMap<parameter_symbol *,PropInfo> CMap;
	CMap records;
public:
	SimplePropStore() : ep(0) {};
	SimplePropStore(extended_pred_symbol * e) : ep(e) {};
	void setEP(extended_pred_symbol * e) {if(!ep) ep = e;};
	extended_pred_symbol * getEP() {return ep;};
	PropInfo * get(const proposition * p) const
	{
		return records.get(p->args->begin(),p->args->end());
	};
	PropInfo * get(FastEnvironment * f,const proposition * p) const
	{
		return records.get(makeIterator(f,p->args->begin()),
							makeIterator(f,p->args->end()));
	};
	PropInfo * get(Environment * f,const proposition * p) const
	{
		return records.get(makeIterator(f,p->args->begin()),
							makeIterator(f,p->args->end()));
	};
	void add(proposition * p,PropInfo * pi)
	{
		records.insert(p->args->begin(),p->args->end(),pi);
	};
	void write(ostream & o) const
	{
		records.write(o);
	};

 	extended_pred_symbol * getEP(FastEnvironment * f,const proposition * p) const
	{
//		cout << "Getting for " << p->head->getName() << "\n";
		return ep;
	};
	
	typedef CascadeMap<parameter_symbol *,PropInfo>::iterator iterator;
	
	iterator begin() {return records.begin();};
	iterator end() {return records.end();};
// Return first match if there is one, when f contains null values for 
// unbound arguments.
	PropInfo * partialGet(FastEnvironment * f,const proposition * p) const;
};

template<class I>
class TypeIterator {
private:
	I i;
public:
	TypeIterator(I & x) : i(x) {};
	
	TypeIterator & operator++() 
	{
		++i;
		return *this;
	};
	pddl_type * operator*() 
	{
		if(!(*i))
		{
			//cout << "Strange type (3)\n";
			return 0;
		};
		return (*i)->type;
	};
	bool operator==(const TypeIterator & t) const
	{
		return i == t.i;
	};
	bool operator!=(const TypeIterator & t) const
	{
		return i != t.i;
	};
};

template<class I>
TypeIterator<I> typeIt(I i)
{
	return TypeIterator<I>(i);
};


class TypeExtractor {
	vector<pair<pddl_type *, vector<const pddl_type *> > > * tpsSets;
	int far;
	int count;
public:
	TypeExtractor(vector<pair<pddl_type *, vector<const pddl_type *> > > & tps,int c) :
		tpsSets(&tps), far(0), count(c)
	{
/*		for(unsigned int i = 0;i < tps.size();++i)
		{
			if(tps[i].first) cout << tps[i].first->getName() << " -\n";
			for(unsigned int j = 0;j < tps[i].second.size();++j)
			{
				if(tps[i].second[j]) cout << "  " << tps[i].second[j]->getName() << "\n";
			};
		};
*/
	};
	TypeExtractor(int arity) : tpsSets(0), far(arity)
	{};
	 
	bool operator==(const TypeExtractor & t) const
	{
		return far==t.far;
	};

	TypeExtractor & operator++()
	{
		++far;
		return *this;
	};

	pddl_type * operator*() 
	{
// 3 x 2 x 2
// 12 elements
// 0	0 0 0	/1;mod 3, /3;mod 2, /6;mod 2
// 1	1 0 0
// 2	2 0 0
// 3	0 1 0
// 4	1 1 0
// 5	2 1 0
// 6	0 0 1
// 7	1 0 1
// 8	2 0 1
// 9	0 1 1
// 10	1 1 1
// 11	2 1 1
		int x = count;
		for(int i = 0; i < far;++i)
		{
			int n = (*tpsSets)[i].second.size();
//			cout << "Seeing set size " << n << " far: " << far << " count: " << count << " x: " << x << "\n";
			if(n >= 2) x /= n;
		};
		int n = (*tpsSets)[far].second.size();
		if(n >= 2) x %= n;
//		cout << "Looking up " << far << " " << n << " " << x << "\n";
		return const_cast<pddl_type *>(n>=2?(*tpsSets)[far].second[x]:
											(*tpsSets)[far].first);
	};
};

class CompoundPropStore : public PropStore {
private:
	typedef CascadeMap<pddl_type *,SimplePropStore> TMap;
	TMap records;

	vector<SimplePropStore *> stores;

public:
	CompoundPropStore(int c,vector<pair<pddl_type *,vector<const pddl_type *> > > & tps,TMap & t,extended_pred_symbol * e,Associater * a);

	extended_pred_symbol * getEP(FastEnvironment * f,const proposition * p) const
	{
		SimplePropStore * s = records.get(typeIt(makeIterator(f,p->args->begin())),
										typeIt(makeIterator(f,p->args->end())));
		if(s) return s->getEP();
		return 0;
	};

	PropInfo * get(const proposition * p) const
	{
		SimplePropStore * s = 
			records.get(typeIt(p->args->begin()),typeIt(p->args->end()));
		if(s) return s->get(p);
		return 0;
	};
	PropInfo * get(FastEnvironment * f,const proposition * p) const
	{
		SimplePropStore * s = 
			records.get(typeIt(makeIterator(f,p->args->begin())),
						typeIt(makeIterator(f,p->args->end())));
		if(s) return s->get(f,p);
		return 0;
	};
	PropInfo * get(Environment * f,const proposition * p) const
	{
		SimplePropStore * s = 
			records.get(typeIt(makeIterator(f,p->args->begin())),
						typeIt(makeIterator(f,p->args->end())));
		if(s) return s->get(f,p);
		return 0;
	};
	PropInfo * partialGet(FastEnvironment * f,const proposition * p) const;
	void add(proposition * p,PropInfo * pi)
	{
	// Note that the SimplePropStore should always be here already, since construction
	// puts them in.
	//cout << "Compound store\n";
		records.get(typeIt(p->args->begin()),typeIt(p->args->end()))->add(p,pi);
	};
	
	void write(ostream & o) const
	{
		for(vector<SimplePropStore *>::const_iterator i = stores.begin();
						i != stores.end();++i)
		{
			(*i)->write(o);
		};
	};
	void notify(void(extended_pred_symbol::*f)(operator_ *,const proposition *),
						operator_ * o,const proposition * p);

	typedef vector<SimplePropStore *>::iterator iterator;
	iterator begin() {return stores.begin();};
	iterator end() {return stores.end();};
};

class FastEnvironment;

class PropInfoFactory {
protected:
	static PropInfoFactory * pf;
public:
	static PropInfoFactory & instance()
	{
		if(!pf) pf = new PropInfoFactory();
		return *pf;
	};
	static void setInstance(PropInfoFactory * p)
	{
		if(pf) delete pf;
		pf = p;
	};
	virtual ~PropInfoFactory() {};
	virtual PropInfo * createPropInfo(proposition * p) 
	{
		return new PropInfo();
	};
	virtual PropInfo * createPropInfo(proposition * p,FastEnvironment * fe)
	{
		return new PropInfo();
	};
};

struct OpProp {
	
	operator_* op;
	derivation_rule* drv;
	const proposition* second;
	

	OpProp(operator_* o, const proposition * p) : op(o), drv(0), second(p) {};
	OpProp(derivation_rule* o, const proposition *p) : op(0), drv(o), second(p) {};
};

typedef vector<pddl_typed_symbol *> Types;

class extended_pred_symbol : public pred_symbol {
public:
	typedef vector<OpProp> OpProps;
protected:
	holding_pred_symbol * parent;

	
// The Types structure contains symbols whose types are what we want, rather than
// the symbols themselves. This means that we always have one layer of indirection
// to get to the types. 
	Types types;
	
	int initialState;
	int posgoalState;
	int neggoalState;
	OpProps pospreconds;
	OpProps negpreconds;
	OpProps adds;
	OpProps dels;


	PropStore * props;

	PropStore * records() const;

	map<double,PropStore *> timedInitials;
	
	PropStore * getAt(double t) const;

	bool owner;
	
	
public:
	virtual ~extended_pred_symbol() 
	{
		if(owner)
		{
			for(unsigned int i = 0;i < types.size();++i)
			{
				types[i]->type = 0;
				delete types[i];
			};
		};
	};
	extended_pred_symbol(pred_symbol * nm,proposition * p) : pred_symbol(*nm), 
			parent(0), types(p->args->size()), initialState(0), posgoalState(0), 
			neggoalState(0),
			pospreconds(), negpreconds(), adds(), dels(), props(0),
			owner(false)
	{
			int i = 0;
			for(parameter_symbol_list::iterator j = p->args->begin();
					j != p->args->end();++j,++i)
			{
				 types[i] = (*j); //->type;
			};
	};
	template<class TI>
	extended_pred_symbol(pred_symbol * nm,TI s,TI e) : pred_symbol(*nm), 
			parent(0), types(e-s,(pddl_type *)0), initialState(0), posgoalState(0), 
			neggoalState(0),
			pospreconds(), negpreconds(), adds(), dels(), props(0), owner(true)
	{
//			cout << "Built for: " << nm->getName() << " " << (e-s) << "\n";
			for(int i = 0;s != e;++s,++i)
			{
//				cout << (*s)->getName() << "\n";
				 types[i] = new pddl_typed_symbol();
				 types[i]->type = (*s);
			};
//			cout << "Completed: ";
//			writeName(cout);
//			cout << "\n";
	};
	Types::iterator tBegin() {return types.begin();};
	Types::iterator tEnd() {return types.end();};
	Types::const_iterator tcBegin() const {return types.begin();};
	Types::const_iterator tcEnd() const {return types.end();};
	OpProps::const_iterator posPresBegin() const {return pospreconds.begin();};
	OpProps::const_iterator posPresEnd() const {return pospreconds.end();};
	OpProps::const_iterator addsBegin() const {return adds.begin();};
	OpProps::const_iterator addsEnd() const {return adds.end();};
	OpProps::const_iterator delsBegin() const {return dels.begin();};
	OpProps::const_iterator delsEnd() const {return dels.end();};
	int arity() const {return types.size();};
	
	holding_pred_symbol * getParent() const {return parent;};

	void setParent(holding_pred_symbol * h) {parent = h;};
	void setGoal(bool posGl) 
	{
		if(posGl) 
		{
			++posgoalState;
		} 
		else 
		{
			++neggoalState;
		};
	};
	void setInitial(proposition * p) 
	{
		PropInfo * pi = PropInfoFactory::instance().createPropInfo(p);
		vector<parameter_symbol *> eitherTypes;
		vector<pddl_type_list::iterator> params;
		for(parameter_symbol_list::iterator i = p->args->begin();i != p->args->end();++i)
		{
			if(!((*i)->type)) 
			{
				if((*i)->either_types)
				{
					eitherTypes.push_back(*i);
					params.push_back((*i)->either_types->begin());
				};
			};
		};
		if(eitherTypes.empty())
		{
			records()->add(p,pi);
		}
		else
		{
			PropStore * ps = records();
			while(params[0] != eitherTypes[0]->either_types->end())
			{
				for(unsigned int i = 0;i < eitherTypes.size();++i)
				{
					eitherTypes[i]->type = *(params[i]);
				};
				//cout << *p << "\n";  
				ps->add(p,pi);
				int k = params.size()-1;
				while(k >= 0)
				{
					//cout << "Incrementing " << k << "\n";
					++params[k];
					if(k && params[k] == eitherTypes[k]->either_types->end())
					{
						params[k] = eitherTypes[k]->either_types->begin();
						--k;
						//cout << "End of the line, moving back\n";
					}
					else
					{
						break;
					};
				};
			};
			for(unsigned int i = 0;i < params.size();++i)
			{
				eitherTypes[i]->type = 0;
			};
		};
		++initialState;
	};
// Note that the machinery for handling either types has not been threaded 
// into the timed initial literals!
	void setInitialPos(proposition * p,double t)
	{
		getAt(t)->add(p,PropInfoFactory::instance().createPropInfo(p));
	};
	void setInitialNeg(proposition * p,double t)
	{
		setInitialPos(p,-t);
	};
	int isPosGoal() const {return posgoalState;};
	int isNegGoal() const {return neggoalState;};
	int isInitial() const {return initialState;};

	bool isStatic() const 
	{
		return adds.empty() && dels.empty();
	};
	
	bool isCompletelyStatic(FastEnvironment * f,const proposition * p) const
	{
		if(!appearsStatic()) return false;
		if(isDefinitelyStatic()) return true;

		extended_pred_symbol * eps = records()->getEP(f,p);
/*		if(eps) 
		{
			cout << "Final check with " << *eps << "\n";
		}
		else
		{
			cout << "Final fail with " << p->head->getName() << "\n";
		};
*/		
		if(eps) return eps->appearsStatic();
		return false;
	};

	bool cannotIncrease() const
	{
		return timedInitials.empty() && adds.empty() && isPrimitiveType();
	}
	
	bool isDefinitelyStatic() const
	{
		if(!appearsStatic()) return false;
		if(types.empty()) return true;
/*
 		vector<vector<const pddl_type *> > alltps;
		for(Types::const_iterator i = types.begin();i != types.end();++i)
		{
			if(!theTC->isLeafType((*i)->type))
			{
				cout << (*i)->type->getName() << " ";
				alltps.push_back(theTC->leaves((*i)->type));
			}
			else
			{
				alltps.push_back(vector<const pddl_type *>(1,(*i)->type));
			};
		};
		cout << "Got " << alltps.size() << " of sizes: ";
		for(vector<vector<const pddl_type *> >::const_iterator i = alltps.begin();
			i != alltps.end();++i)
			{
				cout << i->size() << " ";
			}
			cout << "\n";
*/
		return isPrimitiveType();
	}

	bool isPrimitiveType() const
	{
		for(Types::const_iterator i = types.begin();i != types.end();++i)
		{
			if(!theTC->isLeafType((*i)->type))
			{
//				cout << (*i)->type->getName() << " ";
				return false;
			};
		};
		return true;
	};

	extended_pred_symbol * getPrimitive(FastEnvironment * f,const proposition * p)
	{
		return records()->getEP(f,p);
	};
		
	bool appearsStatic() const
	{
		return isStatic() && timedInitials.empty();
	};
	
	bool decays() const {return adds.empty() && !dels.empty();};
	
	void addPosPre(operator_ * o,const proposition * p) 
	{
		pospreconds.push_back(OpProp(o,p));
		records()->notify(&extended_pred_symbol::addPosPre,o,p);
	};
	void addNegPre(operator_ * o,const proposition * p) 
	{
		negpreconds.push_back(OpProp(o,p));
		records()->notify(&extended_pred_symbol::addNegPre,o,p);
	};
	void addAdd(operator_ * o,const proposition * p) 
	{
		adds.push_back(OpProp(o,p));
		records()->notify(&extended_pred_symbol::addAdd,o,p);
	};
	void addDel(operator_ * o,const proposition * p) 
	{
		dels.push_back(OpProp(o,p));
		records()->notify(&extended_pred_symbol::addDel,o,p);
	};

	void addPosPre(derivation_rule * o,const proposition * p) 
	{
		pospreconds.push_back(OpProp(o,p));
		records()->notify(&extended_pred_symbol::addPosPre,o,p);
	};
	void addNegPre(derivation_rule * o,const proposition * p) 
	{
		negpreconds.push_back(OpProp(o,p));
		records()->notify(&extended_pred_symbol::addNegPre,o,p);
	};
	void addAdd(derivation_rule * o,const proposition * p) 
	{
		adds.push_back(OpProp(o,p));
		records()->notify(&extended_pred_symbol::addAdd,o,p);
	};

	void writeName(ostream & o) const
	{
		o << getName() << "(";
		for(Types::const_iterator i = types.begin();i != types.end();++i)
		{
			if(!(*i))
			{
				//cout << "Strange type\n";
				continue;
			};
			if((*i)->type) 
			{
				o << (*i)->type->getName() << " ";
				o << "[";
				vector<const pddl_type *> ls = theTC->leaves((*i)->type);
				for(vector<const pddl_type *>::const_iterator x = ls.begin();
						x != ls.end();++x)
				{
					o << (*x)->getName() << " ";
				};
				o << "] ";
			}
			else
			{
				o << "? ";
			};
		};
		
		o << ")";
	};
	
	void write(ostream & o) const
	{
		o << "\nReport for: ";
		writeName(o);
		o << "\n------------\nInitial: " << initialState 
				<< " Goal: " << posgoalState << " positive/ " << neggoalState << 
					" negative\nInitial state records:\n";
		records()->write(o);
		o << "\nPreconditions:\n";
		if(!negpreconds.empty()) o << "+ve:\n";
		for(OpProps::const_iterator i = pospreconds.begin();
				i != pospreconds.end();++i)
		{
			if(i->op) o << "\t" << i->op->name->getName() << "\n";
			if(i->drv)   o << "\t" << i->drv->get_head()->head->getName() << "\n";
		};
		if(!negpreconds.empty())
		{
			o << "-ve:\n";
			for(OpProps::const_iterator i = negpreconds.begin();
					i != negpreconds.end();++i)
			{
				if(i->op) o << "\t" << i->op->name->getName() << "\n";
				if(i->drv)   o << "\t" << i->drv->get_head()->head->getName() << "\n";
			};
		};
		if(appearsStatic()) 
		{
			o << "Proposition appears static\n";
			if(isDefinitelyStatic())
			{
				o << "\tReally is static\n";
			};
			return;
		};
		if(decays())
		{
			o << "Proposition decays only\n";
		}
		else
		{
			o << "Adds:\n";
			for(OpProps::const_iterator i = adds.begin();
					i != adds.end();++i)
			{
				if(i->op) o << "\t" << i->op->name->getName() << "\n";
				if(i->drv)   o << "\t" << i->drv->get_head()->head->getName() << "\n";
			};
		};
		o << "Dels:\n";
		for(OpProps::const_iterator i = dels.begin();
				i != dels.end();++i)
		{
			if(i->op) o << "\t" << i->op->name->getName() << "\n";
			if(i->drv)   o << "\t" << i->drv->get_head()->head->getName() << "\n";
		};
		if(cannotIncrease())
		{
			o << "Cannot increase\n";
		}

	};
	void visit(VisitController * v) const
	{
		write(cout);
	};
	bool contains(FastEnvironment * f,const proposition * p) const
	{
		return records()->get(f,p);
	};
	bool partContains(FastEnvironment * f,const proposition * p) const
	{
		return records()->partialGet(f,p);
	};
	PropStore * getInitials() {return props;};
	vector<double> getTimedAchievers(Environment * f,const proposition * prop) const;
};

typedef vector<pddl_type *>::iterator EPSTypeIterator;

struct EPSBuilder {
	virtual ~EPSBuilder() {};
	virtual extended_pred_symbol * operator()(pred_symbol * nm,proposition * p)
	{
		return new extended_pred_symbol(nm,p);
	};
	virtual extended_pred_symbol * operator()(pred_symbol * nm,
		EPSTypeIterator s, EPSTypeIterator e)
	{
		return new extended_pred_symbol(nm,s,e);
	};
};

template<class EPS_T>
struct specEPSBuilder : public EPSBuilder {
	virtual extended_pred_symbol * operator()(pred_symbol * nm,proposition * p)
	{
		return new EPS_T(nm,p);
	};
	virtual extended_pred_symbol * operator()(pred_symbol * nm,
		EPSTypeIterator s,EPSTypeIterator e)
	{
		return new EPS_T(nm,s,e);
	};
};


#define EPS(x) static_cast<VAL::extended_pred_symbol*>(const_cast<VAL::pred_symbol*>(x))

class Associater {
public:
	static auto_ptr<EPSBuilder> buildEPS;
	virtual ~Associater() {};
	virtual Associater * lookup(pddl_type * p)
	{
		return 0;
	};
	
	virtual extended_pred_symbol * get()
	{
		return 0;
	};

	virtual void set(pddl_type *,Associater *) {};
	
	Associater * handle(proposition * p);

	template<class TI>
	extended_pred_symbol * find(pred_symbol * p,TI s,TI e);
};

class NodeAssociater : public Associater {
private:
	map<pddl_type *,Associater *> assoc;
public:
	Associater * lookup(pddl_type * p)
	{
		return assoc[p];
	};

	void set(pddl_type * t,Associater * a)
	{
		assoc[t] = a;
	};
};



class holding_pred_symbol : public pred_symbol {
private:
	Associater * a;
	typedef set<extended_pred_symbol *> Preds;
	Preds preds;

	typedef CascadeMap<pddl_type *,SimplePropStore> TMap;
	TMap baseStores;

	typedef map<double,TMap> TimedBases;
	TimedBases timedBases;
	
public:
	holding_pred_symbol(const string & nm) : pred_symbol(nm), a(new NodeAssociater()),
		preds(), baseStores(), timedBases()
	{};

	void set_prop(proposition * p)
	{
		Associater * aa = a->handle(p);
		if(a != aa)
		{
			delete a;
			a = aa;
		};
	};

	template<class TI>
	extended_pred_symbol * find(pred_symbol* p,TI s,TI e)
	{
		return a->find(p,s,e);
	};

	void add(extended_pred_symbol * e) 
	{
//		cout << "Added " << (int)(e) << " ";
//		e->writeName(cout);
//		cout << "\n";
		preds.insert(e);
		e->setParent(this);
	};
	
	void visit(VisitController * v) const
	{
		for(Preds::const_iterator i = preds.begin();i != preds.end();++i)
		{
			(*i)->visit(v);
		};
	};

	void buildPropStore()
	{
		for(Preds::iterator i = preds.begin();i != preds.end();++i)
		{
			buildPropStore((*i),(*i)->tBegin(),(*i)->tEnd());
		};
	};

	template<typename TI>
	PropStore * buildPropStore(extended_pred_symbol * e,TI ai,TI bi) //const vector<pddl_type *> & tps)
	{
//		cout << "I am at " << this << "\n";
//		cout << "Counting for ";
//		e->writeName(cout); 
//		cout << "\n";
		int c = 1;
		vector<pair<pddl_type *,vector<const pddl_type *> > > xs(bi-ai);//tps.size());
		int j = 0;
		if(current_analysis->the_domain->isTyped())
		{
			for(Types::const_iterator i = ai;i != bi;++i,++j) //tps.begin();i != tps.end();++i,++j)
			{
				if(!(*i))
				{
					//cout << "Strange type (2)\n";
					continue;
				};
				
				if((*i)->type)
				{
//					cout << "Managing: " << (*i)->type->getName() << "\n";
					xs[j] = make_pair((*i)->type,//theTC->accumulateAll((*i)->type)); 
												theTC->leaves((*i)->type));
					
				}
				else
				{
					if(!(*i)->either_types)
					{
//						cout << "Expected either types and there aren't any!\n";
//						e->writeName(cout);
//						cout << "\n";
						continue;
					};
					//cout << "An either type symbol!\n";
	// OK, I think we have to find all the leaves for these either types. 
					vector<const pddl_type *> ttps;
					for(pddl_type_list::iterator k = (*i)->either_types->begin();k != (*i)->either_types->end();++k)
					{
						vector<const pddl_type *> tlvs = theTC->leaves(*k);
						copy(tlvs.begin(),tlvs.end(),back_inserter(ttps));
						if(tlvs.empty())
						{
							ttps.push_back(*k);
						};
					};
					xs[j] = make_pair((pddl_type*)0,ttps);
				};
				int x = xs[j].second.size();
//				cout << "Has " << x << " leaves\n";
				if(x >= 2) c *= x;
			};
		};
		if(c == 1)
		{
			SimplePropStore * s = baseStores.get(typeIt(ai),typeIt(bi)); //tps.begin(),tps.end());
			if(!s) 
			{
//				cout << "Built simple store\n";
				s = new SimplePropStore(e);
				baseStores.insert(typeIt(ai),typeIt(bi),s); //tps.begin(),tps.end(),s);
			}
			else
			{
				s->setEP(e);
			};
			return s;
		}
		else
		{
//			cout << "Building for " << e->getName() << " " << c << " elements\n";
			return new CompoundPropStore(c,xs,baseStores,e,a);
		};
	};

	PropStore * buildPropStore(extended_pred_symbol * e,const Types & tps,
									double t)
	{
		int c = 1;
		vector<pair<pddl_type *,vector<const pddl_type *> > > xs(tps.size());
		int j = 0;
		for(Types::const_iterator i = tps.begin();i != tps.end();++i,++j)
		{
// Note that this probably needs fixing like the previous function to handle
// either types
			xs[j] = make_pair((*i)->type,theTC->leaves((*i)->type));
			int x = xs[j].second.size();
			if(x >= 2) c *= x;
		};
		if(c == 1)
		{
			SimplePropStore * s = timedBases[t].get(typeIt(tps.begin()),typeIt(tps.end()));
			if(!s) 
			{
				s = new SimplePropStore(e);
				timedBases[t].insert(typeIt(tps.begin()),typeIt(tps.end()),s);
			}
			else
			{
				s->setEP(e);
			};
			return s;
		}
		else
		{
			return new CompoundPropStore(c,xs,timedBases[t],e,a);
		};
	};
	
	typedef Preds::iterator PIt;
	PIt pBegin() {return preds.begin();};
	PIt pEnd() {return preds.end();};
};





#define HPS(x) const_cast<VAL::holding_pred_symbol *>(static_cast<const VAL::holding_pred_symbol*>(x))

class LeafAssociater : public Associater {
private:
	extended_pred_symbol * s;
public:
	LeafAssociater(pred_symbol * nm,proposition * p) : 
		s((*buildEPS)(nm,p)) 
	{};
	template<class TI>
	LeafAssociater(pred_symbol * nm,TI st,TI e) : 
		s((*buildEPS)(nm,st,e)) 
	{
		TI x(st);
		while(x != e)
		{
//			if(*x) cout << "**" << (*x)->getName() << " ";
			++x;
		};
//		cout << "I just made ";
//		s->writeName(cout);
//		cout << "\n";
	};
	extended_pred_symbol * get()
	{
		return s;
	};
};

template<class TI>
extended_pred_symbol * Associater::find(pred_symbol * p,TI s,TI e)
{
	TI sorig(s);
	Associater * a = this;
	while(!(s == e))
	{
		pddl_type * t = *s;
		Associater * aa = a->lookup(t);
		++s;
		if(!aa)
		{
//			cout << "Not found " << t->getName() << "\n";
			if(s == e)
			{
				vector<pddl_type *> v;
				copy(sorig,e,inserter(v,v.begin()));
				aa = new LeafAssociater(p,v.begin(),v.end());
//				cout << "Made a LA\n";
			}
			else
			{
//				cout << "Made a NA\n";
				aa = new NodeAssociater();
			};
			a->set(t,aa);
		};
		a = aa;
	};
	return a->get();
};

class TypePredSubstituter : public VisitController {
public: 	
	virtual void visit_simple_goal(simple_goal * s)
	{
		s->getProp()->visit(this);
	};
	virtual void visit_proposition(proposition * p)
	{
		HPS(p->head)->set_prop(p);
	};
	virtual void visit_qfied_goal(qfied_goal * p) 
	{p->getGoal()->visit(this);};
	virtual void visit_conj_goal(conj_goal * p) 
	{p->getGoals()->visit(this);};
	virtual void visit_disj_goal(disj_goal * p) 
	{p->getGoals()->visit(this);};
	virtual void visit_timed_goal(timed_goal * p) 
	{p->getGoal()->visit(this);};
	virtual void visit_imply_goal(imply_goal * p) 
	{
		p->getAntecedent()->visit(this);
		p->getConsequent()->visit(this);
	};
	virtual void visit_neg_goal(neg_goal * p) 
	{
		p->getGoal()->visit(this);
	};
	virtual void visit_simple_effect(simple_effect * p) 
	{
		p->prop->visit(this);
	};
	virtual void visit_forall_effect(forall_effect * p) 
	{
		p->getEffects()->visit(this);
	};
	virtual void visit_cond_effect(cond_effect * p) 
	{
		p->getCondition()->visit(this);
		p->getEffects()->visit(this);
	};
	virtual void visit_timed_effect(timed_effect * p) 
	{
		p->effs->visit(this);
	};
	virtual void visit_timed_initial_literal(timed_initial_literal * p)
	{
		p->effs->visit(this);
	};
	virtual void visit_effect_lists(effect_lists * p) 
	{
		p->add_effects.pc_list<simple_effect*>::visit(this);
		p->forall_effects.pc_list<forall_effect*>::visit(this);
		p->cond_effects.pc_list<cond_effect*>::visit(this);
		p->timed_effects.pc_list<timed_effect*>::visit(this);
		p->del_effects.pc_list<simple_effect*>::visit(this);
	};
	virtual void visit_operator_(operator_ * p) 
	{
		p->precondition->visit(this);
		p->effects->visit(this);
	};
	virtual void visit_derivation_rule(derivation_rule * r)
	{
		if (r->get_body()) r->get_body()->visit(this);
		visit_proposition(r->get_head());
	};
	virtual void visit_action(action * p)
	{
		visit_operator_(static_cast<operator_*>(p));
	};
	virtual void visit_durative_action(durative_action * p) 
	{
		visit_operator_(static_cast<operator_*>(p));
	};
	virtual void visit_process(process * p)
	{
		visit_operator_(static_cast<operator_*>(p));
	};
	virtual void visit_event(event * p)
	{
		visit_operator_(static_cast<operator_*>(p));
	};
	virtual void visit_problem(problem * p) 
	{
		p->initial_state->visit(this);
		if(p->the_goal) p->the_goal->visit(this);
	};

	virtual void visit_domain(domain * p) 
	{
		visit_operator_list(p->ops);
		if (p->drvs) visit_derivations_list(p->drvs);
		if(p->predicates) p->predicates->visit(this);
	};

	virtual void visit_pred_decl(pred_decl * p)
	{
		HPS(p->getPred())->buildPropStore();
	};
};

class Analyser : public VisitController {
private:
	bool initially;
	double when;
	bool finally;
	bool pos;
	bool adding;
	operator_ * op;
	derivation_rule * drv;

	vector<durative_action *> das;

	vector<pred_symbol *> toIgnore;

	bool filterFn(pred_symbol * p)
	{
		return (find(toIgnore.begin(),toIgnore.end(),p) == toIgnore.end());
	};
	
public:
	Analyser() : initially(false), when(0), finally(false), 
		pos(true), adding(true), op(0), drv(0), toIgnore()
	{};
	
	Analyser(const vector<pred_symbol *> & ti) : initially(false), when(0), finally(false),
		pos(true), adding(true), op(0), drv(0), toIgnore(ti)
	{};

	vector<durative_action *> & getFixedDAs() 
	{
		return das;
	};
	
	virtual void visit_pred_decl(pred_decl * p) 
	{
		p->getPred()->visit(this);
	};
	virtual void visit_func_decl(func_decl * p) 
	{
		p->getFunction()->visit(this);
	};
	virtual void visit_simple_goal(simple_goal * p) 
	{
		if(finally) 
		{
			EPS(p->getProp()->head)->setGoal(pos);
		}
		else
		{
			if(filterFn(p->getProp()->head))
			{
				if(pos)
				{
					if (op) EPS(p->getProp()->head)->addPosPre(op,p->getProp());
					if (drv) EPS(p->getProp()->head)->addPosPre(drv,p->getProp());
				}
				else
				{
					if (op) EPS(p->getProp()->head)->addNegPre(op,p->getProp());
					if (drv) EPS(p->getProp()->head)->addNegPre(drv,p->getProp());
				};
			};
		};
	};
	virtual void visit_qfied_goal(qfied_goal * p) 
	{p->getGoal()->visit(this);};
	virtual void visit_conj_goal(conj_goal * p) 
	{if(p) p->getGoals()->visit(this);};
	virtual void visit_disj_goal(disj_goal * p) 
	{p->getGoals()->visit(this);};
	virtual void visit_timed_goal(timed_goal * p) 
	{p->getGoal()->visit(this);};
	virtual void visit_imply_goal(imply_goal * p) 
	{
		p->getAntecedent()->visit(this);
		p->getConsequent()->visit(this);
	};
	virtual void visit_neg_goal(neg_goal * p) 
	{
		pos = !pos;
		p->getGoal()->visit(this);
		pos = !pos;
	};
	virtual void visit_simple_effect(simple_effect * p) 
	{
		if(initially)
		{
			if(when>0)
			{
				if(adding)
				{
					EPS(p->prop->head)->setInitialPos(p->prop,when);
				}
				else
				{
					EPS(p->prop->head)->setInitialNeg(p->prop,when);
				};
			}
			else
			{
				EPS(p->prop->head)->setInitial(p->prop);
			};
		}
		else
		{
			if(filterFn(p->prop->head))
			{
				if(adding)
				{
					EPS(p->prop->head)->addAdd(op,p->prop);
				}
				else
				{
					EPS(p->prop->head)->addDel(op,p->prop);
				};
			};
		};
	};
	virtual void visit_forall_effect(forall_effect * p) 
	{
		p->getEffects()->visit(this);
	};
	virtual void visit_cond_effect(cond_effect * p) 
	{
		p->getCondition()->visit(this);
		p->getEffects()->visit(this);
	};
	virtual void visit_timed_effect(timed_effect * p) 
	{
		p->effs->visit(this);
	};
	virtual void visit_timed_initial_literal(timed_initial_literal * p)
	{
		when = p->time_stamp;
		p->effs->visit(this);
		when = 0;
	};
	virtual void visit_effect_lists(effect_lists * p) 
	{
		p->assign_effects.pc_list<assignment*>::visit(this);
		p->add_effects.pc_list<simple_effect*>::visit(this);
		p->forall_effects.pc_list<forall_effect*>::visit(this);
		p->cond_effects.pc_list<cond_effect*>::visit(this);
		p->timed_effects.pc_list<timed_effect*>::visit(this);
		bool whatwas = adding;
		adding = !adding;
		p->del_effects.pc_list<simple_effect*>::visit(this);
		adding = whatwas;
	};
	virtual void visit_operator_(operator_ * p) 
	{
		op = p;
		adding = true;
		p->precondition->visit(this);
		p->effects->visit(this);
		op = 0;
	};
	virtual void visit_derivation_rule(derivation_rule * p) 
	{
		drv = p;
		adding = true;
		p->get_body()->visit(this);
		if(filterFn(p->get_head()->head)) EPS(p->get_head()->head)->addAdd(drv,p->get_head());
		drv = 0;
	};
	virtual void visit_action(action * p)
	{
		visit_operator_(static_cast<operator_*>(p));
	};
	virtual void visit_durative_action(durative_action * p) 
	{
		visit_operator_(static_cast<operator_*>(p));
		das.push_back(p);
	};
	virtual void visit_process(process * p)
	{
		visit_operator_(static_cast<operator_*>(p));
	};
	virtual void visit_event(event * p)
	{
		visit_operator_(static_cast<operator_*>(p));
	};
	virtual void visit_problem(problem * p) 
	{
		initially = true;
		p->initial_state->visit(this);
		initially = false;
		finally = true;
		if(p->the_goal) p->the_goal->visit(this);
		finally = false;
	};

	virtual void visit_domain(domain * p) 
	{
		visit_operator_list(p->ops);
		if (p->drvs) visit_derivations_list(p->drvs);
		vector<durative_action *> fdas;
		for(vector<durative_action *>::iterator i = das.begin();i != das.end();++i)
		{
			timed_goal * tg = dynamic_cast<timed_goal*>((*i)->dur_constraint);
			const comparison * c = 0;
			if(tg) c = dynamic_cast<const comparison *>(tg->getGoal());
			if(c)
			{
				if(c->getOp() == E_EQUALS)
				{
					// Should check that LHS is duration variable - but we will 
					// be lazy and assume it is for the moment!
					const expression * e = c->getRHS();
					AbstractEvaluator ae;
					e->visit(&ae);
					if(ae().isConstant())
					{
						fdas.push_back(*i);
					};
				};
			}
				
		};
		fdas.swap(das);
	};

	virtual void visit_assignment(assignment * p)
	{
		switch(p->getOp())
		{
			case E_ASSIGN:
				if(initially)
				{
					EFT(p->getFTerm()->getFunction())->addInitial(p);
				}
				else
				{
					EFT(p->getFTerm()->getFunction())->addAssign(op,p);
				};
				break;
			case E_INCREASE:
				EFT(p->getFTerm()->getFunction())->addIncreaser(op,p);
				break;
			case E_DECREASE:
				EFT(p->getFTerm()->getFunction())->addDecreaser(op,p);
				break;
			default:
				EFT(p->getFTerm()->getFunction())->addOther(op,p);
				break;
		};
	};

	virtual void visit_comparison(comparison * p)
	{
		p->getLHS()->visit(this);
		p->getRHS()->visit(this);
	};

	virtual void visit_plus_expression(plus_expression *p) 
	{
		p->getLHS()->visit(this);
		p->getRHS()->visit(this);		
	};
	virtual void visit_minus_expression(minus_expression *p) 
	{
		p->getLHS()->visit(this);
		p->getRHS()->visit(this);
	};
	virtual void visit_mul_expression(mul_expression *p) 
	{
		p->getLHS()->visit(this);
		p->getRHS()->visit(this);
	};
	virtual void visit_div_expression(div_expression *p) 
	{
		p->getLHS()->visit(this);
		p->getRHS()->visit(this);
	};
	virtual void visit_uminus_expression(uminus_expression *p) 
	{
		p->getExpr()->visit(this);
	};
	virtual void visit_func_term(func_term *p) 
	{
		if(finally) 
		{
			EFT(p->getFunction())->addGoal();
		}
		else {
			if (op) EFT(p->getFunction())->addPre(op);
			if (drv) EFT(p->getFunction())->addPre(drv);
		}
	};
};

};

#endif
