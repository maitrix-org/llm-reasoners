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

/*-----------------------------------------------------------------------------
  VAL - The Automatic Plan Validator for PDDL+

  $Date: 2009-02-05 10:50:22 $
  $Revision: 1.2 $

  Maria Fox, Richard Howey and Derek Long - PDDL+ and VAL
  Stephen Cresswell - PDDL Parser

  maria.fox@cis.strath.ac.uk
  derek.long@cis.strath.ac.uk
  stephen.cresswell@cis.strath.ac.uk
  richard.howey@cis.strath.ac.uk

  By releasing this code we imply no warranty as to its reliability
  and its use is entirely at your own risk.

  Strathclyde Planning Group
  http://planning.cis.strath.ac.uk
 ----------------------------------------------------------------------------*/
#include "ptree.h"
#include <iostream>
#include <set>
#include "Environment.h"
#include "Ownership.h"
#include "Polynomial.h"



#ifndef __PROPOSITION
#define __PROPOSITION

//#define map std::map
//#define vector std::vector

using std::vector;
using std::set;
using std::map;

namespace VAL {
  
class State;
class Action;



struct ActiveCtsEffects;
class DerivedGoal;
class AdviceProposition;

bool isPointInInterval(CoScalar p,const vector< pair<intervalEnd,intervalEnd> > & ints);
bool isPointInInterval(CoScalar p, const pair<intervalEnd,intervalEnd> & ints);
bool isPointInInterval(CoScalar p, const vector< pair<intervalEnd,intervalEnd> > & ints, const pair<intervalEnd,intervalEnd> & int1);
pair<intervalEnd,intervalEnd> getIntervalFromPt(intervalEnd p, const vector< pair<intervalEnd,intervalEnd> > & ints,const pair<intervalEnd,intervalEnd> & int1);



Intervals setIntersect(const Intervals & ints1,const Intervals & ints2);
Intervals setUnion(const Intervals & ints1,const Intervals & ints2);
Intervals setComplement(const Intervals & ints,double endPoint);

class Proposition {
protected:
	const Environment & bindings;
	double endOfInterval;
public:
	virtual ~Proposition() {};
	Proposition(const Environment & bs) : bindings(bs),endOfInterval(0) {};
	virtual pair<int,int> rank() const = 0;
	virtual bool evaluate(const State* s,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const = 0;
	virtual Intervals getIntervals(const State* s) const = 0;
	virtual string getPropString(const State* s) const = 0;
	virtual set<const SimpleProposition *> getLiterals() const = 0;
	virtual const AdviceProposition * getAdviceProp(const State* s) const;
	virtual const AdviceProposition * getAdviceNegProp(const State* s) const;
	virtual bool markOwnedPreconditions(const Action* a,Ownership & o,ownership w) const = 0;

	virtual void setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen = false) = 0;
  virtual void resetCtsFunctions() = 0;
  virtual bool evaluateAtPointWithinError(const State* s,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const;
	double getEndOfInterval() const {return endOfInterval;};
	bool markOwnedPreconditions(const Action * a,Ownership & o) const
	{
		return markOwnedPreconditions(a,o,E_PPRE);
	};
	virtual void write(ostream & o) const
	{
		o << "Compound proposition...";
	};
	
	virtual void destroy() const
	{
		delete this;
	};
};

ostream & operator <<(ostream & o,const Proposition & p);

class SimpleProposition : public Proposition {
private:
	static Environment nullEnvironment;

	const proposition * prop;

public:
	SimpleProposition(const parse_category * p,const Environment &bs) :
		Proposition(bs), prop(dynamic_cast<const proposition*>(p))
	{};
	SimpleProposition(const parse_category *p) :
		Proposition(nullEnvironment), prop(dynamic_cast<const proposition*>(p))
	{};

	bool evaluate(const State * s,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const;

  ~SimpleProposition() {};



	Intervals getIntervals(const State* s) const;
	string getPropString(const State* s) const;
   string getPropName() const {return prop->head->getName();};
   string getParameter(int paraNo) const; 
   const AdviceProposition * getAdviceProp(const State* s) const;
   const AdviceProposition * getAdviceNegProp(const State* s) const;
	set<const SimpleProposition *> getLiterals() const;
  const_symbol_list * getConstants(var_symbol_list* variables,parameter_symbol_list* psl,Validator * vld) const;
  bool checkParametersConstantsMatch(parameter_symbol_list* psl) const;
  bool checkConstantsMatch(parameter_symbol_list* psl) const;
	pair<int,int> rank() const;
	bool markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const;
	void setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen = false);
	void resetCtsFunctions();


	const pred_symbol * getPred() const {return prop->head;};
	const proposition * getProp() const {return prop;};
	const Environment * getEnv() const {return &bindings;};
	void write(ostream & o) const;
	string toString() const;
	void destroy() const {};
};



class DerivedGoal : public Proposition {
private:
	static Environment nullEnvironment;


	const proposition * prop;
	

	const Proposition * deriveFormula;
	mutable bool revisit;

   static map<string,bool> DPliterals;
	static vector<string> calledDPsEval;
	static map<string, pair<int,int> > ranks;
	static const pair<int,int> noRank;
	static map<string,bool> evals;
	static const int noEval;
	static map<string,Intervals> intervals;
	static const Intervals noIntervals;
	static map<string,string> propStrings;
	static const string noPropString;
	static map<string,const Action *> preCons;
	static const ActiveCtsEffects * ace;
	static bool rhsOpen;

public:
	DerivedGoal(const parse_category * p,const Proposition * f,const Environment &bs) :
	  Proposition(bs), prop(dynamic_cast<const proposition*>(p)), deriveFormula(f),revisit(false)
	{};
	DerivedGoal(const parse_category *p,const Proposition * f) :
		Proposition(nullEnvironment),prop(dynamic_cast<const proposition*>(p)), deriveFormula(f), revisit(false)
	{};

	bool evaluate(const State * s,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const;
	Intervals getIntervals(const State* s) const;
	string getPropString(const State* s) const;
  set<const SimpleProposition *> getLiterals() const;
	pair<int,int> rank() const;
   const AdviceProposition * getAdviceProp(const State* s) const;
   const AdviceProposition * getAdviceNegProp(const State* s) const;
	bool markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const;
	void setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen = false);
	void resetCtsFunctions();
  
	void removeCalledDP(string dp) const;
	void addCalledDP(string dp) const;
        string getDPName() const;
	bool visited() const;
	bool visited(string dp) const;
    void setRevisit(bool b) const {revisit = b;};
	static void resetLists(const State* s);
	static void resetPreConsList()
	  {preCons.clear();};

	static void setACE(const ActiveCtsEffects * a,bool r)
	  {ace =a; rhsOpen = r;};
	    

	void write(ostream & o) const;


	~DerivedGoal()
	{
	  deriveFormula->destroy();
	};
};

//used when creating a derived predicate that may depend on itself
class FalseProposition : public Proposition {
private:
  bool trueProp; 

public:
	FalseProposition(const Environment &bs,bool tp = false) :
		Proposition(bs), trueProp(tp)
	{};

	bool evaluate(const State * s,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const;
	Intervals getIntervals(const State* s) const {Intervals theAns; return theAns;};
	string getPropString(const State* s) const {if(trueProp) return "true"; else return "false";};
  set<const SimpleProposition *> getLiterals() const {return set<const SimpleProposition *>();};
	pair<int,int> rank() const {return make_pair(0,0);};
	bool markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const;
	void setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen = false) {return;};
	void resetCtsFunctions() {return;};
	void write(ostream & o) const {if(trueProp) o << "(TRUE)"; else o << "(FALSE)";};	
};


class Comparison : public Proposition {
private:


	const comparison * comp;
	const CtsFunction * ctsFtn;
	bool rhsIntervalOpen;//only open for last interval that invariant is checked on		
public:
	Comparison(const comparison * c,const Environment & bs) :

		Proposition(bs), comp(c), ctsFtn(0)
	{};
	bool evaluate(const State * s,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const;
   	bool evaluateAtPoint(const State * s) const;
   	bool evaluateAtPointError(const State * s) const;
   	bool evaluateAtPointWithinError(const State* s,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const;
	Intervals getIntervals(const State* s) const;

   	const comparison * getComparison() const {return comp;};
   	string getPropString(const State* s) const;
   	string getPropAdviceString(const State* s) const;
   	string getExprnString(const expression * e,const Environment & bs, const State * s) const;
   	string getExprnString(const expression * e,const Environment & bs) const;
   	set<const SimpleProposition *> getLiterals() const;
	pair<int,int> rank() const;
		//first value, no comparison with changing fe s, send rank - max degree of polys for in condition
	bool markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const;
	void setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen = false);
  	void resetCtsFunctions();
   	const AdviceProposition * getAdviceProp(const State* s) const;
   	const AdviceProposition * getAdviceNegProp(const State* s) const;
	vector<CoScalar> getRoots(const State* s,CoScalar t) const;
	vector<CoScalar> getRootsForIntervals(const State* s,CoScalar t) const;
	void write(ostream & o) const;
	
  //void destroy() {/*delete this;*/};
  	~Comparison() {/*cout<<"deleting "<<*ctsFtn<<"\n";*/delete ctsFtn;};
};

class ConjGoal : public Proposition {
private:
	const vector<const Proposition *> gs;
public:
	ConjGoal(const conj_goal * c,const vector<const Proposition*> & g,const Environment & bs) :
		Proposition(bs), gs(g) 
	{};
	bool evaluate(const State *,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const;
	Intervals getIntervals(const State* s) const;
	string getPropString(const State* s) const;
   const AdviceProposition * getAdviceProp(const State* s) const;
   const AdviceProposition * getAdviceNegProp(const State* s) const;
  set<const SimpleProposition *> getLiterals() const;
	pair<int,int> rank() const;
	bool markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const;
	void setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen = false);
	void resetCtsFunctions();
  
	~ConjGoal()
	{
		for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
		{
			(*i)->destroy();
		};
	};
	void write(ostream & o) const;
};


class DisjGoal : public Proposition {
private:
	const vector<const Proposition *> gs;
public:
	DisjGoal(const disj_goal * d,const vector<const Proposition *> & g,const Environment & bs) :
		Proposition(bs), gs(g)
	{};
	bool evaluate(const State *,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const;
	Intervals getIntervals(const State* s) const;
	string getPropString(const State* s) const;
   const AdviceProposition * getAdviceProp(const State* s) const;
   const AdviceProposition * getAdviceNegProp(const State* s) const;
   set<const SimpleProposition *> getLiterals() const;
	pair<int,int> rank() const;
	bool markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const;
	void setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen = false);
	void resetCtsFunctions();
  
	~DisjGoal()
	{
		for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
		{
              (*i)->destroy();
		};
	};
	void write(ostream & o) const;
};

class ImplyGoal : public Proposition {
private:
	const Proposition * ant;
	const Proposition * cons;
public:
	ImplyGoal(const imply_goal * i,const Proposition * a,const Proposition * c,
					const Environment & bs) :
			Proposition(bs), ant(a), cons(c)
	{};
	bool evaluate(const State *,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const;
	Intervals getIntervals(const State* s) const;
	string getPropString(const State* s) const;
   const AdviceProposition * getAdviceProp(const State* s) const;
   const AdviceProposition * getAdviceNegProp(const State* s) const;
   set<const SimpleProposition *> getLiterals() const;
	pair<int,int> rank() const;
	bool markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const;
	void setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen = false);
	void resetCtsFunctions();
  
	~ImplyGoal()
	{
		ant->destroy();
		cons->destroy();
	};
	void write(ostream & o) const;
};

class NegGoal : public Proposition {
private:
	const Proposition * p;
public:
	NegGoal(const neg_goal * n,const Proposition * pp,const Environment & bs) :
		Proposition(bs), p(pp) 
	{};
	bool evaluate(const State *,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const;
	Intervals getIntervals(const State* s) const;
	string getPropString(const State* s) const;
   set<const SimpleProposition *> getLiterals() const;
   const AdviceProposition * getAdviceProp(const State* s) const;
   const AdviceProposition * getAdviceNegProp(const State* s) const;

	pair<int,int> rank() const;
	bool markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const;
	void setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen = false);
	void resetCtsFunctions();
  
	~NegGoal()
	{
		p->destroy();
	};
	void write(ostream & o) const;
};


class PreferenceGoal : public Proposition {
private:
	const preference * pref;
	const Proposition * thePref;

	Validator * vld;

public:
	PreferenceGoal(Validator * v,const preference * p,const Proposition * prp,const Environment & bs) :
		Proposition(bs), pref(p), thePref(prp), vld(v)
	{};
	bool evaluate(const State *,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const;
	Intervals getIntervals(const State* s) const;
	string getPropString(const State* s) const;
    set<const SimpleProposition *> getLiterals() const;
	pair<int,int> rank() const;
	bool markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const;
	void setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen = false);
	const AdviceProposition * getAdviceProp(const State * s) const;
    void resetCtsFunctions();
	~PreferenceGoal()
	{
		thePref->destroy();
	};
	void write(ostream & o) const;
};

class ConstraintGoal : public Proposition {
private:
	const constraint_goal * constraint;
	const Proposition * trigger;
	const Proposition * requirement;

	Validator * vld;
	
public:
	ConstraintGoal(Validator * v,const constraint_goal * cg,
		const Proposition * t,const Proposition * r,const Environment & bs) :
		Proposition(bs), constraint(cg), trigger(t), requirement(r), vld(v)
	{};
	
	constraint_sort getCons() const {return constraint->getCons();};
	const Proposition * getTrigger() const {return trigger;};
	const Proposition * getRequirement() const {return requirement;};
	double getFrom() const {return constraint->getFrom();};
	double getDeadline() const {return constraint->getDeadline();};
    void resetCtsFunctions();

	~ConstraintGoal()
	{
		trigger->destroy();
		requirement->destroy();
	};
	bool evaluate(const State *,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const;
	Intervals getIntervals(const State* s) const;
	string getPropString(const State* s) const;
    set<const SimpleProposition *> getLiterals() const;
	pair<int,int> rank() const;
	bool markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const;
	void setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen = false);
	const AdviceProposition * getAdviceProp(const State * s) const;
	void write(ostream & o) const;
};

/* Problem with quantified goals: Propositions are expected to have ground leaves,
 * but this is obviously not true of quantified goals. Should we expand them out
 * here? If so, how does this impact on the environment? Presumably we have a local
 * environment for this case, extending the general environment?
 * 

class QfiedGoal : public Proposition {
private:

	const qfied_goal * qfg;
	Proposition 

};

*/

class QfiedGoal : public Proposition {
private:
	const qfied_goal * qg;
	Environment * env;

	Validator * vld;

	// Mutable to allow single update. 
	mutable Proposition * pp;

	mutable vector<const Proposition *> props;
	mutable var_symbol_list::const_iterator i;
  bool createLiterals; //to do with evaluting qfied goals when we wish not to create literals
public:
	QfiedGoal(Validator * v,const qfied_goal * q,const Environment & bs,bool b = true) :
		Proposition(bs), qg(q), env(bs.copy(v)), vld(v), pp(0), 
		props(), i(qg->getVars()->begin()), createLiterals(b) {};

	void create() const;
	

	bool evaluate(const State * s,vector<const DerivedGoal*> = vector<const DerivedGoal*>()) const;
  bool evaluateQfiedGoal(const State * s,vector<const DerivedGoal*> DPs) const;
	Intervals getIntervals(const State* s) const;
	string getPropString(const State* s) const;
   set<const SimpleProposition *> getLiterals() const;
   const AdviceProposition * getAdviceProp(const State* s) const;
   const AdviceProposition * getAdviceNegProp(const State* s) const;
	pair<int,int> rank() const;
	bool markOwnedPreconditions(const Action *,Ownership & o,ownership w) const;
	void setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen = false);
	void resetCtsFunctions();
	void write(ostream & o) const;
	void deletepp() const;
  
  ~QfiedGoal()
	{
		delete pp;
	};

};

//class TimedGoal : public Proposition still to do.
//class SimpleGoal ?

class Validator;


class PropositionFactory {
private:
	map<string,const SimpleProposition *> literals;
    
	Validator * vld;

	struct buildProp {
		PropositionFactory * myPF;
		const Environment & myEnv;
		bool buildNewLiterals;
    const State * state;
    
		buildProp(PropositionFactory * pf,const Environment & e,bool b = true,const State * s = 0) :
			myPF(pf), myEnv(e), buildNewLiterals(b), state(s)
		{};
		const Proposition * operator()(const goal * g) 
		{
			return myPF->buildProposition(g,myEnv,buildNewLiterals,state);
		};
	};
			
public:
	PropositionFactory(Validator * v) : literals(), vld(v) {};


	~PropositionFactory()
	{
		for(map<string,const SimpleProposition *>::iterator i = literals.begin();
				i != literals.end();++i)
			delete (i->second);
	};
	
	const SimpleProposition * buildLiteral(const proposition * p)
	{
		string s(p->head->getName());
		for(parameter_symbol_list::const_iterator i = p->args->begin();
					i != p->args->end();++i)
		{
			//s += ".";
			s += (*i)->getName(); //(unsigned int)(*i);
		};
		map<string,const SimpleProposition*>::const_iterator i1 = literals.find(s);
		if(i1 != literals.end())
			return i1->second;
		const SimpleProposition * prp = literals[s] = new SimpleProposition(p);
		return prp;
	};

	const SimpleProposition * buildLiteral(const simple_effect * eff)
	{
		return buildLiteral(eff->prop);
	};

	const SimpleProposition * buildLiteral(const proposition * p,const Environment & bs)
	{
		string s(p->head->getName());
		for(parameter_symbol_list::const_iterator i = p->args->begin();
					i != p->args->end();++i)
		{

			if(dynamic_cast<const var_symbol*>(*i))
			{
/*			cout << "DEBUG: " << (**i) << "\n";
			for(Environment::const_iterator x = bs.begin();
				x != bs.end(); ++x)
				{
				cout << *(x->first) << " -> " << *(x->second) << "\n";
				}
*/
				s += bs.find(dynamic_cast<const var_symbol*>(*i))->second->getName();	
			}
			else
			{
				s += (*i)->getName();
			};
		
		};
	      
		map<string,const SimpleProposition*>::const_iterator i1 = literals.find(s);
		if(i1 != literals.end())
			{return i1->second;}
		const SimpleProposition * prp = literals[s] = new SimpleProposition(p,bs);
		return prp;
	};

	const SimpleProposition * buildLiteral(const simple_effect * eff,const Environment & bs)
	{
		return buildLiteral(eff->prop,bs);
	};

  //bool evaluate(const proposition * p,const Environment & bs,const State * state) const;
	const Proposition * buildProposition(const goal * g,const Environment &bs,bool buildNewLiterals = true,const State * state = 0);
	const Proposition * buildProposition(const goal * g,bool buildNewLiterals = true,const State * state = 0);
};

class AdvicePropositionConj;
class AdvicePropositionDisj;
class AdvicePropositionLiteral;
class AdvicePropositionDP;
class AdvicePropositionComp;

class APVisitor {

public:
	virtual ~APVisitor() {};
	virtual void visitAPConj(const AdvicePropositionConj *) {};
	virtual void visitAPDisj(const AdvicePropositionDisj *) {};
	virtual void visitAPLiteral(const AdvicePropositionLiteral *) {};
	virtual void visitAPDP(const AdvicePropositionDP *) {};
	virtual void visitAPComp(const AdvicePropositionComp *) {};
};

class AdviceProposition{
private:
   
public:


    AdviceProposition() {};
    virtual ~AdviceProposition() {};
    virtual bool isAdvice() const =0;
    virtual void display(int indent = 0) const =0;
    virtual void displayLaTeX(int depth = 0) const =0;
    virtual void visit(APVisitor * apv) const {};
};

class AdvicePropositionConj : public AdviceProposition{
private:
    vector<const AdviceProposition *> adviceProps;
public:

    AdvicePropositionConj() : adviceProps() {};
    ~AdvicePropositionConj();
    bool isAdvice() const {return (adviceProps.size() > 0);};
    void addAdviceProp(const AdviceProposition * ap) {adviceProps.push_back(ap);};
    void display(int indent = 0) const;
    void displayLaTeX(int depth = 0) const;
	void visitAll(APVisitor * apv) const
	{
		for(vector<const AdviceProposition *>::const_iterator i = adviceProps.begin();
				i != adviceProps.end();++i)
		{
			(*i)->visit(apv);
		};
	};
    void visit(APVisitor * apv) const {apv->visitAPConj(this);};
};

class FastEnvironment;

class AdvicePropositionDisj : public AdviceProposition{
private:
    vector<const AdviceProposition *> adviceProps;
public:

    AdvicePropositionDisj() : adviceProps() {};
    ~AdvicePropositionDisj();
    bool isAdvice() const {return (adviceProps.size() > 0);};
    void addAdviceProp(const AdviceProposition * ap) {adviceProps.push_back(ap);};
    void display(int indent = 0) const;
    void displayLaTeX(int depth = 0) const;

	void visitAll(APVisitor * apv) const
	{
		for(vector<const AdviceProposition *>::const_iterator i = adviceProps.begin();
				i != adviceProps.end();++i)
		{
			(*i)->visit(apv);
		};
	};
    void visit(APVisitor * apv) const {apv->visitAPDisj(this);};
};

class AdvicePropositionLiteral : public AdviceProposition{
private:
    bool thereIsAdvice;
    const SimpleProposition * sp;
    bool advice;
public:


    AdvicePropositionLiteral(bool isAd, const SimpleProposition * sp, bool a) : thereIsAdvice(isAd), sp(sp), advice(a) {};
    ~AdvicePropositionLiteral() {}; //do not delete sp!
    bool isAdvice() const {return thereIsAdvice;};
    void changeAdvice(bool isAd, const SimpleProposition * sprop, bool a) {thereIsAdvice = isAd; sp = sprop; advice =a; };
    void display(int indent = 0) const;
    void displayLaTeX(int depth = 0) const;

	const SimpleProposition * getProp() const {return sp;};

    void visit(APVisitor * apv) const {apv->visitAPLiteral(this);};
};

class AdvicePropositionDP : public AdviceProposition{
private:
    const DerivedGoal * dp;
    bool neg;
public:

    AdvicePropositionDP(const DerivedGoal * p, bool n) : dp(p), neg(n) {};
    ~AdvicePropositionDP() {}; //do not delete dp!
    bool isAdvice() const {return false;};
    void display(int indent = 0) const;
    void displayLaTeX(int depth = 0) const;
	bool isNeg() const {return neg;};
	const DerivedGoal * getDG() const {return dp;};

  void visit(APVisitor * apv) const {apv->visitAPDP(this);};
};




class AdvicePropositionComp : public AdviceProposition{
private:
    bool thereIsAdvice;
    const Comparison * comp;
    string advice;
    bool neg;
public:

    AdvicePropositionComp(bool isAd, const Comparison * c, string a, bool n) : thereIsAdvice(isAd), comp(c), advice(a), neg(n) {};
    ~AdvicePropositionComp() {}; //do not delete comp!
    bool isAdvice() const {return thereIsAdvice;};
    void display(int indent = 0) const;
    void displayLaTeX(int depth = 0) const;
	bool isNeg() const {return neg;};
	const Comparison * getComp() const {return comp;};
  void visit(APVisitor * apv) const {apv->visitAPComp(this);};
};

};

#endif
