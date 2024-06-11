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

  $Date: 2009-02-05 10:50:21 $
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
#include <vector>
#include <functional>
#include <algorithm>
#include <map>
#include <iostream>
#include "ptree.h"
#include "Exceptions.h"
#include "main.h"
#include "Ownership.h"
#include "Polynomial.h"
#include "State.h"

#ifndef __PLAN
#define __PLAN

// Use the following switch if your compiler/STL doesn't use std.
// #define NO_STD_NAMESPACE

//#define vector std::vector

namespace VAL {
  
class State;
class Proposition;
class FuncExp;
class FuncExpFactory;
class Action;
class InvariantAction;
class CtsEffectAction;
class CondCommunicationAction;
struct ExecutionContext;


class Update {
private:
	const FuncExp * fe;
	assign_op aop;
	FEScalar value;
public:
	Update(const FuncExp * f,assign_op ao,FEScalar v) :
		fe(f), aop(ao), value(v)
	{};

	void update(State * s) const;
   void updateChange(State * s) const;
};

class EffectsRecord {
private:
	vector<const SimpleProposition *> adds;
	vector<const SimpleProposition *> dels;
	map<const SimpleProposition *,set<const Action *> > responsibleForProps;

	vector<Update> updates;
	map<const FuncExp *,set<const Action *> > responsibleForPNEs;

public:
	void pushAdd(const SimpleProposition * p,const Action * act)
	{
		adds.push_back(p);
		if(act) responsibleForProps[p].insert(act);
	};
	void pushDel(const SimpleProposition * p,const Action * act)
	{
		dels.push_back(p);
		if(act) responsibleForProps[p].insert(act);
	};
	void addFEffect(const FuncExp * fe,assign_op aop,FEScalar value,const Action * act)
	{
		updates.push_back(Update(fe,aop,value));
		if(act) responsibleForPNEs[fe].insert(act);

	};

	void enact(State * s) const;
};

class Happening {
private:
	Validator * vld;

	double time;
	vector<const Action*> actions;

	Happening(Validator * v) :
		vld(v), time(0.0), actions(), eventHappening(false),realHappening(false), afterPlan(false)
	{};

   bool eventHappening;
	bool realHappening;
	bool afterPlan;

public:
	friend struct ExecutionContext;
	friend struct ActiveCtsEffects;

private:
	template<typename X> struct select2nd {
	  typename X::second_type operator()(X p){return p.second;};
	};


public:
	Happening(Validator * v,const vector<pair<double,Action*> > & as,double timeEndPlan);
	Happening(Validator * v,double timeToExecute,const vector<pair<double,Action*> > & as);

  Happening(Validator * v,vector<const Action*>  acts,bool event = false);   //for creating event happenings
  Happening(Validator * v, vector<const Action*> acts,double t,bool event =false);
	~Happening();

	void adjustContext(ExecutionContext &) const;
	void adjustContextInvariants(ExecutionContext &) const; //invariant interval is ( , ] or ( , ) ?
	void adjustActiveCtsEffects(ActiveCtsEffects &) const;

	double getTime() const {return time;};
	int getNoActions() const {return actions.size();};
	const vector<const Action*> * getActions() const {return &actions;};

  void clearActions() {actions.clear();};
	bool canHappen(const State * s) const;
	bool applyTo(State * s) const;


	void write(ostream & o) const;

	bool isAfterPlan() const {return afterPlan;};
   bool isRegularHappening() const {return realHappening;};
   void inject(Action * a) {actions.push_back(a);};
};

struct ExecutionContext {
	Happening invariants;	// Also includes the temporal conditional effects monitors.

	ExecutionContext(Validator * v);

	void addInvariant(const InvariantAction * a);
	void removeInvariant(const InvariantAction * a);
	void addCondAction(const CondCommunicationAction * ca);
	bool removeCondAction(const CondCommunicationAction * ca);
	void setTime(double t);
	void setActiveCtsEffects(ActiveCtsEffects * ace);

	const Happening * getInvariants() const {return &invariants;};
	bool hasInvariants() const;
	~ExecutionContext();

};


typedef pair<pair<const expression *,bool>, const Environment *> ExprnPair; //bool is true if increasing

struct ActiveFE {

	const FuncExp * fe;
	vector<const ActiveFE*> parentFEs;	//an active FE that this FE depends on, there may be 0 or many

	int colour;		//for topological sort

	vector<ExprnPair> exprns;  

	const CtsFunction * ctsFtn; 				//cts fn defining FEs values on interval it is changing on

	FEScalar evaluate(double time) const; 	//time since start of active interval;
	bool isLinear() const;

	ActiveFE(const FuncExp * f) : fe(f),ctsFtn(0) {};

	void addParentFE(const ActiveFE * a);
	void removeParentFE(const ActiveFE * a);
	void addParentFEs(const ActiveCtsEffects * ace,const expression * e,const Environment * bs);
   bool appearsInEprsn(const ActiveCtsEffects * ace,const expression * e,const Environment * bs) const;
   bool canResolveToExp(const map<const FuncExp*,ActiveFE*> activeFEs,Validator *) const;

	~ActiveFE();
};


bool isConstLinearChangeExpr(const ExprnPair & exp,const map<const FuncExp *,ActiveFE *> activeFEs,Validator * vld);
bool isConstant(const expression * exp,const Environment * env,const map<const FuncExp *,ActiveFE *> activeFEs,Validator * vld);
const expression* getRateExpression(const expression* aExpression);

//the following class contains all the active cts effects that are to be updated before each regular happening at
//the same point in time
struct ActiveCtsEffects {
	Happening ctsEffects;				// contains all the active cts effects, length of time is given from the last happening
	map<const FuncExp *, ActiveFE*> activeFEs;  // contains all the active FEs and their dependencies on other FEs, also how to update the FEs
	bool ctsEffectsProcessed;
   mutable Happening ctsUpdateHappening;

	Validator * vld;
	double localUpdateTime; //time since last happening
   double eventTime;

	ActiveCtsEffects(Validator * v);

	void addCtsEffect(const CtsEffectAction * a);
	void removeCtsEffect(const CtsEffectAction * a);

	void setTime(double t);
   void setEventTime(double t) {eventTime =t;};
   double getEventTime() const {return eventTime;};
	void setLocalUpdateTime(double ht);
	void addActiveFEs(bool reCalc = false);
	void addActiveFE(assignment * e,const Environment & bs);
	void buildAFECtsFtns();
	void visitActiveFE(ActiveFE * afe,vector<ActiveFE*> & topSAFEs); //for topological sort
	const Polynomial * buildPoly(const ActiveFE * afe);
	const CtsFunction * buildExp(const ActiveFE * afe);
	const CtsFunction * buildNumericalSoln(const ActiveFE * afe);
	const Happening * getCtsEffects() const {return &ctsEffects;};
	const Happening * getCtsEffectUpdate() const;

	bool hasCtsEffects() const;
  void clearCtsEffects();
	bool areCtsEffectsLinear() const;
	bool isFEactive(const FuncExp * fe) const;

	~ActiveCtsEffects();
};

struct after {
	double time;
	double tolerance;

	after(double t,double tol) : time(t), tolerance(tol) {};



	bool operator()(const pair<double,Action*> p) const
	{
		return p.first > time + tolerance/10;
	};
};

struct sameTime {
	double time;

	sameTime(double t) : time(t) {};

	bool operator()(const pair<double,Action*> p) const
	{
		return p.first > time;
	};
};

class Plan {
public:
	typedef vector<pair<double,Action*> > timedActionSeq;
// Using a list allows us to add Happenings to the end of the plan without
// invalidating the iterators.
	typedef list<Happening *> HappeningSeq;

private:
	HappeningSeq happenings;
	Validator * vld;
	double timeToProduce;

	struct planBuilder {
		Validator * vld;

		timedActionSeq & tas;
		const operator_list * ops;
		double defaultTime;
		timedActionSeq extras;

		planBuilder(Validator * v,timedActionSeq & ps,
						const operator_list * os) :
			vld(v), tas(ps), ops(os), defaultTime(1), extras() {};

		void handleDurativeAction(const durative_action *,const const_symbol_list *,double,double,const plan_step * ps);
		void operator()(const plan_step * ps);

	};

public:
	Validator* getValidator() const {return vld;};
	HappeningSeq::const_iterator getFirstHappening() const {return happenings.begin();};
	HappeningSeq::const_iterator getEndHappening() const {return happenings.end();};

	Plan(Validator* v,const operator_list * ops,const plan * p);
	~Plan()
	{
		for(HappeningSeq::const_iterator i = happenings.begin();
				i != happenings.end();++i)
		{
			delete (*i);
		};

	};

	Happening * lastHappening() const
	{
		if(happenings.size()==0) return 0;

		return happenings.back();

	};


	class const_iterator;


	friend class const_iterator;

	class const_iterator : public 
#ifndef OLDCOMPILER
		std::iterator
#endif
#ifdef OLDCOMPILER
		std::forward_iterator
#endif
				<std::input_iterator_tag,const Happening *>
	{
	private:
		int pos;
		const Plan * myPlan;
		double currenttime;		// currenttime will always be the time of the state prevailing.
								//(Last distinct time! May be different type of happening happenings with same time)

		ExecutionContext ec;	// Records invariant checks.
		ActiveCtsEffects ace;   //Records all the active cts effects

		enum HappeningType {INVARIANT, CTS, REGULAR, END }; //The different types of happening to be executed in that order
													//at each regular happening timestamp if invariants and cts effects exist

		HappeningType executeHappening;

		HappeningSeq::const_iterator i;
								// The iterator always points at the next happening to be considered
								// (if there is one), ignoring possible invariant checks.

	public:
		const_iterator(const Plan * p) : pos(0),
			myPlan(p), currenttime(0.0), ec(p->getValidator()), ace(p->getValidator()), executeHappening(REGULAR), i(p->getFirstHappening())
		{
			if(i != p->getEndHappening())
			{
				(*i)->adjustContext(ec);
				(*i)->adjustActiveCtsEffects(ace);

			};
		};

	int operator-(const const_iterator & x)
	{
		return pos - x.pos;
	};

		double getTime()
		{
			if(executeHappening == REGULAR)
				return (*i)->getTime();

			else
			{
				HappeningSeq::const_iterator j = i;
				++j;
				return (*j)->getTime();
			};


		};


       void deleteActiveFEs()

       {        
           for(map<const FuncExp *, ActiveFE*>::iterator j = ace.activeFEs.begin(); j != ace.activeFEs.end(); ++j)
            {    //cout << " deleting"<< *(j->second->fe) <<"\n";
                delete j->second;
            };

                    
	          ace.activeFEs.clear();
       };

		bool isRegular() const
		{
			return (executeHappening == REGULAR);
		};

      ActiveCtsEffects * getActiveCtsEffects() {return &ace;};
      ExecutionContext * getExecutionContext() {return &ec;};
      const ActiveCtsEffects * getActiveCtsEffects() const {return &ace;};
      const ExecutionContext * getExecutionContext() const {return &ec;};
      
		bool isInvariant() const
		{
			return (executeHappening == INVARIANT);
		};

		void toEnd()
		{
			i = myPlan->getEndHappening();
			currenttime = 0;
			executeHappening = END;

		};




		bool operator ==(const const_iterator & c) const
		{
		  	return currenttime == c.currenttime && executeHappening == c.executeHappening;

		};



		bool operator !=(const const_iterator & c) const
		{
		    return !(operator==(c));
		};

		const Happening * operator*() const
		{

			switch(executeHappening) {
			  case INVARIANT:

			    return ec.getInvariants();
			    break;
			  case CTS:     
			    return ace.getCtsEffectUpdate();
			    break;
			  case REGULAR:
			    if(i != myPlan->getEndHappening())
			    return *i;
			  default:
			  	break;
		     }

		  return 0;
		};

		const_iterator & operator++()
		{

			HappeningSeq::const_iterator j = i;
			++j;
			if((j) == myPlan->getEndHappening())
				{
					++i;
					++pos;
					currenttime = 0;
					executeHappening = END; //set value so we know the plan has finished
					return *this;
				};
           
			currenttime = (*(j))->getTime();//

			//value of executeHappening represents the last type of happening to be executed
			 switch(executeHappening) {
				 case INVARIANT:

                    
					if(ace.hasCtsEffects())
					{       
						handleCtsHappening();
						return *this;
					};

					executeHappening = REGULAR;

				    break;

				 case CTS:
                 
				    executeHappening = REGULAR;

				    break;

				 case REGULAR:
                 
					if(ec.hasInvariants())
					{
					    handleInvHappening();
					    (*i)->adjustContextInvariants(ec);
					    (*(j))->adjustContextInvariants(ec);
					    return *this;
					}
					else if(ace.hasCtsEffects())
					{    
					    handleCtsHappening();
					    return *this;
					};


				 default:

			  		break;

			   };//end of happening type switch

			    //deal with regular happenings here
				++i;

				(*i)->adjustContext(ec);
				(*i)->adjustActiveCtsEffects(ace);

				return *this;

		};

		void handleInvHappening()
		{
			 executeHappening = INVARIANT;
			 ec.setActiveCtsEffects(&ace); //cout << "inv current time = "<<currenttime<<"\n";
			 ec.setTime(currenttime); //same time as next regular happening
       if(ace.getEventTime() > (*i)->getTime()) ace.setLocalUpdateTime(currenttime - ace.getEventTime());
       else ace.setLocalUpdateTime(currenttime - (*i)->getTime()); 

		};

		void handleCtsHappening()
		{
             
			 executeHappening = CTS;
		     ace.setTime(currenttime); //same time as next regular happening
		     ace.setLocalUpdateTime(currenttime - (*i)->getTime());  //set value of LocalUpdateTime - time since last regular happening
         if(ace.getEventTime() > (*i)->getTime()) ace.setLocalUpdateTime(currenttime - ace.getEventTime());
         else ace.setLocalUpdateTime(currenttime - (*i)->getTime());  //set value of LocalUpdateTime - time since last regular happening
    
           
		};



		const_iterator operator++(int)
		{
			const_iterator ii = *this;
			++(*this);
			return ii;
		};

	};

	const_iterator begin() const
	{
		return const_iterator(this);
	};


	const_iterator end() const
	{
		const_iterator c(this);
		c.toEnd();
		return c;
	};

	void display() const;
	int length() const;
	double getTime() const {return timeToProduce;};
	double timeOf(const Action * a) const;
	void show(ostream & o) const;
	void addHappening(Happening * h);
};

ostream & operator <<(ostream & o,const Plan & p);
ostream & operator <<(ostream & o,const Happening * h);
inline ostream & operator <<(ostream & o,const Happening & h)
{
	h.write(o);
	return o;
};

void insert_effects(effect_lists * el,effect_lists * more);
Polynomial getPoly(const expression * e,const ActiveCtsEffects * ace,const Environment & bs,CoScalar endInt = 0);
Polynomial getPoly(const expression * e,const ActiveCtsEffects * ace,const Environment * bs,CoScalar endInt = 0);
Polynomial getPoly(const expression * e,bool inc,const ActiveCtsEffects * ace,const Environment & bs,CoScalar endInt = 0);
Polynomial getPoly(const expression * e,bool inc,const ActiveCtsEffects * ace,const Environment * bs,CoScalar endInt = 0);

};

#endif
