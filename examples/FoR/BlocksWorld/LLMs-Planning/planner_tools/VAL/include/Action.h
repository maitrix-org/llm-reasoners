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

  $Date: 2009-02-05 10:50:10 $
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
#include "Proposition.h"
#include "main.h"
#include <iostream>
#include <algorithm>
#include <iterator>
#include <string>
#include <set>

using std::ostream_iterator;
using std::set;

#ifndef __ACTION
#define __ACTION

namespace VAL {
  
class State;
class Ownership;
class EffectsRecord;
class Validator;
struct ExecutionContext;
struct ActiveCtsEffects;
class StartAction;
class EndAction;

struct safeaction : public action {

	safeaction(operator_symbol* nm,
	    var_symbol_list* ps,
	    goal* pre,
	    effect_lists* effs,
	    var_symbol_table* st) : action(nm,ps,pre,effs,st) {};

	~safeaction() 
	{
		conj_goal * cg = dynamic_cast<conj_goal*>(precondition);
		if(cg) const_cast<goal_list *>(cg->getGoals())->clear();
						// Mustn't delete the preconditions.
		symtab = 0;		// Mustn't delete the symbol table either.
		parameters = 0;	// Or the parameters.
						// Finally, we don't own the effects, so mustn't 
						// clobber those either.
		effects->add_effects.clear();
		effects->del_effects.clear();
		effects->forall_effects.clear();
		effects->cond_effects.clear();
		effects->assign_effects.clear();			
	};
};	


class Action {
protected:
		const operator_ * act;
		Environment bindings;

		bool timedInitialLiteral;

		Validator * vld;
		
		const Proposition * pre;
    string actionName;
    const plan_step * planStep;
    
	bool handleEffects(Ownership & o,EffectsRecord & e,
							const State * s,const effect_lists * effs,const Environment & env,bool markPreCons) const;
	bool handleEffects(Ownership & o,EffectsRecord & e,
							const State * s,const effect_lists * effs,bool markPreCons) const;

		struct ActionParametersOutput {

			const Environment & bindings;

			ActionParametersOutput(const Environment & bs) : bindings(bs) {};
			string operator()(const var_symbol * v) const
			{
				return bindings.find(v)->second->getName();
			};
		};

	friend struct FAEhandler;
	friend struct ActiveCtsEffects;
	
public:
	Action(Validator * v,const operator_ * a,const const_symbol_list* bs);
	Action(Validator * v,const operator_ * a,Environment * bs);
	Action(Validator * v,const operator_ * a,const vector<const_symbol *> & vs);
  Action(Validator * v,const operator_ * a,const const_symbol_list* bs,const plan_step * ps);
	
	virtual ~Action();

	const effect_lists * getEffects() const {return act->effects;};
	const Environment & getBindings() const {return bindings;};
	const operator_ * getAction() const {return act;};
  const plan_step * getPlanStep() const {return planStep;};
    const Proposition * getPrecondition() const {return pre;};
	string getName() const;
	string getName0() const;
	virtual void displayDurationAdvice(const State * s) const {};
	virtual void displayEventInfomation() const;                                                         
	virtual bool confirmPrecondition(const State * s) const;
   virtual void addErrorRecord(double t,const State * s) const;
	virtual void markOwnedPreconditions(Ownership & o) const;	
	virtual bool constructEffects(Ownership & o,EffectsRecord & e,const State * s,bool markPreCons) const;
	virtual void adjustContext(ExecutionContext &) const;
	virtual void adjustContextInvariants(ExecutionContext &) const;
	virtual void adjustActiveCtsEffects(ActiveCtsEffects &) const;
	virtual void addTriggeredEvents(vector<const Action *> & triggeredEvents,vector<const Action *> & oldTriggeredEvents, vector<const StartAction *> & triggeredStartProcesses, vector<const EndAction *> & triggeredEndProcesses) const;
	virtual void write(ostream & o) const
	{
		o << getName();
	};

	virtual bool isRealAction() const {return !timedInitialLiteral;};
	virtual bool isRegAction() const {return true;};

  	virtual const Action * startOfAction() const {return this;};
	bool operator==(const plan_step & ps) const;
};

class InvariantAction : public Action {
private:
	ActiveCtsEffects * ace;
	StartAction * start;
	mutable bool rhsIntervalOpen;//only open for last interval that invariant is checked on
public:	
	InvariantAction(Validator * v,StartAction * sa,const action * a,const const_symbol_list* bs,const plan_step * ps = 0) :
		Action(v,a,bs,ps), start(sa), rhsIntervalOpen(false)
	{};
  
	~InvariantAction();
	bool confirmPrecondition(const State * s) const;
   void addErrorRecord(double t,const State * s) const;
	void setActiveCtsEffects(ActiveCtsEffects * a) {ace = a;};
	void setRhsIntervalOpen(bool rhs) const {rhsIntervalOpen = rhs;};
	bool isRealAction() const {return false;};
	bool isRegAction() const {return false;};
  
	void write(ostream & o) const
	{
		if(LaTeX)
		{
			o << "\\actioninv{";
			Action::write(o);
			o << "}";
		}
		else
		{
			o << "Invariant for ";
			Action::write(o);
		};
	};

	const Action * partner() const;

	const Action * startOfAction() const {return (const Action *) start;};
};





	

class CondCommunicationAction : public Action {
private:
	mutable bool status;
	ActiveCtsEffects * ace;
	mutable bool rhsIntervalOpen;
	StartAction * start;
  
	conj_goal * gls;
	const Proposition * initPre;
	conj_goal * gli;
	const Proposition * invPre;

	effect_lists * els;


public:
	CondCommunicationAction(Validator * v,const durative_action * a,const const_symbol_list * bs,
		goal_list * gs,goal_list * gi,goal_list * ge,
		effect_lists * es,effect_lists * el);
	CondCommunicationAction(Validator * v,const durative_action * a,const const_symbol_list * bs,
		goal_list * gs,goal_list * gi,goal_list * ge,
		effect_lists * es,effect_lists * el,Environment * vs);
	~CondCommunicationAction();

	void write(ostream & o) const 

	{
		
		if(LaTeX)
		{
			o << "\\condeffmon{";
			Action::write(o);
			o << "}";
		}
		else
		{
			Action::write(o);
			o << " - conditional effect monitor";
		};
			
	};

	void markInitialPreconditions(Ownership & o) const;


	void markOwnedPreconditions(Ownership & o) const;
	bool confirmPrecondition(const State * s) const;
	bool constructEffects(Ownership & o,EffectsRecord & e,const State * s,bool markPreCons) const;
	void setActiveCtsEffects(ActiveCtsEffects * a) {ace = a;};
	void setRhsIntervalOpen(bool rhs) const {rhsIntervalOpen = rhs;};
	bool confirmInitialPrecondition(const State * s) const;
	bool constructFinalEffects(Ownership & o,EffectsRecord & e,const State * s) const;
	bool isActive() const {return status;};
   bool isRealAction() const {return false;};
	bool isRegAction() const {return false;};
	const Action * partner() const;
	const Action * startOfAction() const {return (const Action *) start;};
};

void buildForAllCondActions(Validator * vld,const durative_action * da,
	const const_symbol_list * params,goal_list * gls,
		goal_list * gli,goal_list * gle,effect_lists * locels,
		effect_lists * locele,const var_symbol_list * vars,
		var_symbol_list::const_iterator i,
		vector<const CondCommunicationAction *> & condActions,
		Environment * env);
		
class CtsEffectAction : public Action {
private:
	ActiveCtsEffects * ace;
	StartAction * start;
public:
	CtsEffectAction(Validator * v,const action * a,const const_symbol_list* bs,const vector<const CondCommunicationAction*> & cas) :
		Action(v,a,bs), condActions(cas)
	{};	

	~CtsEffectAction();

	const vector<const CondCommunicationAction *> condActions;
	
	bool constructEffects(Ownership & o,EffectsRecord & e,const State * s,bool markPreCons) const;
	void setHasht(double ht);
	void setActiveCtsEffects(ActiveCtsEffects * a) {ace = a;};
	void displayCtsFtns() const;
	bool isRealAction() const {return false;};
	bool isRegAction() const {return false;};
  
	void write(ostream & o) const
	{
		
		if(LaTeX)
		{

			o << "\\updatectspne";
		}
		else
			o << "Update of continuously changing Primitive Numerical Expressions";
	};
	const Action * partner() const;
	const Action * startOfAction() const {return (const Action *) start;};
};

class DurativeActionElement : public Action {
protected:
	double duration;
	const goal_list * durs;

	const InvariantAction * invariant;
	const CtsEffectAction * ctsEffects;
	
	const vector<const CondCommunicationAction *> condActions;
	
public:
	DurativeActionElement(Validator * v,const action * a,const const_symbol_list* bs,
					double d,const goal_list * ds,const InvariantAction * inv,
			                const CtsEffectAction * ctsEff,
					const vector<const CondCommunicationAction *> & cas,const plan_step * ps = 0) :
		Action(v,a,bs,ps), 
		duration(d), durs(ds), invariant(inv), ctsEffects(ctsEff), condActions(cas)
	{
		bindings.duration = duration;
	};
	virtual ~DurativeActionElement();

	void markOwnedPreconditions(Ownership &) const;
	bool confirmPrecondition(const State * s) const;
	double getDuration() const {return duration;};
	bool isRegAction() const {return true;};
	
};

class EndAction;

class StartAction : public DurativeActionElement {
private:
	mutable EndAction * otherEnd;
public:
	friend class EndAction;
	
	StartAction(Validator * v,const action * a,const const_symbol_list* bs,
					const conj_goal * inv, effect_lists * elc, double d,const goal_list * ds,
					const vector<const CondCommunicationAction*> & cas,
					const vector<const CondCommunicationAction*> & ccas,const plan_step * ps = 0) :
		   DurativeActionElement(v,
								 a,
								 bs,
								 d,
								 ds,
								 (inv->getGoals()->empty())?0:	//do not create invariant action if no invariants to check!							 				
								 new InvariantAction(v,this,
													 new safeaction(a->name,
																	a->parameters,
																	const_cast<conj_goal *>(inv),
																	new effect_lists(),
																	a->symtab),
													 bs,ps),
								(elc->assign_effects.empty() && elc->forall_effects.empty() && ccas.empty() )?0: //similarly do not create cts effect action if no cts effects!
								new CtsEffectAction(v,
													new safeaction(a->name,
																   a->parameters,
																   new conj_goal(new goal_list()), //no preconditions
																   elc,
																   a->symtab),
								  					bs,
								  					ccas),
				      			cas,ps)
	{
   if(inv->getGoals()->empty()) delete inv;
   if(elc->assign_effects.empty() && elc->forall_effects.empty() && ccas.empty()) delete elc;
  };


	~StartAction() 
	{
		delete invariant;
		delete ctsEffects;
	};
	
	void adjustContext(ExecutionContext &) const;
	void adjustContextInvariants(ExecutionContext &) const;
	void adjustActiveCtsEffects(ActiveCtsEffects &) const;
	void markOwnedPreconditions(Ownership & o) const;
	bool confirmPrecondition(const State *) const;
	void displayDurationAdvice(const State * s) const;
	void addTriggeredEvents(vector<const Action *> & triggeredEvents,vector<const Action *> & oldTriggeredEvents, vector<const StartAction *> & triggeredStartProcesses, vector<const EndAction *> & triggeredEndProcesses) const;
  void displayEventInfomation() const;
	void write(ostream & o) const
	{
		
		if(LaTeX)
		{
			o << "\\actionstart{";
			Action::write(o);
			o << "}";
		}
		else
		{
			Action::write(o);
			o << " - start";
		};
	};
	const Action * partner() const;
	const Action * starter() const {return this;};
};

class EndAction : public DurativeActionElement {
private:
	const StartAction * otherEnd;
public:
	EndAction(Validator * v,const action * a,const const_symbol_list* bs,
				const StartAction * sa,double d, const goal_list * ds,const plan_step * ps = 0) :
		DurativeActionElement(v,a,bs,d,ds,sa->invariant,sa->ctsEffects, sa->condActions,ps),
		otherEnd(sa)
	{
		sa->otherEnd = this;
	};
	~EndAction()
	{};

	void adjustContext(ExecutionContext &) const;

	void adjustContextInvariants(ExecutionContext &) const;
	void adjustActiveCtsEffects(ActiveCtsEffects &) const;
	bool constructEffects(Ownership & o,EffectsRecord & e,const State * s,bool markPreCons) const;
	const Action * partner() const {return otherEnd;};
	void addTriggeredEvents(vector<const Action *> & triggeredEvents,vector<const Action *> & oldTriggeredEvents,
  vector<const StartAction *> & triggeredStartProcesses, vector<const EndAction *> & triggeredEndProcesses) const;
  void displayEventInfomation() const;
	void write(ostream & o) const
	{
		if(LaTeX)
		{
			o << "\\actionend{";
			Action::write(o);
			o << "}";
		}
		else
		{
			Action::write(o);
			o << " - end";
		};
	};
	const Action * startOfAction() const {return otherEnd;};

};

	
ostream & operator <<(ostream & o,const Action & a);
ostream & operator << (ostream & o, const Action * const a);

template<typename T>
const Environment buildBindings(const operator_ * a,const T & bs)
{
	Environment bindings;
	typename T::const_iterator j = bs.begin();
	for(var_symbol_list::iterator i = a->parameters->begin();
			i != a->parameters->end();++i,++j)
	{
		bindings[*i] = *j;
	}; 
	return bindings;
};


};

#endif
















