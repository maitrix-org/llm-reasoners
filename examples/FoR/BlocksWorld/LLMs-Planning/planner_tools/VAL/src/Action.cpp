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
#include "State.h"
#include "Action.h"
#include "Plan.h"
#include "main.h"
#include "Validator.h"
#include "Ownership.h"
#include "Utils.h"
#include "Proposition.h"
#include "LaTeXSupport.h"
#include "RobustAnalyse.h"
#include <string>
#include <cmath>


//#define vector std::vector
//#define list std::list
//#define map std::map

using std::map;
using std::list;
using std::vector;
using std::for_each;

namespace VAL {
  
Action::~Action()
{
	pre->destroy();
};

bool Action::operator==(const plan_step & ps) const
{
	if(act->name != ps.op_sym) return false;

	var_symbol_list::const_iterator i = act->parameters->begin();
	for(const_symbol_list::const_iterator j = ps.params->begin();
					j != ps.params->end();++j,++i)
	{
		if(bindings.find(*i)->second != *j)



			return false;

	};
	return true;
};

string Action::getName() const
{
	string n;
	
	if(LaTeX) n = "\\action{";
	
	n += "(" + act->name->getName();

	for( var_symbol_list::const_iterator i = act->parameters->begin() ; i != act->parameters->end(); ++i)
	{
		n += " " +  bindings.find(*i)->second->getName();

	};
	
	n += ")";
	
	if(LaTeX)
	{
		n += "}";
		latexString(n);
	};
	
	return n;
};

//minimal action name for internal use
string Action::getName0() const
{
  return actionName;
	/*string n = act->name->getName();

	for( var_symbol_list::const_iterator i = act->parameters->begin() ; i != act->parameters->end(); ++i)
	{
		n += bindings.find(*i)->second->getName();
	};

	return n; */
};

InvariantAction::~InvariantAction()
{

	delete act;
};

CtsEffectAction::~CtsEffectAction()
{
	delete act;
};

DurativeActionElement::~DurativeActionElement()
{
	delete act;
	const_cast<goal_list *>(durs)->clear();	// We don't own those expressions, so don't delete them!
	delete durs;
};

bool Action::confirmPrecondition(const State * s) const
{       
	bool ans = pre->evaluate(s);
	if(LaTeX && !ans) *report << " \\notOK \\\\\n \\> ";
	return ans;
};

bool InvariantAction::confirmPrecondition(const State * s) const
{

	if(TestingPNERobustness) ace->addActiveFEs(true);
  else ace->addActiveFEs();
  	const_cast<Proposition*>(pre)->setUpComparisons(ace,rhsIntervalOpen);
	DerivedGoal::setACE(ace,rhsIntervalOpen);
	bool ans = pre->evaluate(s);
	DerivedGoal::setACE(0,rhsIntervalOpen);
	if(LaTeX && !ans) *report << " \\notOK \\\\\n \\> ";

	return ans;

};


void Action::addTriggeredEvents(vector<const Action *> & triggeredEvents,vector<const Action *> & oldTriggeredEvents, vector<const StartAction *> & triggeredStartProcesses, vector<const EndAction *> & triggeredEndProcesses) const
{     
  triggeredEvents.push_back(this);
  oldTriggeredEvents.push_back(this);
};

void EndAction::addTriggeredEvents(vector<const Action *> & triggeredEvents,vector<const Action *> & oldTriggeredEvents, vector<const StartAction *> & triggeredStartProcesses, vector<const EndAction *> & triggeredEndProcesses) const
{    
  oldTriggeredEvents.push_back(this);
  triggeredEndProcesses.push_back(this);
};

void StartAction::addTriggeredEvents(vector<const Action *> & triggeredEvents,vector<const Action *> & oldTriggeredEvents, vector<const StartAction *> & triggeredStartProcesses, vector<const EndAction *> & triggeredEndProcesses) const
{       
  oldTriggeredEvents.push_back(this);
  triggeredStartProcesses.push_back(this);

};



void Action::addErrorRecord(double t,const State * s) const
{
    vld->getErrorLog().addPrecondition(t,this,s); 
};

void InvariantAction::addErrorRecord(double t,const State * s) const
{
    try
    {                              
      vld->getErrorLog().addUnsatInvariant(t - pre->getEndOfInterval(), t, pre->getIntervals(s),this,s);
    }
    catch(PolyRootError & prError)
    {
      vld->getErrorLog().addUnsatInvariant(t - pre->getEndOfInterval(), t, Intervals(),this,s,true);

    };
};

bool 
DurativeActionElement::confirmPrecondition(const State * s) const
{
  double testDuration = duration;
  double testTolerance = s->getTolerance();
  
  if(Robust) //test the duration constraint according to the original duration given in the plan
  {
    testDuration = planStep->originalDuration; 
    testTolerance = 0.001;// 0.00000001; //1e-08, to within acceptable limits for calculations, should be as small as possible
  };
  
	for(goal_list::const_iterator i = durs->begin();i != durs->end();++i)
	{
		const comparison * c = dynamic_cast<const comparison *>(*i);
		double d = s->evaluate(c->getRHS(),bindings);
		bool test = true;
		switch(c->getOp())
		{
			case E_GREATER:
				test = (testDuration > d);
				break;
			case E_LESS:
				test = (testDuration < d);
				break;
			case E_GREATEQ:

				test = (testTolerance >= d - testDuration);
				if(!test && Verbose)
				{
					if(LaTeX) *report << "\\notOK \\\\\n \\> ";
					*report << "Tolerance of " << d-testDuration 
							<< " required for " << this << "\n";
				};
				break;
			case E_LESSEQ:
				test = (testDuration - d <= testTolerance);
				if(!test && Verbose)
				{
					if(LaTeX) *report << "\\notOK \\\\\n \\> ";
					*report << "Tolerance of " << testDuration - d 
							<< " required for " << this << "\n";
				};
				break;
			case E_EQUALS:
				test = (testDuration > d?testDuration - d:d - testDuration) < testTolerance;
            
				if(!test && Verbose)
				{
					if(LaTeX) *report << " \\notOK \\\\\n \\> ";
					*report << "Tolerance of " << ((testDuration > d)?testDuration -d:d-testDuration) 
							<< " required for " << this << "\n";
				};
				break;
			default:
				break;
		};
		if(!test)
		{
			if(Verbose)
			{
				if(LaTeX) *report << "\\\\\n \\> ";
				*report << "Failed duration constraint in " << this << "\n";
				if(LaTeX) *report << "\\\\\n \\> ";
			};
      if(ErrorReport)
      {  
        double time = s->getValidator()->getCurrentHappeningTime();
        s->getValidator()->getErrorLog().addUnsatDurationCondition(time,this,s,fabs(d-testDuration));
      }; 
			return false;

		};

	};

	bool ans = pre->evaluate(s);
	if(LaTeX && !ans) *report << " \\notOK \\\\\n \\> ";
	return ans;
};

void StartAction::displayDurationAdvice(const State * s) const
{
 double testDuration = planStep->originalDuration;
 double testTolerance =  0.001;

 
for(goal_list::const_iterator i = durs->begin();i != durs->end();++i)
	{
		const comparison * c = dynamic_cast<const comparison *>(*i);
		double d = s->evaluate(c->getRHS(),bindings);
		bool test;
		switch(c->getOp())
		{
			case E_GREATER:
				test = (testDuration > d);
        	if(!test)
				{
					*report << "Failed duration constraint: Increase duration by at least " << d-testDuration;

              if(LaTeX) *report << "\\\\";
            *report << "\n";
				};      
				break;
			case E_LESS:
				test = (testDuration < d);
  				if(!test)
				{
					*report << "Failed duration constraint: Decrease duration by at least " << testDuration - d;
              if(LaTeX) *report << "\\\\";
              *report << "\n";
				};
				break;
			case E_GREATEQ:
				test = (s->getTolerance() >= d - testDuration);
				if(!test)
				{
					*report << "Failed duration constraint: Increase duration by at least " << d-testDuration;
              if(LaTeX) *report << "\\\\";
              *report << "\n";
				};
				break;
			case E_LESSEQ:
				test = (testDuration - d <= testTolerance);
				if(!test)
				{

					*report << "Failed duration constraint: Decrease duration by at least " << testDuration - d;
              if(LaTeX) *report << "\\\\";
              *report << "\n";
				};
				break;
			case E_EQUALS:
				test = (testDuration > d?testDuration - d:d - testDuration) < testTolerance;
				if(!test)
				{

					*report << "Failed duration constraint: Set the duration to " << d;
              if(LaTeX) *report << "\\\\";
            *report << "\n";
				};
				break;
			default:
				break;
		};

     };
    
};

void Action::displayEventInfomation() const
{
  if(LaTeX)
  {
       *report << "\\> \\aeventtriggered{"<<*this<<"}\\\\\n";
  }
  else if(Verbose)
  {
      *report << "Triggered event "<<*this<<"\n";    
  };
  
};

void StartAction::displayEventInfomation() const
{
  if(LaTeX)
  {
      *report << "\\> \\aprocessactivated{"<<getName()<<"}\\\\\n"; 
  }
  else if(Verbose)
  {
      *report << "Activated process "<<getName()<<"\n";
  };

};

void EndAction::displayEventInfomation() const
{
  if(LaTeX)
  {        
      *report << "\\> \\aprocessunactivated{"<<getName()<<"}\\\\\n";
  }
  else if(Verbose)
  {
      *report << "Unactivated process "<<getName()<<"\n";
  };


};


struct MIP {

	Ownership & own;

	MIP(Ownership & o) : own(o) {};

	void operator()(const CondCommunicationAction * cca) 
	{
		cca->markInitialPreconditions(own);
	};
};

void
StartAction::markOwnedPreconditions(Ownership & o) const
{
	for_each(condActions.begin(),condActions.end(),MIP(o));

	DurativeActionElement::markOwnedPreconditions(o);
};

void CondCommunicationAction::markInitialPreconditions(Ownership& o) const
{
	if(initPre)
		initPre->markOwnedPreconditions(this,o);
};
	
bool
StartAction::confirmPrecondition(const State * s) const
{
	for(vector<const CondCommunicationAction *>::const_iterator i = condActions.begin();
			i != condActions.end();++i)
	{
		if(!(*i)->confirmInitialPrecondition(s))
		{
			if(LaTeX) *report << " \\notOK \\\\\n \\> ";
			return false;
		};
	};

	if(ctsEffects != 0) //may be no ctsEffect action if no cts effects
	{
		for(vector<const CondCommunicationAction *>::const_iterator i = ctsEffects->condActions.begin();
				i != ctsEffects->condActions.end();++i)
		{
			if(!(*i)->confirmInitialPrecondition(s))
			{	
				if(LaTeX) *report << " \\notOK \\\\\n \\> ";
				return false;
			};
		};
	};

	return DurativeActionElement::confirmPrecondition(s);
};

bool CondCommunicationAction::confirmInitialPrecondition(const State * s) const
{		
	cout << "First one being checked\n";
	if(!initPre) 
	{
		status = true;
		return true;
	};
	
	status = initPre->evaluate(s);
	cout << "Status now " << status << "\n";
	return true;
};



void
Action::markOwnedPreconditions(Ownership & o) const
{
	pre->markOwnedPreconditions(this,o);
};


void
DurativeActionElement::markOwnedPreconditions(Ownership & o) const
{
	pre->markOwnedPreconditions(this,o);
	for(goal_list::const_iterator i = durs->begin();i != durs->end();++i)
	{
		const comparison * c = dynamic_cast<const comparison *>(*i);	

		o.markOwnedPreconditionFEs(this,c->getRHS(),bindings);
	};
};

// Perhaps should be void - throw an exception if the ownership conditions are
// violated?
bool 
Action::constructEffects(Ownership & o,EffectsRecord & e,const State * s,bool markPreCons) const
{
	/* We are going to work through the effect_lists to handle each component in
	 * turn. We have a PropositionFactory that we can use to construct the literals
	 * for the terminal conditions.
	 * 
	 * There is recursion required to handle conditional effects.
	 */

	return handleEffects(o,e,s,act->effects,markPreCons);
};


struct FAEhandler {
	Validator * vld;
	const Action * a;
	Ownership & o;
	EffectsRecord & e;
	const State * s;
	const effect_lists * effs;
	Environment & bds;
	
	var_symbol_list::const_iterator i;
	const var_symbol_list::const_iterator endpt;
	vector<const_symbol *> cs;

// Switched from var_symbol_tab to var_symbol_list (var_symbol_tab seems to be empty
// in certain cases where it is not expected to be...)
	FAEhandler(Validator * v,const Action * ia,Ownership & io,EffectsRecord & ie,const State *is,
					const forall_effect * eff,const Environment & bs) :
			vld(v), a(ia), o(io), e(ie), s(is), effs(eff->getEffects()), bds(*bs.copy(v)),
			i(eff->getVarsList()->begin()), endpt(eff->getVarsList()->end()),
			cs(vld->range(*i)) {};

	bool handle(bool markPreCons)
	{
	//cout << "About to handle an effect\n";
		if(i == endpt) 
		{
		//cout << "i was ended\n";
			Environment * env = bds.copy(vld);
			return a->handleEffects(o,e,s,effs,*env,markPreCons);
		};
		var_symbol_list::const_iterator j = i++;
		// Inefficient to repeat the lookup for the range of (*i) when dropping down through this variable
		// for second+ times.
		vector<const_symbol *> ds = i != endpt?vld->range(*i):vector<const_symbol *>();
		ds.swap(cs);
		for(vector<const_symbol *>::iterator k = ds.begin();k != ds.end();++k)
		{
			//cout << "Handling " << (*j)->getName() << " = " << (*k)->getName() << " and i: " << (i==endpt) << "\n";
			bds[*j] = *k;
			if(!handle(markPreCons))
			{
				i = j;
				ds.swap(cs);
				return false;
			}
		};
		
		i = j;
		ds.swap(cs);
		return true;
	};
};

bool 
Action::handleEffects(Ownership & o,EffectsRecord & e,
						const State * s,const effect_lists * effs,const Environment & bds,bool markPreCons) const
{
	for(list<simple_effect*>::const_iterator i = effs->add_effects.begin();
					i != effs->add_effects.end();++i)
	{   
		const SimpleProposition * p = vld->pf.buildLiteral((*i)->prop,bds);
		if(!o.ownsForAdd(this,p)) 
		{
			return false;
		};
		e.pushAdd(p,this);
	};

	for(list<simple_effect*>::const_iterator i1 = effs->del_effects.begin();

					i1 != effs->del_effects.end();++i1)
	{
		const SimpleProposition * p = vld->pf.buildLiteral((*i1)->prop,bds);
		if(!o.ownsForDel(this,p))
		{
			return false;
		};
		e.pushDel(p,this);
	};
	
	for(list<cond_effect*>::const_iterator i2 = effs->cond_effects.begin();
			i2 != effs->cond_effects.end();++i2)
	{
		// First check preconditions are satisfied.
		const Proposition * p = vld->pf.buildProposition((*i2)->getCondition(),bds);
		//cout << "Checking " << *p << " in " << *s << " to get " << p->evaluate(s) << "\n";
		if(p->evaluate(s))
		{

			if( (markPreCons && !p->markOwnedPreconditions(this,o)) ||
				!handleEffects(o,e,s,(*i2)->getEffects(),bds,markPreCons)) 
			{
				p->destroy();
				if(Verbose) 
					*report << "Violation in conditional effect in " << this;
				return false;
			};
		};
		p->destroy();

	};

	

	for(list<assignment*>::const_iterator i3 = effs->assign_effects.begin();

			i3 != effs->assign_effects.end();++i3)
	{
		// LHS is owned for appropriate update.
		// RHS will be owned as if for preconditions.

		// Assignment cannot be applied because of the usual problem of conditional
		// effects. RHS can be evaluated and then the update recorded.
		const FuncExp * lhs = vld->fef.buildFuncExp((*i3)->getFTerm(),bds);
		FEScalar v = s->evaluate((*i3)->getExpr(),bds);
		if(!o.markOwnedEffectFE(this,lhs,(*i3)->getOp(),(*i3)->getExpr(),bds))
		{
			return false;
		};
		e.addFEffect(lhs,(*i3)->getOp(),v,this);
	};

	for(list<forall_effect*>::const_iterator i4 = effs->forall_effects.begin();

			i4 != effs->forall_effects.end();++i4)
	{
		FAEhandler faeh(vld,this,o,e,s,*i4,bds);
		if(!faeh.handle(markPreCons)) return false;
	};
	
	return true;
};

bool Action::handleEffects(Ownership & o,EffectsRecord & e,
							const State * s,const effect_lists * effs,bool markPreCons) const
{
	return handleEffects(o,e,s,effs,bindings,markPreCons);
};

Action::Action(Validator * v,const operator_ * a,const const_symbol_list* bs) : 
		act(a), bindings(buildBindings(a,*bs)), 
		timedInitialLiteral(a->name->getName().substr(0,6)=="Timed "),
		vld(v), pre(vld->pf.buildProposition(act->precondition,bindings)), planStep(0)
{
  string n = act->name->getName();

	for( var_symbol_list::const_iterator i = act->parameters->begin() ; i != act->parameters->end(); ++i)
	{
		n += bindings.find(*i)->second->getName();
	};

  actionName = n;
};

Action::Action(Validator * v,const operator_ * a,Environment * bs) : 
		act(a), bindings(*bs), 
		timedInitialLiteral(a->name->getName().substr(0,6)=="Timed "),
		vld(v), pre(vld->pf.buildProposition(act->precondition,bindings)), planStep(0)
{
  string n = act->name->getName();

	for( var_symbol_list::const_iterator i = act->parameters->begin() ; i != act->parameters->end(); ++i)
	{
		n += bindings.find(*i)->second->getName();
	};

  actionName = n;

  cout << "Just built " << n << "\n";
};


Action::Action(Validator * v,const operator_ * a,const vector<const_symbol *> & bs) :
		act(a), bindings(buildBindings(a,bs)), 
		timedInitialLiteral(a->name->getName().substr(0,6)=="Timed "),
		vld(v), pre(vld->pf.buildProposition(act->precondition,bindings)), planStep(0)
{
  string n = act->name->getName();

	for( var_symbol_list::const_iterator i = act->parameters->begin() ; i != act->parameters->end(); ++i)
	{
		n += bindings.find(*i)->second->getName();
	};

  actionName = n;
};


Action::Action(Validator * v,const operator_ * a,const const_symbol_list* bs,const plan_step * ps) :
		act(a), bindings(buildBindings(a,*bs)),
		timedInitialLiteral(a->name->getName().substr(0,6)=="Timed "),
		vld(v), pre(vld->pf.buildProposition(act->precondition,bindings)), planStep(ps)
{
  string n = act->name->getName();

	for( var_symbol_list::const_iterator i = act->parameters->begin() ; i != act->parameters->end(); ++i)
	{
		n += bindings.find(*i)->second->getName();
	};

  actionName = n;
};


void buildForAllCondActions(Validator * vld,const durative_action * da,
	const const_symbol_list * params,goal_list * gls,
		goal_list * gli,goal_list * gle,effect_lists * locels,
		effect_lists * locele,const var_symbol_list * vars,
		var_symbol_list::const_iterator i,
		vector<const CondCommunicationAction *> & condActions,
		Environment * env)
{
	cout << "OK, ready to go\n";

	if(i == vars->end())
	{
		cout << "Ready to construct one\n";
		condActions.push_back(new CondCommunicationAction(vld,da,params,gls,gli,gle,locels,locele,env));
	}
	else
	{
		vector<const_symbol *> vals = vld->range(*i);
		const var_symbol * v = *i;
		
		++i;
		for(vector<const_symbol*>::iterator j = vals.begin();j != vals.end();++j)
		{
		  cout << " considering value " << (*j)->getName() << "\n";
			(*env)[v] = *j;
			buildForAllCondActions(vld,da,params,gls,gli,gle,locels,locele,vars,i,condActions,env);
		};
		--i;
	}
}


		
CondCommunicationAction::CondCommunicationAction(Validator * v,const durative_action * a,const const_symbol_list * bs,
	goal_list * gs,goal_list * gi,goal_list * ge,
	effect_lists * es,effect_lists * el) : 
	Action(v,new safeaction(a->name,a->parameters,new conj_goal(const_cast<goal_list*>(ge)),el,a->symtab),bs), 
	status(true),
	gls(new conj_goal(const_cast<goal_list*>(gs))), 
	initPre(gs->empty()?0:vld->pf.buildProposition(gls,bindings)),
	gli(new conj_goal(const_cast<goal_list*>(gi))), 
	invPre(gi->empty()?0:vld->pf.buildProposition(gli,bindings)),
	els(es) {};


CondCommunicationAction::CondCommunicationAction(Validator * v,const durative_action * a,const const_symbol_list * bs,
	goal_list * gs,goal_list * gi,goal_list * ge,
	effect_lists * es,effect_lists * el,Environment * vs) : 
	Action(v,new safeaction(a->name,a->parameters,new conj_goal(const_cast<goal_list*>(ge)),el,a->symtab),vs), 
	status(true),
	gls(new conj_goal(const_cast<goal_list*>(gs))), 
//	initPre(gs->empty()?0:vld->pf.buildProposition(gls,bindings)),
	gli(new conj_goal(const_cast<goal_list*>(gi))), 
//	invPre(gi->empty()?0:vld->pf.buildProposition(gli,bindings)),
	els(es)
{
	cout << "I have a real forall CCA to build for variables: ";
	

};
	
CondCommunicationAction::~CondCommunicationAction() {
	delete initPre;
	delete invPre;
	delete act;
	conj_goal * cg = dynamic_cast<conj_goal*>(gls);
	if(cg) const_cast<goal_list *>(cg->getGoals())->clear();
	cg = dynamic_cast<conj_goal*>(gli);

	if(cg) const_cast<goal_list *>(cg->getGoals())->clear();

	delete gls;
	delete gli;
	els->add_effects.clear();
	els->del_effects.clear();
	els->forall_effects.clear();
	els->cond_effects.clear();
	els->assign_effects.clear();	
	delete els;
};


void Action::adjustContext(ExecutionContext & ec) const
{};

void Action::adjustContextInvariants(ExecutionContext & ec) const

{};

void Action::adjustActiveCtsEffects(ActiveCtsEffects & ace) const
{};

struct ContextAdder {


	ExecutionContext & ec;

	ContextAdder(ExecutionContext & e) : ec(e) {};

	void operator()(const CondCommunicationAction* cca)
	{
		if(cca->isActive())
			ec.addCondAction(cca);
	};
};

void StartAction::adjustContext(ExecutionContext & ec) const
{
	ec.addInvariant(invariant);
	for_each(condActions.begin(),condActions.end(),ContextAdder(ec));
};

void StartAction::adjustActiveCtsEffects(ActiveCtsEffects & ace) const
{      
	ace.addCtsEffect(ctsEffects);        
};

struct ContextRemover {

	ExecutionContext & ec;

	ContextRemover(ExecutionContext & e) : ec(e) {};

	void operator()(const CondCommunicationAction* cca)
	{
			ec.removeCondAction(cca);
	};
};

void EndAction::adjustContextInvariants(ExecutionContext & ec) const
{	
	if(invariant != 0) invariant->setRhsIntervalOpen(true);
	for(vector<const CondCommunicationAction *>::const_iterator i = condActions.begin(); i != condActions.end(); ++i)
	  {
	    	(*i)->setRhsIntervalOpen(true);	
	  };

};



void StartAction::adjustContextInvariants(ExecutionContext & ec) const
{
	if(invariant != 0) invariant->setRhsIntervalOpen(false);
	for(vector<const CondCommunicationAction *>::const_iterator i = condActions.begin(); i != condActions.end(); ++i)
	  {
	    	(*i)->setRhsIntervalOpen(false);	

	  };	
};

void EndAction::adjustContext(ExecutionContext & ec) const
{
	ec.removeInvariant(invariant);
	for_each(condActions.begin(),condActions.end(),ContextRemover(ec));
};

void EndAction::adjustActiveCtsEffects(ActiveCtsEffects & ace) const

{
	ace.removeCtsEffect(ctsEffects);


};

ostream & operator <<(ostream & o,const Action & a)
{
	a.write(o);
	return o;
};

ostream & operator <<(ostream & o,const Action * const a)
{
	a->write(o);
	return o;
};

void CondCommunicationAction::markOwnedPreconditions(Ownership & o) const 
{
	if(invPre && status)

		invPre->markOwnedPreconditions(this,o);
};

bool CondCommunicationAction::confirmPrecondition(const State * s) const
{
cout << "Checking a CondAction prec\n";
	if(invPre && status)
	{
		if(TestingPNERobustness) ace->addActiveFEs(true);
    else ace->addActiveFEs();
  		const_cast<Proposition*>(invPre)->setUpComparisons(ace,rhsIntervalOpen);
		DerivedGoal::setACE(ace,rhsIntervalOpen);
		if(!invPre->evaluate(s)) status = false;
		DerivedGoal::setACE(0,rhsIntervalOpen);
	};

	
	return true;
};

bool CondCommunicationAction::constructEffects(Ownership & o,EffectsRecord & e,const State * s,bool markPreCons) const
{
	return true;
};

bool CondCommunicationAction::constructFinalEffects(Ownership & o,EffectsRecord & e,const State * s) const

{
	if(status) 

	{

		return Action::constructEffects(o,e,s,true);
	};
	return true;
};

bool EndAction::constructEffects(Ownership & o,EffectsRecord & e,const State * s,bool markPreCons) const 
{
	if(!Action::constructEffects(o,e,s,markPreCons)) 
	{
		return false;
	};
	for(vector<const CondCommunicationAction*>::const_iterator i = condActions.begin();
				i != condActions.end();++i)
	{
		
			if(!(*i)->constructFinalEffects(o,e,s))
			{
				if(Verbose)
				{
					*report << "Failure in application of effects for temporal conditional effect in " << (*i) << "\n";
				};
				return false;
			};
		
	};
	return true;
};

void CtsEffectAction::displayCtsFtns() const
{
	//for LaTeX use in plan validation
	for(map<const FuncExp *,ActiveFE*>::const_iterator afe = ace->activeFEs.begin(); afe != ace->activeFEs.end(); ++afe)
	{
		*report << " \\> \\function{"<<*(afe->first)<<"}{"<< *(afe->second->ctsFtn) << "}\\\\\n";
	};

};

bool CtsEffectAction::constructEffects(Ownership & o,EffectsRecord & e,const State * s,bool markPreCons) const

{
         
	//deal with all cts effects at once as given from ace 
	if(ace != 0)
	{           
		//process cts effects, build polys etc, conditional cts effects handled here also
		if(TestingPNERobustness) ace->addActiveFEs(true);
    else ace->addActiveFEs();

                
		
		for(map<const FuncExp *,ActiveFE*>::const_iterator i = ace->activeFEs.begin();
				i != ace->activeFEs.end();++i)
		{

	                     
		      e.addFEffect(i->first,E_ASSIGN_CTS, i->second->evaluate(ace->localUpdateTime), this );
		  
		};

		//record points for graph drawing
		if(LaTeX)
		{
			latex.LaTeXBuildGraph(ace,s);
		};		
	}
	else
	{
		 UnrecognisedCondition uc;
		 throw uc;	 
	};

	
	return true;

};

const Action * StartAction::partner() const {return otherEnd;};
const Action * InvariantAction::partner() const {return start;};
const Action * CtsEffectAction::partner() const {return start;};
const Action * CondCommunicationAction::partner() const {return start;};

};


