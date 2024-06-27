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
#include "State.h"
#include "Plan.h"
#include "Action.h"
#include "Validator.h"
#include "main.h"
#include "Polynomial.h"
#include "Exceptions.h"
#include "ptree.h"
#include "RobustAnalyse.h"

using std::make_pair;
using std::ptr_fun;
using std::find;
using std::remove;
using std::sort;
using std::find_if;
using std::copy;
using std::for_each;
using std::bind1st;
using std::mem_fun;

//#define vector std::vector
//#define map std::map
//#define list std::list

namespace VAL {
  
Happening::Happening(Validator * v,const vector<pair<double,Action*> > & as,double timeEndPlan) : vld(v),
 time(0.0), actions(), eventHappening(false), realHappening(false), afterPlan(false)
{
	time = as.begin()->first;
	afterPlan = time > timeEndPlan;
	std::transform(as.begin(),as.end(),std::back_inserter(actions),select2nd<pair<double,Action*> >());
	realHappening = (find_if(actions.begin(),actions.end(),mem_fun(&Action::isRegAction)) != actions.end());
  
};

Happening::Happening(Validator * v,double timeToExecute,const vector<pair<double,Action*> > & as) :
	vld(v), time(v->getState().getTime()+timeToExecute),actions(),
	eventHappening(false), realHappening(false), afterPlan(false)
{
	std::transform(as.begin(),as.end(),std::back_inserter(actions),select2nd<pair<double,Action*> >());
	realHappening = (find_if(actions.begin(),actions.end(),mem_fun(&Action::isRegAction)) != actions.end());
};


//for creating event happenings
Happening::Happening(Validator * v, vector<const Action*> acts,bool eve): vld(v),
 time(v->getState().getTime()), actions(acts), eventHappening(eve), realHappening(false), afterPlan(false)
{
 //afterPlan = time > timeEndPlan; may need later for processes, then events may be after the main plan
	//std::transform(acts.begin(),acts.end(),std::back_inserter(actions));
	
};

//for creating event happenings
Happening::Happening(Validator * v, vector<const Action*> acts,double t,bool eve): vld(v),
 time(t), actions(acts), eventHappening(eve), realHappening(false), afterPlan(false)
{
 //afterPlan = time > timeEndPlan; may need later for processes, then events may be after the main plan
	//std::transform(acts.begin(),acts.end(),std::back_inserter(actions));

};

bool
Happening::canHappen(const State * s) const
{
   if(!eventHappening)
   {
   	if(LaTeX)
   		*report << "\\atime{"<<time<<"} \\> \\checkhappening";
   	else if(Verbose) cout << "Checking next happening (time " << time << ")\n";
   }
   else
   {    
       	if(LaTeX)
   		*report << "\\atime{"<<time<<"} \\> \\eventtriggered \\\\\n";
   	else if(Verbose) cout << "EVENT triggered at (time " << time << ")\n";
     if(Verbose)
     {
       	for(vector<const Action*>::const_iterator act = actions.begin();act != actions.end();++act)
       	{
            (*act)->displayEventInfomation();
        };
     };
    
      return true; //no need to check event preconditions, these are all satisfied! (if not then there is an error calculating the events!)    
   };
        
  bool isOK = true;
	for(vector<const Action*>::const_iterator a = actions.begin();a != actions.end();++a)
	{   
                
		if(!(*a)->confirmPrecondition(s))

		{
			if(LaTeX)
			{
				*report << "Plan failed because of unsatisfied precondition in:\\\\\n \\> " << (*a)<< "\\\\\n";
			}
			else if(Verbose) cout << "Plan failed because of unsatisfied precondition in:\n" << (*a)<< "\n";

          if(Verbose || ErrorReport) (*a)->addErrorRecord(time,s);
			if(!ContinueAnyway) return false;
          isOK = false;
		};
        
	};
      
   if(!isOK) return false;
   
	if(LaTeX) *report << "\\happeningOK\\\\\n"; 
	         

	return true;
	
};

Happening::~Happening()
{


	for(vector<const Action *>::const_iterator i = actions.begin();i != actions.end();++i)
	{
			delete (*i);
	};
};




// Is it best to return a bool or throw an exception?
bool 
Happening::applyTo(State * s) const
{
	/* This function will update the state:
	 * 
	 * 	- for each action, need to determine which conditional effects are active;
	 * 	- must also mark ownership of propositions and functional expressions;
	 * 	- confirm no interaction;
	 * 	- set the time in new state.
	 */


  

	// First establish ownership of preconditions.
	Ownership own(vld);

  vector<const Action *> eventsForMutexCheck = vld->getEvents().getEventsForMutexCheck(); //this is in case an event is triggered at the same time as an action
  if(!eventsForMutexCheck.empty())
  {
      bool repeated;
      string eventName;
      vector<const Action*>::iterator b = eventsForMutexCheck.begin();
      for(;b != eventsForMutexCheck.end();)
      {
          repeated = false;
          eventName = (*b)->getName0();
          for(vector<const Action*>::const_iterator c = actions.begin();c != actions.end();++c)
	        {
             if((*c)->getName0() == eventName) repeated = true;  //there will be only a few possible values of b and c...
          };
          if(repeated) b = eventsForMutexCheck.erase(b);
          else ++b;
      };

        for(vector<const Action*>::const_iterator b2 = eventsForMutexCheck.begin();b2 != eventsForMutexCheck.end();++b2)
	      {
           (*b2)->markOwnedPreconditions(own);

	      };   
  };

  
	//if one action do not mark since cant conflict
	bool markPreCons = ((actions.size() + eventsForMutexCheck.size()) > 1);
	if(markPreCons)
	  {
	    for(vector<const Action*>::const_iterator a = actions.begin();a != actions.end();++a)
	    {     
		      (*a)->markOwnedPreconditions(own);
	    };
         
	  };


  
	EffectsRecord effs;
	                   
	for(vector<const Action*>::const_iterator a1 = actions.begin();a1 != actions.end();++a1)
	{            
		if(!(*a1)->constructEffects(own,effs,s,markPreCons)) return false;
	};

	EffectsRecord effsDummy; //do not want to apply the effects now only check they do not conflict
	for(vector<const Action*>::const_iterator b1 = eventsForMutexCheck.begin();b1 != eventsForMutexCheck.end();++b1)
	{
		
     		if(!(*b1)->constructEffects(own,effsDummy,s,markPreCons)) return false;
	};
  
	//display polys/fns
	if(LaTeX)
	{
      if(actions.size() != 0)
      {
      		vector<const Action*>::const_iterator a2 = actions.begin();
      		if(CtsEffectAction * cea = dynamic_cast<CtsEffectAction*>(const_cast<Action*>(*a2)))
      		{
      			cea->displayCtsFtns();

      		};
     };

	};
	              
	effs.enact(s);
    s->nowUpdated(this);
    
	return true;

};

void Happening::adjustContext(ExecutionContext & ec) const
{
	for(vector<const Action*>::const_iterator a = actions.begin();a != actions.end();++a)
	{
		(*a)->adjustContext(ec);	
	};
};

void Happening::adjustContextInvariants(ExecutionContext & ec) const
{


	for(vector<const Action*>::const_iterator a = actions.begin();a != actions.end();++a)
	{	
		(*a)->adjustContextInvariants(ec);	
	};
};

void Happening::adjustActiveCtsEffects(ActiveCtsEffects & ace) const
{
	for(vector<const Action*>::const_iterator a = actions.begin();a != actions.end();++a)
	{		

		(*a)->adjustActiveCtsEffects(ace);
	};

	ace.ctsEffectsProcessed = false;
};

void ExecutionContext::addCondAction(const CondCommunicationAction * ca)
{
	invariants.actions.push_back(ca);
};

bool ExecutionContext::removeCondAction(const CondCommunicationAction * ca)
{
	vector<const Action *>::iterator i = find(invariants.actions.begin(),invariants.actions.end(),ca);
	if(i != invariants.actions.end())
	{
		invariants.actions.erase(i);
		return true;
	};
	return false;
};

ExecutionContext::ExecutionContext(Validator * v) :
	invariants(v)
{};

void ExecutionContext::addInvariant(const InvariantAction * a)
{
	if(a != 0) invariants.actions.push_back(a);
};	

void ExecutionContext::removeInvariant(const InvariantAction * a)
{
	invariants.actions.erase(remove(invariants.actions.begin(),invariants.actions.end(),a),
										invariants.actions.end());
};

void ExecutionContext::setTime(double t)
{
	invariants.time = t;
};

ExecutionContext::~ExecutionContext()
{
	invariants.actions.clear();
};



bool ExecutionContext::hasInvariants() const
{
	return !(invariants.actions.empty());

};


void ActiveFE::addParentFE(const ActiveFE * afe)



{
	

	//check if activeFE with this FE already exists, if not add it
	vector<const ActiveFE*>::const_iterator i = find(parentFEs.begin(),parentFEs.end(),afe);
	
	if(i == parentFEs.end()) parentFEs.push_back(afe);
			
};

void ActiveFE::removeParentFE(const ActiveFE * a)
{
	parentFEs.erase(remove(parentFEs.begin(),parentFEs.end(),a),parentFEs.end());
};


void ExecutionContext::setActiveCtsEffects(ActiveCtsEffects * ace)
{
	for(vector<const Action*>::const_iterator a = invariants.actions.begin();a != invariants.actions.end();++a)
	{		
		if(InvariantAction * ia = dynamic_cast<InvariantAction*>(const_cast<Action*>(*a)))
			ia->setActiveCtsEffects( ace );
		else if(CondCommunicationAction * cca = dynamic_cast<CondCommunicationAction*>(const_cast<Action*>(*a)))
			cca->setActiveCtsEffects( ace );		
	};
	
};


ActiveCtsEffects::ActiveCtsEffects(Validator * v) :
	ctsEffects(v), ctsUpdateHappening(v), vld(v), eventTime(0)
{};


void ActiveCtsEffects::addCtsEffect(const CtsEffectAction * a)

{
	if(a != 0) ctsEffects.actions.push_back(a);
};

void ActiveCtsEffects::removeCtsEffect(const CtsEffectAction * a)
{

	ctsEffects.actions.erase(remove(ctsEffects.actions.begin(),ctsEffects.actions.end(),a),
										ctsEffects.actions.end());
};

void ActiveCtsEffects::setTime(double t)
{
	ctsEffects.time = t;
};

void ActiveCtsEffects::setLocalUpdateTime(double t)
{			   
	localUpdateTime = t;
};



const Happening * ActiveCtsEffects::getCtsEffectUpdate() const

{
	//return a new happening with only one ctseffectaction that has a pointer to the ace so that they may all be updated
  
   ctsUpdateHappening.actions.clear();
   
	ctsUpdateHappening.time = ctsEffects.time;
	
	vector<const Action*>::const_iterator a = ctsEffects.actions.begin();

	dynamic_cast<CtsEffectAction*>( const_cast<Action*>(*a) )->setActiveCtsEffects( const_cast<ActiveCtsEffects*>(this) );
	
	ctsUpdateHappening.actions.push_back(*a);
      
	return &ctsUpdateHappening;
};
	
bool ActiveCtsEffects::isFEactive(const FuncExp * fe) const
{
	map<const FuncExp *,ActiveFE*>::const_iterator i = activeFEs.find(fe);

	if(i != activeFEs.end()) return true;

	return false;

};

struct FACtsEhandler {
	Validator * vld;
	ActiveCtsEffects * ace;
	const effect_lists * effs;
	Environment & bds;
	
	var_symbol_table::const_iterator i;
	const var_symbol_table::const_iterator endpt;
	vector<const_symbol *> cs;

	FACtsEhandler(Validator * v,ActiveCtsEffects * a,const forall_effect * eff,const Environment & bs) :
			vld(v), ace(a), effs(eff->getEffects()), bds(*bs.copy(v)),
			i(eff->getVars()->begin()), endpt(eff->getVars()->end()),
			cs(vld->range(i->second)) {};

	bool handle()
	{
			
		if(i == endpt) 
		{
			Environment * env = bds.copy(vld);
			for(list<assignment*>::const_iterator ae = effs->assign_effects.begin();

				ae != effs->assign_effects.end();++ae)
			{
				ace->addActiveFE(*ae,*env);
			};

			return true;
		};

		var_symbol_table::const_iterator j = i++;
		vector<const_symbol *> ds = i != endpt?vld->range(i->second):vector<const_symbol *>();
		ds.swap(cs);
		for(vector<const_symbol *>::iterator k = ds.begin();k != ds.end();++k)
		{
			bds[j->second] = *k;
			if(!handle()) return false;
		};
		return true;
	};
};

void ActiveCtsEffects::addActiveFE(assignment * e,const Environment & bs)

{


	ActiveFE * afe;
	const FuncExp * lhs = vld->fef.buildFuncExp(e->getFTerm(),bs);
	bool increase = true;
            //cout << *lhs <<"\n";
	//check if activeFE with this FE already exists, if not add it
	map<const FuncExp *,ActiveFE*>::const_iterator i = activeFEs.find(lhs);
	if(i != activeFEs.end())
	{
		afe = i->second;
	}
	else
	{
		afe = new ActiveFE(lhs);
		activeFEs[lhs] = afe;
		//cout<<*lhs<<" \\\\\n added as active fe \n";
	};

	if(e->getOp() == E_DECREASE) increase = false;
                         
	//record the expressions that update the FE and if its increasing
	afe->exprns.push_back(pair<pair<const expression *,bool>,const Environment *>(pair<const expression *,bool>(e->getExpr(),increase),&bs) ); 
                  
};
		
void ActiveCtsEffects::addActiveFEs(bool reCalc)
{
        
	if(ctsEffectsProcessed && !reCalc) return;
         
  
  for(map<const FuncExp *, ActiveFE*>::iterator i = activeFEs.begin(); i != activeFEs.end(); ++i)
  {           //cout << " deleting"<< *(i->first) <<"\n";
      delete i->second;
  };
  
	activeFEs.clear();
           
	//first create list of effects, cts effects will all be assign effects(or cond assign)
	for(vector<const Action*>::iterator a = ctsEffects.actions.begin();

		a != ctsEffects.actions.end();++a)
	{		

		
		effect_lists * effects = new effect_lists();
		              
		for(list<assignment*>::const_iterator ae = (*a)->getEffects()->assign_effects.begin();
			ae != (*a)->getEffects()->assign_effects.end();++ae)
		{                           
				effects->assign_effects.push_back(*ae);
				
		};
                             
		for(list<forall_effect*>::const_iterator ae1 = (*a)->getEffects()->forall_effects.begin();
			ae1 != (*a)->getEffects()->forall_effects.end();++ae1)
		{
				effects->forall_effects.push_back(*ae1);
		};
		
		//status will be true when displaying plan so if there is a possiblity of nonlinear effects an exception will be thrown
		//loop thro' conditional effects if they are active add them to the effect list
		for(vector<const CondCommunicationAction*>::const_iterator ca =  dynamic_cast<const CtsEffectAction *>(*a)->condActions.begin();
				ca != dynamic_cast<const CtsEffectAction *>(*a)->condActions.end();++ca)
		{
				if((*ca)->isActive())
				{
					
					for(list<assignment*>::const_iterator j = (*ca)->getEffects()->assign_effects.begin();
					j != (*ca)->getEffects()->assign_effects.end();++j)
					{
							effects->assign_effects.push_back(*j);
							
					};

					for(list<forall_effect*>::const_iterator j1 = (*ca)->getEffects()->forall_effects.begin();
					j1 != (*ca)->getEffects()->forall_effects.end();++j1)


					{
							effects->forall_effects.push_back(*j1);

					};
				};
		
		};

                     
		//loop thro' effects and create active FE objects
		for(list<assignment*>::const_iterator e = effects->assign_effects.begin();
				e != effects->assign_effects.end();++e)
		{                   
			addActiveFE(*e,(*a)->bindings);
		};
			
		//loop thro' for all effects and create active FE objects
		for(list<forall_effect*>::const_iterator e1 = effects->forall_effects.begin();
				e1 != effects->forall_effects.end();++e1)
		{
			FACtsEhandler faceh(vld,this,*e1,(*a)->bindings);
			faceh.handle();
		};

     effects->assign_effects.clear();
     effects->forall_effects.clear();
     delete effects;

	};//end of creating active FEs from actions
                            
	//add parentFEs now
	for(map<const FuncExp *,ActiveFE*>::iterator a1 = activeFEs.begin();

		a1 != activeFEs.end();++a1)
	{
	  for(vector<pair<pair<const expression*,bool>,const Environment *> >::const_iterator i = a1->second->exprns.begin();

		i != a1->second->exprns.end();++i)
	    {
	      a1->second->addParentFEs(this,(*i).first.first,(*i).second );
	    };
	  
	};
                        
 //build cts functions for each active FE now
	buildAFECtsFtns();
  
	ctsEffectsProcessed = true;
};

void ActiveFE::addParentFEs(const ActiveCtsEffects * ace,const expression * e,const Environment * bs)
{
	
	if(dynamic_cast<const num_expression *>(e)) return;
	
	if(const func_term * fexpression = dynamic_cast<const func_term *>(e))

	{

		const FuncExp * fexp = ace->vld->fef.buildFuncExp(fexpression,*bs ); 
		
		//check that the FE is changing, it may it constant for this interval
		for(map<const FuncExp *,ActiveFE*>::const_iterator j = ace->activeFEs.begin();j != ace->activeFEs.end();++j)
		{
			
			if(j->second->fe == fexp)
			{	 	
				//cout<<j->second->fe<<" is parent of "<<this->fe<<"\n";
			 	addParentFE(j->second);
			 	break;
			};
			
		};

		return;
		
	};

	
	if(const binary_expression * bexp = dynamic_cast<const binary_expression *>(e))
	{
		addParentFEs(ace,bexp->getLHS(),bs);
		addParentFEs(ace,bexp->getRHS(),bs);
		return;
	};

	
	if(const uminus_expression * uexp = dynamic_cast<const uminus_expression *>(e))
	{
		addParentFEs(ace,uexp->getExpr(),bs);
		return;
	};
	
	if(dynamic_cast<const special_val_expr *>(e))
	{
		return;
	};



	if(Verbose) *report << "Unrecognised expression type\n";
	UnrecognisedCondition uc;
	throw uc;
	
};

bool ActiveFE::appearsInEprsn(const ActiveCtsEffects * ace,const expression * e,const Environment * bs) const
{

	if(const func_term * fexpression = dynamic_cast<const func_term *>(e))
	{
		const FuncExp * fexp = ace->vld->fef.buildFuncExp(fexpression,*bs );
		if(fe == fexp) return true;
	};


	if(const binary_expression * bexp = dynamic_cast<const binary_expression *>(e))

	{
		if(appearsInEprsn(ace,bexp->getLHS(),bs)) return true;
		if(appearsInEprsn(ace,bexp->getRHS(),bs)) return true;
	};


	if(const uminus_expression * uexp = dynamic_cast<const uminus_expression *>(e))
	{

		if(appearsInEprsn(ace,uexp->getExpr(),bs)) return true;
	};

	return false;

};

void ActiveCtsEffects::buildAFECtsFtns()
{
  //mark these PNEs as changed ctsly, in order to judder for robustness analysis
  if(Robust) for(map<const FuncExp*,ActiveFE*>::iterator i = activeFEs.begin();i != activeFEs.end();++i) const_cast<FuncExp*>(i->first)->setChangedCtsly();
  
   //sort out exp (etc) functions first then...
	//loop thro active FEs in correct order given by dependicies in diff equns (topological sort)
	//build polys from expresions and then integrate
	vector<ActiveFE*> topSortActiveFEs;
	vector<ActiveFE*> listToTopSortTempActiveFEs;
	vector<ActiveFE*> listToTopSortActiveFEs;
   vector<ActiveFE*> loopDepTempActiveFEs;
	vector<ActiveFE*> loopDepActiveFEs;
	vector<ActiveFE*> loopDepActiveFEs2;
	bool inList;

	//sift out loop dependencies for exp functions (df/dt = f)
	for(map<const FuncExp*,ActiveFE*>::iterator i = activeFEs.begin();i != activeFEs.end();++i)
	{
            bool selfDepend = false;
            for(vector<const ActiveFE*>::iterator j = i->second->parentFEs.begin(); j != i->second->parentFEs.end();++j)
            {
                if(*j == i->second)
                {
                    selfDepend = true;
                    break;
                };
            };

            if(selfDepend) loopDepTempActiveFEs.push_back(i->second);
            else listToTopSortTempActiveFEs.push_back(i->second);
	};

  //sift out dependencies for diff equations that are self dependent and depend on another from the list
   for(vector<ActiveFE*>::iterator i = loopDepTempActiveFEs.begin();i != loopDepTempActiveFEs.end();++i)

	{

            bool otherDepend = false;
            for(vector<const ActiveFE*>::iterator j = (*i)->parentFEs.begin(); j != (*i)->parentFEs.end();++j)
            {
                  //check if it depends on another FE from loopDep list
                  for(vector<ActiveFE*>::iterator k = loopDepTempActiveFEs.begin();k != loopDepTempActiveFEs.end();++k)
                 {


    			       if( *k == *j && *i != *j) { otherDepend = true; break;};
                  };          
                  if(otherDepend) break;

            };

            if(otherDepend) loopDepActiveFEs2.push_back(*i);
            else loopDepActiveFEs.push_back(*i);

    };
    
	//sift out loop dependencies for exp functions from above list (d f_2/dt = f_1, where f_1 from above list)
	for(vector<ActiveFE*>::iterator i = listToTopSortTempActiveFEs.begin();i != listToTopSortTempActiveFEs.end();++i)
	{
		if( (*i)->parentFEs.size() > 0)
		{
			//is in loopDepActiveFEs
			inList = false;

			for(vector<ActiveFE*>::iterator j = loopDepActiveFEs.begin();j != loopDepActiveFEs.end();++j)
			{
             for(vector<const ActiveFE*>::iterator k = (*i)->parentFEs.begin();k != (*i)->parentFEs.end();++k)
             {

			       if( *k == *j) { inList = true; break;};
              };
              if(inList) break;
			};

			if(inList)
			{
				loopDepActiveFEs2.push_back(*i);
				//cout << "\\\\\\> Exp fn 2 "<< (*i)->fe <<"\\\\\n";
			}
			else
			{
				//cout << "\\> Top sort fn 2 "<< (*i)->fe <<"\\\\";

				listToTopSortActiveFEs.push_back(*i);
			};


		}

		else
		{
			//cout << "\\> Top sort fn default 2 "<< (*i)->fe <<"par szie="<<(*i)->parentFEs.size()<<"\\\\";

			listToTopSortActiveFEs.push_back(*i);
		};
	};

	//do topological sort on remaining AFEs
	for(vector<ActiveFE*>::iterator i = listToTopSortActiveFEs.begin();i != listToTopSortActiveFEs.end();++i)
	{
		(*i)->colour = 0; //white
	};

	for(vector<ActiveFE*>::iterator i = listToTopSortActiveFEs.begin();i != listToTopSortActiveFEs.end();++i)
	{
		visitActiveFE(*i,topSortActiveFEs);

	};

	//define polys from top sorted list should be no errors
	for(vector<ActiveFE*>::iterator i = topSortActiveFEs.begin();i != topSortActiveFEs.end();++i)
	{                          
		if( (*i)->ctsFtn == 0) (*i)->ctsFtn = buildPoly(*i);
	};

	//define exps now and numerical solutions

	for(vector<ActiveFE*>::iterator i = loopDepActiveFEs.begin();i != loopDepActiveFEs.end();++i)
	{                        
//		cout << *((*i)->fe) << " is the expression and " << (*i)->exprns.size() << " its size\n";
//		for(vector<pair< pair<const expression *,bool> ,const Environment *> >::const_iterator j = (*i)->exprns.begin();
//			j != (*i)->exprns.end();++j)
//		{
//			cout << *(j->first.first) << " " << j->first.second << "\n";
//			
//		}
		  if( (*i)->canResolveToExp(activeFEs,vld))
         {
  //       	cout << "We're going to try to build an Exponential here\n";
           if( (*i)->ctsFtn == 0) (*i)->ctsFtn = buildExp(*i);
         }
         else    
         {                            
            if((*i)->ctsFtn == 0) (*i)->ctsFtn = buildNumericalSoln(*i);
         };
	};

	//define other exps now
	for(vector<ActiveFE*>::iterator i = loopDepActiveFEs2.begin();i != loopDepActiveFEs2.end();++i)
	{                                  
       if( (*i)->ctsFtn == 0) (*i)->ctsFtn = buildExp(*i); 
	};
};

bool ActiveFE::canResolveToExp(const map<const FuncExp*,ActiveFE*> activeFEs,Validator * vld) const
{
	if(exprns.size() == 1) return true;
// Add a new case to handle the situation in which we have two expressions, but one is a constant
// multiple of #t
	if(exprns.size() == 2)
	{
		if(isConstLinearChangeExpr(exprns[0],activeFEs,vld) || isConstLinearChangeExpr(exprns[1],activeFEs,vld))
			return true;
	};
	return false;
};

bool isConstLinearChangeExpr(const ExprnPair & exp,const map<const FuncExp *,ActiveFE *> activeFEs,Validator * vld)
{
	const expression * ex = getRateExpression(exp.first.first);
	return isConstant(ex,exp.second,activeFEs,vld);
};

bool isConstant(const expression * exp,const Environment * env,const map<const FuncExp *,ActiveFE *> activeFEs,Validator * vld)
{
	const func_term * ft = dynamic_cast<const func_term *>(exp);
	if(ft) 
	{
		const FuncExp * ftt = vld->fef.buildFuncExp(ft,*env);
		if(activeFEs.find(ftt) != activeFEs.end())
		{
//			cout << "Found non-constant FuncTerm " << *ftt << "\n";
			return false;
		}
		else return true;
	};
	const num_expression * nt = dynamic_cast<const num_expression *>(exp);
	if(nt)
	{
		return true;
	};
	const binary_expression * be = dynamic_cast<const binary_expression *>(exp);
	if(be)
	{
		return isConstant(be->getLHS(),env,activeFEs,vld) &&
			isConstant(be->getRHS(),env,activeFEs,vld);
	};
	return false;	
};

void ActiveCtsEffects::visitActiveFE(ActiveFE * afe,vector<ActiveFE*> & topSAFEs)

{
	if(afe->colour == 1) //check if already been visted and thus a loop
	{
		HighOrderDiffEqunError hodee;
		throw hodee;
	};
	
	if(afe->colour != 0) return;

	afe->colour = 1; //grey


	for(vector<const ActiveFE*>::iterator i = afe->parentFEs.begin();i != afe->parentFEs.end();++i)
	{
		visitActiveFE(const_cast<ActiveFE*>(*i),topSAFEs);
	};

	afe->colour = 2; //black

	topSAFEs.push_back(afe);
	return;
	
};


//extract just the rate of change for cts effect
	// - if the syntax of cts effects changes then this will need to be changed
const expression* getRateExpression(const expression* aExpression)
{
  const expression * rateExprn;
  
	if(const mul_expression * me = dynamic_cast<const mul_expression *>(aExpression))
	{
		if(dynamic_cast<const special_val_expr *>(me->getLHS()))
		{
			rateExprn = me->getRHS();
		}
		else if(dynamic_cast<const special_val_expr *>(me->getRHS()))
		{
			rateExprn = me->getLHS();
		}
		else
		{
			DiffEqunError dee;
			throw dee;
		};
	}
	else
	{

		DiffEqunError dee;
		throw dee;
	};


   return rateExprn;
};

const Polynomial * ActiveCtsEffects::buildPoly(const ActiveFE * afe)
{
	Polynomial thePoly;
	
	//loop thro exprns and create poly from each, the sum integrated is our poly
	for(vector<pair<pair<const expression*,bool>,const Environment *> >::const_iterator i = afe->exprns.begin();
		i != afe->exprns.end();++i)
	{
		//extract just the rate of change for cts effect
		// - if the syntax of cts effects changes then this will need to be changed
		if(dynamic_cast<const mul_expression *>(i->first.first))
		{
           thePoly  += getPoly(getRateExpression(i->first.first),i->first.second,this,i->second);
		}
		else if(const special_val_expr * sve = dynamic_cast<const special_val_expr *>(i->first.first))
		{

			if(sve->getKind() == E_HASHT)
			{
				Polynomial timet;
				timet.setCoeff(0,1);
				if(i->first.second)
					thePoly  += timet;
				else
					thePoly  -= timet;
			}
			else
			{
				thePoly  += getPoly(i->first.first,i->first.second,this,i->second);
			};
		
		}
		else
		{
			thePoly  += getPoly(i->first.first,i->first.second,this,i->second);
		};
		
		
	};
	


	thePoly = thePoly.integrate();

	//add boundary condition
	double bc =  afe->fe->evaluate(&(vld->getState())); 
	thePoly.setCoeff(0,bc);
	             //cout << "Boundary for "<<*(afe->fe)<<" is "<<bc<<" -- "<<thePoly <<"\n";
	const Polynomial * newPoly = new Polynomial(thePoly);
	return newPoly;
	
};

const CtsFunction * ActiveCtsEffects::buildExp(const ActiveFE * afe)
{
                                
   const Polynomial * expPoly; // p(t)
   CoScalar kValue = 0, cValue = 0; // for f(t) = K e^{p(t)} + c
	const expression* rateExprn;
	const expression* constExprn = 0;
	const expression* FEExprn = 0;

	bool simpleExp = true; //equn of form df/dt = A*f
	const FuncExp * fexp = 0;

	//not handling sums of exp functions, yet
	if(!afe->canResolveToExp(activeFEs,vld))
	{
		DiffEqunError dee;
		throw dee;
	};

	pair<pair<const expression*,bool>,const Environment *> exprn = * afe->exprns.begin();
   	rateExprn = getRateExpression(exprn.first.first);
   	const Environment * env = exprn.second;
   	bool incr = exprn.first.second;
   	const expression * constExpA = 0;
 	const Environment * envC = 0;
//cout << "Stage 1\n";
	if(afe->exprns.size() == 2)
	{
//		cout << "Stage 2\n";
		if(isConstant(rateExprn,exprn.second,activeFEs,vld))
		{
			constExpA = rateExprn;
			envC = env;
			rateExprn = getRateExpression(afe->exprns[1].first.first);
			env = afe->exprns[1].second;
			incr = afe->exprns[1].first.second;
		}
		else
		{
			constExpA = getRateExpression(afe->exprns[1].first.first);
			envC = afe->exprns[1].second;
		};
	};
//	cout << "Stage 3: " << *rateExprn << "\n";
	if(const func_term * fexpression = dynamic_cast<const func_term *>(rateExprn))
	{
		fexp = vld->fef.buildFuncExp(fexpression,*env);

		if(fexp != afe->fe) simpleExp = false;

	  }
	else if(const mul_expression * me = dynamic_cast<const mul_expression *>(rateExprn))
	{
//		cout << "Stage 4\n";
         if(afe->appearsInEprsn(this,me->getLHS(),env))
         {
             constExprn = me->getRHS();   fexp = afe->fe;
             FEExprn = me->getLHS();
         }
         else if(afe->appearsInEprsn(this,me->getRHS(),env))
         {
             constExprn = me->getLHS();   fexp = afe->fe;
             FEExprn = me->getRHS();
         }
         else
         {
             simpleExp = false;
             if(afe->parentFEs.size() == 1) fexp = (*afe->parentFEs.begin())->fe;
             
             if((*afe->parentFEs.begin())->appearsInEprsn(this,me->getRHS(),env))
             {
                 constExprn = me->getLHS();
                 FEExprn = me->getRHS();
             }


             else
             {
                  constExprn = me->getRHS();
                  FEExprn = me->getLHS();
             };

         };

	}
	else
	{
		DiffEqunError dee;
		throw dee;
	};
         //end of extracting const expression and FE expression
      
              
//cout << "We are here with " << *fexp << " and " << *constExprn << "\n";

	if(fexp == 0)
	{
		DiffEqunError dee;
		throw dee;
	};



	if(simpleExp)  //simple in the sense that does not depend on other PNES that are exps
	{
		//const exprn depends on the FE?
		if(afe->appearsInEprsn(this,constExprn,env))
		{
			DiffEqunError dee;
			throw dee;
		};


    Polynomial bPoly;

    if(constExprn != 0)
    {                                    
      		 bPoly = getPoly(constExprn,this,env,localUpdateTime);    
             bPoly = bPoly.integrate(); //leave constant term zero, this part of constant
    }
    else
             bPoly.setCoeff(1,1);
                       
      if(!(incr)) bPoly = - bPoly;

		//add boundary condition
		double bc =  afe->fe->evaluate(&(vld->getState()));




		//if poly is constant zero the exp is not an exp!
		if(bPoly.getDegree() == 0 && bPoly.getCoeff(0) == 0)
		{
			Polynomial aPoly;
         aPoly.setCoeff(0,bc);
			return new Polynomial(aPoly);
		};

      //handle different cases for FE expression, may be (a-f), (f-a) or f
      if(dynamic_cast<const func_term *>(FEExprn))
		{
//cout << "This is the case....\n";
       kValue = bc;  // divided by e^{b(0)}   

       // We can also deal with the special case where we have two expressions in the exprns list for this
       // activeFE and one is constant, A, and the other is B.f for constant B.
       if(constExpA)
       {
//       	cout << "We are handling our special case...\n";
       	cValue = -(vld->getState().evaluate(constExpA,*envC)/vld->getState().evaluate(constExprn,*env));
       	if(!incr) cValue = -cValue;
       	kValue -= cValue;
       };
       	
      }
      else if(const minus_expression * minexprn = dynamic_cast<const minus_expression *>(FEExprn))
      {


             if(dynamic_cast<const func_term *>(minexprn->getLHS()) && afe->appearsInEprsn(this,minexprn->getLHS(),env))
             {
                                 if(afe->appearsInEprsn(this,minexprn->getRHS(),env))
                            		{

                            			DiffEqunError dee;
                            			throw dee;
                            		};
                           FEScalar minExprnConst = vld->getState().evaluate(minexprn->getRHS(),*env);
                           kValue = bc - minExprnConst;
                           cValue = minExprnConst;
             }
             else if(dynamic_cast<const func_term *>(minexprn->getRHS()) && afe->appearsInEprsn(this,minexprn->getRHS(),env))

             {
                                 if(afe->appearsInEprsn(this,minexprn->getLHS(),env))
                            		{
                            			DiffEqunError dee;
                            			throw dee;

                            		};
                           FEScalar minExprnConst = vld->getState().evaluate(minexprn->getLHS(),*env);
                           kValue = bc - minExprnConst;
                           cValue = minExprnConst;
                           bPoly = - bPoly;
             }
              else
   			{
   				DiffEqunError dee;
   				throw dee;
   			};

      }
      else

			{
				DiffEqunError dee;
				throw dee;
			};

         expPoly = new Polynomial(bPoly);

	}
	else      // dg/dt = k f, where f(t) = k' e^{at}  (not simple exp)
	{

		map<const FuncExp *,ActiveFE*>::const_iterator i = activeFEs.find(fexp);
		if(i != activeFEs.end())
		{
			if(const Exponential * cf = dynamic_cast<const Exponential *>((*i).second->ctsFtn))
			{
            if( cf->getPolynomial()->getDegree() != 1)
            {

              DiffEqunError dee;
      			throw dee;
            };

				CoScalar aExpVal = cf->getPolynomial()->getCoeff(1);//cf->getA();
				CoScalar kExpVal = cf->getK();
				if(!(exprn.first.second)) kValue = - kValue;
				CoScalar constVal = 1;
				CoScalar bc =  afe->fe->evaluate(&(vld->getState()));

				if(constExprn != 0)
					constVal = vld->getState().evaluate(constExprn,*env);

				//if constants are zero the exp is not an exp!
				if(constVal == 0)
				{
					const Polynomial * newPoly = new Polynomial();
					return newPoly;
				};


				//for d f_2 /dt = a_2 f_1
				//f_2 (t) = (a_2 / a_1) * K_1 e^{a_1 t} + f_0 - (a_2 / a_1) * K_1
				kValue = (constVal/aExpVal)*kExpVal;
            cValue = bc - (constVal/aExpVal)*kExpVal;
            expPoly = new Polynomial(*cf->getPolynomial());

			}
			else if(const Polynomial * cf = dynamic_cast<const Polynomial *>((*i).second->ctsFtn))
			{
				Polynomial aPoly;
				aPoly = cf->integrate();

				//add boundary condition
				CoScalar bc =  afe->fe->evaluate(&(vld->getState()));
				aPoly.setCoeff(0,bc);

				const Polynomial * newPoly = new Polynomial(aPoly);
				return newPoly;
			}
			else
			{
				DiffEqunError dee;
				throw dee;
			};

		}
		else
		{

			DiffEqunError dee;
			throw dee;
		};
	};



	const Exponential * newExp = new Exponential(kValue,expPoly,cValue);
             
	return newExp;

};


//build numerical solution to equn: dy/dt = p(t) (m-y) + q(t)
const CtsFunction * ActiveCtsEffects::buildNumericalSoln(const ActiveFE * afe)
{                 
   CoScalar accuracy = 0.05;
   CoScalar mValue; // for  dy/dt = p(t) (m-y) + q(t)
	const expression* rateExprn;
	const expression* constExprn = 0;
	const expression* FEExprn = 0;
   vector<pair<const CtsFunction *,bool> > discharge;
   pair<pair<const expression*,bool>,const Environment *> exprn;
   bool exprnDefined = false;
   //const FuncExp * fexp = 0; // this was assigned to, but never used
            
   for(vector<pair<pair<const expression*,bool>,const Environment *> >::const_iterator i = afe->exprns.begin(); i != afe->exprns.end(); ++i)
   {

      if(!afe->appearsInEprsn(this,(*i).first.first ,i->second))
      { 
        Polynomial aPoly = (getPoly((*i).first.first,this,i->second)).diff();
        discharge.push_back(make_pair(new Polynomial(aPoly),i->first.second));
        }
      else
      {
          if(exprnDefined)
          {            
            DiffEqunError dee;
		      throw dee;
          };
          exprn = *i;
          exprnDefined = true;
      };

   };

       
   rateExprn = getRateExpression(exprn.first.first);

	if(const func_term * fexpression = dynamic_cast<const func_term *>(rateExprn))
	{
		/*fexp =*/ vld->fef.buildFuncExp(fexpression,*exprn.second);
	  }
	else if(const mul_expression * me = dynamic_cast<const mul_expression *>(rateExprn))
	{

         if(afe->appearsInEprsn(this,me->getLHS(),exprn.second))
         {
             constExprn = me->getRHS();//   fexp = afe->fe;
             FEExprn = me->getLHS();
         }


         else if(afe->appearsInEprsn(this,me->getRHS(),exprn.second))
         {
             constExprn = me->getLHS();//   fexp = afe->fe;
             FEExprn = me->getRHS();
         }
         else
         {
             DiffEqunError dee;
             throw dee;
         };

	}
	else
	{
		DiffEqunError dee;
		throw dee;
	};
         //end of extracting const expression and FE expression



     //const exprn depends on the FE?
		if(afe->appearsInEprsn(this,constExprn,exprn.second))
		{
			DiffEqunError dee;
			throw dee;
		};


    Polynomial bPoly;
    if(constExprn != 0)

    {                                    
      		 bPoly = getPoly(constExprn,this,exprn.second);        
    }
    else
             bPoly.setCoeff(0,1);

      if(!(exprn.first.second)) bPoly = - bPoly;

       		//if poly is constant zero then soln will be something...
    
      //handle different cases for FE expression, may be (m-y), (y-m) or y
      if(dynamic_cast<const func_term *>(FEExprn))
		{
       mValue = 0;   bPoly = - bPoly;
      }
      else if(const minus_expression * minexprn = dynamic_cast<const minus_expression *>(FEExprn))
      {



             if(dynamic_cast<const func_term *>(minexprn->getLHS()) && afe->appearsInEprsn(this,minexprn->getLHS(),exprn.second))
             {
                                 if(afe->appearsInEprsn(this,minexprn->getRHS(),exprn.second))
                            		{
                            			DiffEqunError dee;
                            			throw dee;
                            		};
                          
                           mValue = vld->getState().evaluate(minexprn->getRHS(),*exprn.second);
                           bPoly = - bPoly;
             }
             else if(dynamic_cast<const func_term *>(minexprn->getRHS()) && afe->appearsInEprsn(this,minexprn->getRHS(),exprn.second))
             {
                                 if(afe->appearsInEprsn(this,minexprn->getLHS(),exprn.second))
                            		{

                            			DiffEqunError dee;

                            			throw dee;
                            		};
                         
                           mValue = vld->getState().evaluate(minexprn->getLHS(),*exprn.second); 

                          
             }
              else
   			{
   				DiffEqunError dee;
   				throw dee;
   			};

      }
      else
			{
				DiffEqunError dee;
				throw dee;
			};

//      fexp = fexp;   // why was this line of code here?
      
    const BatteryCharge * newBatteryCharge = new BatteryCharge(new Polynomial(bPoly),mValue,discharge,0, localUpdateTime,afe->fe->evaluate(&(vld->getState())),accuracy);
               
	return newBatteryCharge;
 
};

FEScalar ActiveFE::evaluate(double time) const
{
    
	return ctsFtn->evaluate(time);
};

bool ActiveFE::isLinear() const

{
   if(ctsFtn->isLinear()) return true;
	return false;
};



bool ActiveCtsEffects::areCtsEffectsLinear() const
{
	//return true;	
	for(map<const FuncExp*,ActiveFE*>::const_iterator i = activeFEs.begin();i != activeFEs.end();++i)
	{
		//cout << i->second->fe<<" fe chking\n";
		if( !(i->second->parentFEs.empty()) ) return false;

	};

	return true;	
};

ActiveCtsEffects::~ActiveCtsEffects()
{
 
	ctsEffects.actions.clear();
   ctsUpdateHappening.actions.clear();
	
};

ActiveFE::~ActiveFE()
{

	parentFEs.clear();
	exprns.clear();
	delete ctsFtn;
};

bool ActiveCtsEffects::hasCtsEffects() const
{
	return !(ctsEffects.actions.empty());
};

void ActiveCtsEffects::clearCtsEffects()
{
	ctsEffects.actions.clear();
};

void EffectsRecord::enact(State * s) const
{
   if(!(s->getValidator()->hasEvents()) && !s->hasObservers())
   {
      	for(vector<const SimpleProposition *>::const_iterator i = dels.begin();i != dels.end();++i)

      	{
      		s->del(*i);
      	};
      	for(vector<const SimpleProposition *>::const_iterator i1 = adds.begin();i1 != adds.end();++i1)
      	{
      		s->add(*i1);
      	};
      	for(vector<Update>::const_iterator i2 = updates.begin();i2 != updates.end();++i2)
      	{
      		i2->update(s);
      	};
   }
   else

   {
      	s->recordResponsibles(responsibleForProps,responsibleForPNEs);
          	for(vector<const SimpleProposition *>::const_iterator i = dels.begin();i != dels.end();++i)
      	{
      		s->delChange(*i);
      	};
      	for(vector<const SimpleProposition *>::const_iterator i1 = adds.begin();i1 != adds.end();++i1)
      	{
      		s->addChange(*i1);
      	};
      	for(vector<Update>::const_iterator i2 = updates.begin();i2 != updates.end();++i2)
      	{
      		i2->updateChange(s);
      	};   
  };
//	for_each(dels.begin(),dels.end(),bind1st(ptr_fun(Deleter),s));
//	for_each(adds.begin(),adds.end(),bind1st(ptr_fun(Adder),s));
//	for_each(updates.begin(),updates.end(),bind1st(ptr_fun(Assigner),s));
};

/* Tricky bit: 
 * 	The effect_lists of each action will include universal effects, conditional 
 * 	effects, add and delete effects and assign effects (ignore timed effects for
 * 	the moment). Add and delete effects are easy. Universal effects must be 
 * 	instantiated for all possible values of the quantified variables - no way round

 * 	this I don't think. Conditional effects are more problematic: we have to first
 * 	confirm that the condition is satisfied in the current state, marking ownership
 * 	as well and then treat the effects using the
 * 	standard effect_lists handling machinery - use recursion here.
 * 	

 * 	At the end of the process we want all the postconditions to be tidily separated 
 * 	into add, delete and assign effects. Can't enact them as they are checked because
 * 	of conditional effects. Therefore, we need to build up the effects as we go along.
 *
 */ 

void insert_effects(effect_lists * el,effect_lists * more)
{
	el->add_effects.insert(el->add_effects.begin(),
					more->add_effects.begin(),more->add_effects.end());
	el->del_effects.insert(el->del_effects.begin(),
					more->del_effects.begin(),more->del_effects.end());	
	el->forall_effects.insert(el->forall_effects.begin(),
					more->forall_effects.begin(),more->forall_effects.end());					
	el->cond_effects.insert(el->cond_effects.begin(),
					more->cond_effects.begin(),more->cond_effects.end());
	el->assign_effects.insert(el->assign_effects.begin(),
					more->assign_effects.begin(),more->assign_effects.end());
	// Don't need to handle timed effects because this is used to insert effects
	// from one timed effects structure into another and they cannot be nested.
};


bool partOfPlan(const pair<double,Action *> & a)
{
	return a.second->isRealAction();
};

Plan::Plan(Validator * v,const operator_list * ops,const plan * p) :
	vld(v), timeToProduce(p->getTime())
{

  if(p->empty()) return;
	timedActionSeq planStructure;

	planStructure.reserve(p->size());

	for_each(p->begin(),p->end(),planBuilder(v,planStructure,ops));
	

	sort(planStructure.begin(),planStructure.end());
	double d = find_if(planStructure.rbegin(),planStructure.rend(),partOfPlan)->first; 
  //*report << "Calculated last useful time as " << d << "\n"; //you don't want this do you?
  timedActionSeq::iterator i = planStructure.begin();



  if(!Robust)
  {
    	while(i != planStructure.end())
    	{
    		timedActionSeq::iterator j
    				= find_if(i,planStructure.end(),after(i->first,v->getTolerance()));
    		timedActionSeq vs(i,j);
    		happenings.push_back(new Happening(vld,vs,d));
    		i = j;

    	};
  }
  else
  {
    while(i != planStructure.end())
    	{
    		timedActionSeq::iterator j
    				= find_if(i,planStructure.end(),sameTime(i->first));
    		timedActionSeq vs(i,j);
    		happenings.push_back(new Happening(vld,vs,d));
    		i = j;   
    	};   
  };
};

int Plan::length() const
{
	return happenings.size();
};

void Plan::display() const
{
	if(!LaTeX)

	{
		*report << "Plan size: " << length() << "\n";
		if(timeToProduce >= 0) *report << "Planner run time: " << timeToProduce << "\n";

	};
	
};

ostream & operator << (ostream & o,const Plan & p)
{
	p.display();
  if(p.length() == 0) return o;
	if(LaTeX)
	{

		Plan::const_iterator h = p.begin();
		double lastTime = h.getTime();
			
		o << "\\begin{tabbing}\n";
		o << "\\headingtimehappening \n";
		for(;h != p.end();++h)
		{
			if(lastTime != h.getTime()) o << "\\\\";
			o << "\\atime{"<< h.getTime() << "} ";
			o << *h << "\n";
			lastTime = h.getTime();
			


		};
		o << "\\end{tabbing}\n";
	}
	else
	{
		copy(p.begin(),p.end(),ostream_iterator<const Happening *>(o," \n"));
	};
	return o;

};

void Happening::write(ostream & o) const
{
	if(LaTeX)
	{
		
		
		for(vector<const Action*>::const_iterator a = actions.begin();a != actions.end();++a)
		{
			o << " \\> \\listrow{" << *a << "}\\\\";		
		};
		
	}
	else
	{
		o << time << ":\n";
		copy(actions.begin(),actions.end(),ostream_iterator<const Action * const>(o,"\n"));

	};
		
	
};


ostream & operator << (ostream & o,const Happening * h)
{
	h->write(o);
	return o;
};


void Update::update(State * s) const

{
	s->update(fe,aop,value);
};

void Update::updateChange(State * s) const
{
	s->updateChange(fe,aop,value);
};






void handleDAgoals(const goal * gl,goal_list * gls,goal_list * gli,goal_list * gle)
{
	if(const conj_goal * cg = dynamic_cast<const conj_goal *>(gl))
	{
		for(goal_list::const_iterator i = cg->getGoals()->begin();i != cg->getGoals()->end();++i)
		{
			if(const timed_goal * tg = dynamic_cast<const timed_goal *>(*i))
			{

				switch(tg->getTime())
				{
					case E_AT_START:
						gls->push_back(const_cast<goal*>(tg->getGoal()));
						continue;
					case E_AT_END:
						gle->push_back(const_cast<goal*>(tg->getGoal()));
						continue;
					case E_OVER_ALL:

						gli->push_back(const_cast<goal*>(tg->getGoal()));
						continue;


					default:
						continue;
				};
			}
			else
			{
				if(Verbose) *report << "Untimed precondition in a durative action!\n";
				UnrecognisedCondition uc;
				throw uc;
			};
		};
	}
	else
	{
		if(const timed_goal * tg = dynamic_cast<const timed_goal *>(gl))
			{
				switch(tg->getTime())
				{
					case E_AT_START:
						gls->push_back(const_cast<goal*>(tg->getGoal()));
						break;
					case E_AT_END:

						gle->push_back(const_cast<goal*>(tg->getGoal()));
						break;
					case E_OVER_ALL:
						gli->push_back(const_cast<goal*>(tg->getGoal()));
						break;

					default:
						break;
				};
			}

			else
			{
				if(Verbose) *report << "Untimed precondition in a durative action!\n";
				UnrecognisedCondition uc;
				throw uc;
			};

	};
};


void handleDAeffects(const effect_lists * efcts,effect_lists * els,effect_lists * ele,effect_lists * elc)
{
	for(pc_list<timed_effect*>::const_iterator i = efcts->timed_effects.begin();
			i != efcts->timed_effects.end();++i)
	{
		switch((*i)->ts)
		{
			case E_AT_START:
				insert_effects(els,(*i)->effs);
				continue;
			case E_AT_END:
				insert_effects(ele,(*i)->effs);
				continue;
			case E_CONTINUOUS:
		
				insert_effects(elc,(*i)->effs);
			
				continue;
			default:
				continue;
		};

	};

	//split forall effects into forall effects at the start 
	for(list<forall_effect*>::const_iterator i1 = efcts->forall_effects.begin();
			i1 != efcts->forall_effects.end();++i1)
	{

		effect_lists * elsfa = new effect_lists();
		effect_lists * elefa = new effect_lists();
		effect_lists * elcfa = new effect_lists(); 
		

		handleDAeffects((*i1)->getEffects(),elsfa,elefa,elcfa);


		if( !(elsfa->add_effects.empty() && elsfa->del_effects.empty() && elsfa->forall_effects.empty()
				&& elsfa->assign_effects.empty() ) )
			{
				els->forall_effects.push_back(new forall_effect(elsfa,const_cast<var_symbol_list*>((*i1)->getVarsList()),const_cast<var_symbol_table *>((*i1)->getVars()) ));
			};



		if( !(elcfa->add_effects.empty() && elcfa->del_effects.empty() && elcfa->forall_effects.empty()
				&& elcfa->assign_effects.empty() ) )
			{
				elc->forall_effects.push_back(new forall_effect(elcfa,const_cast<var_symbol_list*>((*i1)->getVarsList()),const_cast<var_symbol_table *>((*i1)->getVars()) ));
			};

		if( !(elefa->add_effects.empty() && elefa->del_effects.empty() && elefa->forall_effects.empty()
				&& elefa->assign_effects.empty() ) )
			{
				ele->forall_effects.push_back(new forall_effect(elefa,const_cast<var_symbol_list*>((*i1)->getVarsList()),const_cast<var_symbol_table *>((*i1)->getVars()) ));


			};
		
	};
};

struct handleDAConditionalEffects {

	Validator * vld;
	const durative_action * da;
	const const_symbol_list * params;
	effect_lists * els;
	effect_lists * ele;
	effect_lists * elc; 

	const var_symbol_list * vars;
	
	vector<const CondCommunicationAction *> condActions;
	vector<const CondCommunicationAction *> ctsCondActions;
	
	handleDAConditionalEffects(Validator * v,const durative_action * d,const const_symbol_list * ps,
										effect_lists * es,effect_lists * ee,effect_lists * ec) :
			vld(v), da(d), params(ps), els(es), ele(ee), elc(ec), vars(0)
	{};

	handleDAConditionalEffects(Validator * v,const durative_action * d,const const_symbol_list * ps,
										effect_lists * es,effect_lists * ee,effect_lists * ec,const var_symbol_list * vs) :
			vld(v), da(d), params(ps), els(es), ele(ee), elc(ec), vars(vs)
	{};

	void operator()(const forall_effect * fa)
	{
		if(!fa->getEffects()->cond_effects.empty())
		{
			effect_lists * lels = new effect_lists();
			effect_lists * lele = new effect_lists();
			effect_lists * lelc = 0; //cts effects - I don't believe this is ever used
			
			handleDAConditionalEffects hDAc
				= for_each(fa->getEffects()->cond_effects.begin(),fa->getEffects()->cond_effects.end(),
							handleDAConditionalEffects(vld,da,params,lels,lele,lelc,fa->getVarsList()));	
							
			if(!hDAc.ctsCondActions.empty())
			{
				cout << "Continuous effects spanning durative action, in quantified conditions...\n";
				cout << "This is really complex stuff! Unfortunately, VAL doesn't even try to deal with it.\n";
				cout << "Tell Derek to fix this!\n";
				SyntaxTooComplex stc;
				throw(stc);
			};

			if(!hDAc.condActions.empty())
			{
				condActions.insert(condActions.end(),hDAc.condActions.begin(),hDAc.condActions.end());
			}
	
// These use empty symbol tables, which will protect the variables from being
// deleted twice. 
			if(!lels->cond_effects.empty())
			{
			  forall_effect * fas = new forall_effect(lels,new var_symbol_list(*(fa->getVarsList())),new var_symbol_table ());
			  els->forall_effects.push_back(fas);
			}
			else
			  {
			    delete lels;
			  };
			if(!lele->cond_effects.empty())
			{
			  forall_effect * fae = new forall_effect(lele,new var_symbol_list(*(fa->getVarsList())),new var_symbol_table());
			  ele->forall_effects.push_back(fae);
			}
			else
			  {
			    delete lele;
			  };
		};
	};
	void operator()(const cond_effect * ce)
	{
		effect_lists * locels = new effect_lists();
		effect_lists * locele = new effect_lists();
		effect_lists * locelc = new effect_lists();
		
		handleDAeffects(ce->getEffects(),locels,locele,locelc);

		goal_list * gls = new goal_list();
		goal_list * gle = new goal_list();
		goal_list * gli = new goal_list();

		handleDAgoals(ce->getCondition(),gls,gli,gle);

		//check for bad conditional effects
		if( ( ! gli->empty() || ! gle->empty() ) &&
			( ! locels->add_effects.empty() || ! locels->del_effects.empty() || ! locels->forall_effects.empty() ||
				! locels->assign_effects.empty() 
				|| ! locelc->assign_effects.empty() || ! locelc->forall_effects.empty() ) )
		{
			TemporalDAError tdae;
			throw tdae;

		};
		

		if(!(locelc->assign_effects.empty() && locelc->forall_effects.empty())) //cts effects will always be assign effects or forall
		{
			
			ctsCondActions.push_back(new CondCommunicationAction(vld,da,params,gls,gli,gle,0,locelc));//cts cond actions

			if(locele->add_effects.empty() && locele->del_effects.empty() && locele->forall_effects.empty() 
				&& locele->assign_effects.empty() &&
				locels->add_effects.empty() && locels->del_effects.empty() && locels->forall_effects.empty() 
				&& locels->assign_effects.empty())
				{
					delete locels;
					delete locele;
				 	return;
				};
		}
		else
		{
			delete locelc;
			locelc = 0;
		};

			
		if(locele->add_effects.empty() && locele->del_effects.empty() && locele->forall_effects.empty() //only start effs and goals
				&& locele->assign_effects.empty() && gli->empty() && gle->empty())
		{
			//goal_list * gls = new goal_list();
			//handleDAgoals(ce->getCondition(),gls,0,0);
			cond_effect * nce = new cond_effect(new conj_goal(gls),locels);
			els->cond_effects.push_back(nce);
					
			delete locele;
			return;
		}
		else if(!(locels->add_effects.empty() && locels->del_effects.empty() && locels->forall_effects.empty() //if start effs and other effects
				&& locels->assign_effects.empty()))
		{
			cond_effect * nce = new cond_effect(new conj_goal(gls),locels);
			els->cond_effects.push_back(nce);
		};

		

		if(gls->empty() && gli->empty()) //only end cond and effs
		{
			delete gls;
			delete gli;
			cond_effect * nce = new cond_effect(new conj_goal(gle),locele);
			ele->cond_effects.push_back(nce);
			delete locels;
			return;
		};

		Environment bs = buildBindings(da,*params);
		if(vars)
		{
			buildForAllCondActions(vld,da,params,gls,gli,gle,locels,locele,vars,vars->begin(),condActions,&bs);
		}
		else
		{
			condActions.push_back(new CondCommunicationAction(vld,da,params,gls,gli,gle,locels,locele));
		}	
	};
};

void 
Plan::planBuilder::handleDurativeAction(const durative_action * da,const const_symbol_list * params,
												double start,double duration,const plan_step * ps)
{
	goal_list * gls = new goal_list();
	goal_list * gle = new goal_list();
	goal_list * gli = new goal_list();
	conj_goal * cgs = new conj_goal(gls);
	conj_goal * cge = new conj_goal(gle);
	conj_goal * inv = new conj_goal(gli);

	handleDAgoals(da->precondition,gls,gli,gle);

	effect_lists * els = new effect_lists();
	effect_lists * ele = new effect_lists();
	effect_lists * elc = new effect_lists(); //cts effects
	
	handleDAeffects(da->effects,els,ele,elc);
	handleDAConditionalEffects hDAc
		= for_each(da->effects->cond_effects.begin(),da->effects->cond_effects.end(),
				handleDAConditionalEffects(vld,da,params,els,ele,elc));

// Conditional effects can appear in quantified effects, too....
// What we should do is to instantiate the quantified variables in all possible ways 
// and then create a conditional effect for each one. We only need to do it for the 
// conditional effects that spread across the span of the action. Others can be 
// handled as standard quantified conditional effects.

	hDAc = for_each(da->effects->forall_effects.begin(),da->effects->forall_effects.end(),
				hDAc);
					
	goal_list * ds = new goal_list();
	goal_list * de = new goal_list();
	
	if(const conj_goal * cg = dynamic_cast<const conj_goal *>(da->dur_constraint))
	{
		for(goal_list::const_iterator i = cg->getGoals()->begin();
				i != cg->getGoals()->end(); ++i)
		{
			if(const timed_goal * tg = dynamic_cast<const timed_goal *>(*i))
			{
				switch(tg->getTime())
				{
					case E_AT_START:
						ds->push_back(const_cast<goal*>(tg->getGoal()));
						continue;
					case E_AT_END:
						de->push_back(const_cast<goal*>(tg->getGoal()));
						continue;

					default:
						continue;
				};
			};
		};

	}
	else
	{
		if(const timed_goal * tg = dynamic_cast<const timed_goal *>(da->dur_constraint))
		{
			switch(tg->getTime())
			{
				case E_AT_START:
					ds->push_back(const_cast<goal*>(tg->getGoal()));
					break;
				case E_AT_END:
					de->push_back(const_cast<goal*>(tg->getGoal()));
					break;

				default:
					break;
			};
		};
	};
	
	action * das = new safeaction(da->name,da->parameters,cgs,els,da->symtab);
	action * dae = new safeaction(da->name,da->parameters,cge,ele,da->symtab);
		// Note that we must use safeactions here, to ensure that the actions we create don't
		// think they own their components for deletion.

	StartAction * sa = new StartAction(vld,das,params,inv,elc,duration,ds,hDAc.condActions,hDAc.ctsCondActions,ps);
	tas.push_back(make_pair(start,sa));
	
	tas.push_back(make_pair(start+duration,new EndAction(vld,dae,params,sa,duration,de,ps)));
	
};

void Plan::planBuilder::operator()(const plan_step * ps)
{
	double t;
	if(ps->start_time_given)
	{
		t = ps->start_time;
	}
	else
	{
		t = defaultTime++;
	};

	for(operator_list::const_iterator i = ops->begin();i != ops->end();++i)
	{
		if((*i)->name->getName() == ps->op_sym->getName())
		{
			if(const action * a = dynamic_cast<const action *>(*i))
			{
				tas.push_back(make_pair(t,new Action(vld,a,ps->params,ps)));
				return;
			};

			if(const durative_action * da = dynamic_cast<const durative_action *>(*i))
			{
				handleDurativeAction(da,ps->params,t,ps->duration,ps);
				return;
			};

     
			if(Verbose) *report << "Unknown operator type in plan: " << (*i)->name->getName() << "\n";
			BadOperator bo;

			throw bo;
		};
	};

	if(Verbose) *report << "No matching action defined for " << ps->op_sym->getName() << "\n";

	BadOperator bo;
	throw bo;
};

Polynomial getPoly(const expression * e,const ActiveCtsEffects * ace,const Environment & bs,CoScalar endInt)
{
	if(const div_expression * fexpression = dynamic_cast<const div_expression *>(e))
	{
		//return getPoly(dynamic_cast<const div_expression*>(e)->getLHS(),ace,state) /
		//		getPoly(dynamic_cast<const div_expression*>(e)->getRHS(),ace,state);

		Polynomial numer =  getPoly(fexpression->getLHS(),ace,bs,endInt);
		Polynomial denom =  getPoly(fexpression->getRHS(),ace,bs,endInt);
		Polynomial poly;

		if(denom.getDegree() == 0)
		{
			poly = numer / denom.getCoeff(0); //beware if value is zero!
		}
		else
		{
			//throw not handled error or use quotient poly
			*report << "Quotient polynomials are not handled yet!\n";
			SyntaxTooComplex stc;
			throw stc;
		};

		return poly;
	};

	if(dynamic_cast<const minus_expression *>(e))
	{
		return getPoly(dynamic_cast<const minus_expression*>(e)->getLHS(),ace,bs,endInt) -
				getPoly(dynamic_cast<const minus_expression*>(e)->getRHS(),ace,bs,endInt);

	};

	if(dynamic_cast<const mul_expression *>(e))
	{
		return getPoly(dynamic_cast<const mul_expression*>(e)->getLHS(),ace,bs,endInt) *
				getPoly(dynamic_cast<const mul_expression*>(e)->getRHS(),ace,bs,endInt);
	};


	if(dynamic_cast<const plus_expression *>(e))
	{
		return getPoly(dynamic_cast<const plus_expression*>(e)->getLHS(),ace,bs,endInt) +
				getPoly(dynamic_cast<const plus_expression*>(e)->getRHS(),ace,bs,endInt);

	};

	if(dynamic_cast<const num_expression*>(e))
	{
		Polynomial constant;
		constant.setCoeff(0,dynamic_cast<const num_expression*>(e)->double_value());
		return constant;
	};

	if(dynamic_cast<const uminus_expression*>(e))
	{

		return -(getPoly(dynamic_cast<const uminus_expression*>(e)->getExpr(),ace,bs,endInt));
	};

	if(const func_term * fexpression = dynamic_cast<const func_term *>(e))
	{
		const FuncExp * fexp;


		fexp = ace->vld->fef.buildFuncExp(fexpression,bs);

		map<const FuncExp *,ActiveFE*>::const_iterator i = ace->activeFEs.find(fexp);


		if(i !=  ace->activeFEs.end())
		{
			if(i->second->ctsFtn != 0)
			{           
				if(dynamic_cast<const Polynomial *>(i->second->ctsFtn))
					return * dynamic_cast<const Polynomial *>(i->second->ctsFtn);
				else if(dynamic_cast<const Exponential *>(i->second->ctsFtn)) //use approx poly
				{               
					if(endInt != 0)
						return (dynamic_cast<const Exponential *>(i->second->ctsFtn))->getApproxPoly(endInt);
					else
					{         
						UndefinedPolyError upe;

						throw upe;
					};
				}
            else if(dynamic_cast<const NumericalSolution *>(i->second->ctsFtn))
            {
            	cout << *(i->second->ctsFtn) << "\n";
                 InvariantError  ie;
                 throw ie;
            };
        
			}
			else
			{                  
           
				UndefinedPolyError upe;
				throw upe;
			};
		}
		else

		{


			Polynomial constant;
			constant.setCoeff(0,fexp->evaluate(&(ace->vld->getState())));
			return constant;

		};
	};


	if(const special_val_expr * sp = dynamic_cast<const special_val_expr *>(e))
	{
		if(sp->getKind() == E_TOTAL_TIME)
		{
			Polynomial constant;

			if(ace->vld->durativePlan())
			{
				constant.setCoeff(0,ace->ctsEffects.getTime() );
			}
			else
			{
				constant.setCoeff(0,ace->vld->simpleLength());
			};

			return constant;


		};

		if(sp->getKind() == E_DURATION_VAR)

		{

			Polynomial constant;
			constant.setCoeff(0,bs.duration);
			return constant;
		};

		if(sp->getKind() == E_HASHT)
		{
			Polynomial timet;
			timet.setCoeff(1,1);
			return timet;
		}
	};

             
	UnrecognisedCondition uc;

	throw uc;
};




Polynomial getPoly(const expression * e,const ActiveCtsEffects * ace,const Environment * bs,CoScalar endInt)
{
	return getPoly(e,ace,*bs,endInt);

};


Polynomial getPoly(const expression * e,bool inc,const ActiveCtsEffects * ace,const Environment & bs,CoScalar endInt)
{
	if(inc) return getPoly(e,ace,bs,endInt);

	return - getPoly(e,ace,bs,endInt);

};

Polynomial getPoly(const expression * e,bool inc,const ActiveCtsEffects * ace,const Environment * bs,CoScalar endInt)
{
	return getPoly(e,inc,ace,*bs,endInt);
};

double Plan::timeOf(const Action * a) const
{
	for(HappeningSeq::const_iterator h = happenings.begin();h != happenings.end();++h)
	{
		if(find((*h)->getActions()->begin(),(*h)->getActions()->end(),a) !=
					(*h)->getActions()->end())
		{
			//cout << "Time of " << *a << " is " << (*h)->getTime() << "\n";
			return (*h)->getTime();
		};
	};
	return 0;
};

void Plan::show(ostream & o) const 
{
	for(HappeningSeq::const_iterator i = getFirstHappening();i != getEndHappening();++i)
	{
		o << *i << "\n";
	};
};

void Plan::addHappening(Happening * h)
{
	happenings.push_back(h);
};
};


