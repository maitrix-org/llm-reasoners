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

  $Date: 2009-02-05 10:50:12 $
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

 #include "Events.h"
 #include "Plan.h"
 #include "Validator.h"
 #include "Action.h"
 #include "RobustAnalyse.h"

namespace VAL {

Events::Events(const operator_list * ops) : ungroundEvents(), ungroundProcesses(),
  triggeredEvents(), triggeredProcesses(), untriggeredProcesses(), activeProcesses(), eventsForMutexCheck(), oldTriggeredEvents(), lastHappeningTime(0), ctsEventTriggered(false)  //, noevents(0)
{

   for(operator_list::const_iterator i = ops->begin();i != ops->end();++i)
	{
    if(event * e = dynamic_cast<event *>(*i))
	  {
          ungroundEvents.push_back(e);
    }
     else if(process * p = dynamic_cast<process *>(*i))
     {
          ungroundProcesses.push_back(p);
     };
    };


 };

Events::~Events()
{
    ungroundEvents.clear();
    triggeredEvents.clear();
    ungroundProcesses.clear();
    triggeredProcesses.clear();
    for(vector<const Action *>::iterator i = oldTriggeredEvents.begin(); i != oldTriggeredEvents.end(); ++i)
     delete *i;

    for(map<const StartAction *, pair<const Proposition *,const_symbol_list *> >::iterator j = activeProcesses.begin(); j != activeProcesses.end(); ++j)
          {j->second.first->destroy(); delete j->second.second;};

    activeProcesses.clear();
};

string Events::getName(operator_ * op,const_symbol_list * csl) const
{
  string eventName = "("+op->name->getName();
  for(const_symbol_list::iterator cs = csl->begin(); cs != csl->end(); ++cs)
  {
    eventName += " "+(*cs)->getName();
  };

  return eventName +")";

};

bool Events::isTriggered(event * eve,const_symbol_list * csl) const
{
    string  eventName = eve->name->getName();
    for(const_symbol_list::iterator cs = csl->begin(); cs != csl->end(); ++cs)
    {
      eventName += (*cs)->getName();
    };

    set<string>::const_iterator i = triggeredEvents.find(eventName);

    if(i != triggeredEvents.end())
    {
          eventName = "(" + eve->name->getName();
          for(const_symbol_list::iterator cs = csl->begin(); cs != csl->end(); ++cs)
          {
            eventName += " "+(*cs)->getName();
          };
          eventName += ")";

          if(LaTeX)
          {
            *report << "\\errorr{Attempt to trigger event \\exprn{"<<eventName<<"} twice}\\\\\n";
          }
          else if(Verbose) *report << "Attempt to trigger event "<<eventName<<" twice\n";

          return true;
     };

   return false;
};

bool Events::isTriggered(const Action * act) const
{

    string eventName = act->getName0();

    set<string>::const_iterator i = triggeredEvents.find(eventName);

    if(i != triggeredEvents.end())
    {
          if(LaTeX)
          {
            *report << "\\errorr{Attempt to trigger event \\exprn{"<<*act<<"} twice}\\\\\n";
          }
          else if(Verbose) *report << "Attempt to trigger event "<<*act<<" twice\n";

          return true;
     };

   return false;
};

bool Events::isProcessTriggered(const StartAction * sa) const
{
   set<const StartAction *>::iterator i = triggeredProcesses.find(sa);

   return (i != triggeredProcesses.end());
};

bool Events::isProcessActive(process * pro,const_symbol_list * csl) const
{
    string  processName = pro->name->getName();
    for(const_symbol_list::iterator cs = csl->begin(); cs != csl->end(); ++cs)
    {
      processName += (*cs)->getName();
    };

    for(map<const StartAction *, pair<const Proposition *,const_symbol_list *> >::const_iterator i = activeProcesses.begin(); i != activeProcesses.end(); ++i)
    {
        if(processName == i->first->getName0())
        {
          return true;
        };

    };

   return false;

};

bool Events::isProcessUntriggered(process * pro,const_symbol_list * csl) const
{
    string  processName = pro->name->getName();
    for(const_symbol_list::iterator cs = csl->begin(); cs != csl->end(); ++cs)
    {
      processName += (*cs)->getName();
    };

   set<string>::iterator i = untriggeredProcesses.find(processName);

   return (i != untriggeredProcesses.end());
};

void deleteParameters(vector<const_symbol_list*> & vcsl)
{
  for(vector<const_symbol_list*>::iterator p = vcsl.begin(); p != vcsl.end(); ++p)
                delete *p;
};

const vector<const_symbol_list*> removeRepeatedParameters(const vector<const_symbol_list*> & vcsl)
{
  vector<const_symbol_list*> listOfParameters;

  set<string> parameterLists;
  for(vector<const_symbol_list*>::const_iterator i = vcsl.begin(); i != vcsl.end(); ++i)
  {
     string eventName;
     for(const_symbol_list::iterator cs = (*i)->begin(); cs != (*i)->end(); ++cs)
     {
         eventName += (*cs)->getName();
     };

     set<string>::const_iterator pl = parameterLists.find(eventName);
     if(pl == parameterLists.end())
     {
        listOfParameters.push_back(*i);
        parameterLists.insert(eventName);
     }
     else
        delete *i;
  };

  return listOfParameters;
};

const_symbol_list * newBlankConstSymbolList(var_symbol_list * parameters,Validator * v)
{
   //create blank list of parameters, ie every parameter is undefined
   const_symbol_list * blankcsl = new const_symbol_list();
   for(var_symbol_list::const_iterator i = parameters->begin(); i != parameters->end(); ++i)
   {
    const_symbol * cs = 0; //v->getAnalysis()->const_tab.symbol_get("");
    blankcsl->push_back(cs);
   };

   return blankcsl;
};

const EndAction * addEndProcess(vector<const Action*> & processes,const StartAction * sa,const_symbol_list * csl,Validator * v)
{
   //create end action to end the process
    const operator_ * pro = sa->getAction();
    action * dae = new safeaction(pro->name,pro->parameters,new conj_goal(new goal_list()),new effect_lists(),pro->symtab);
    const EndAction * endAct = new EndAction(v,dae,csl,sa,0,new goal_list());
    processes.push_back(endAct);
    return endAct;
};

void Events::updateHappeningTime(double t)
{
   if(lastHappeningTime != t)
   {
      triggeredEvents.clear(); //to check for events being triggered twice at the same time point
      triggeredProcesses.clear(); //to ensure a process is not triggered and untriggered at same time point due to rounding errors
      untriggeredProcesses.clear(); //to ensure a processes is not untriggered and then retriggered at the same time point due to rounding when triggering processes by discrete change!
   };
   lastHappeningTime = t;

};

void Events::updateEventsForMutexCheck(Validator * v)
{
  eventsForMutexCheck.clear();
  vector<const_symbol_list*> listOfParameters0;
  vector<const_symbol_list*> listOfParameters00;
  set<var_symbol*> svs;
  vector<const_symbol_list*> listOfParameters;

  for(vector<event*>::iterator e = ungroundEvents.begin(); e != ungroundEvents.end(); ++e)
  {
       listOfParameters0 = getParametersCts((*e)->precondition,*e,v,false,true);
       listOfParameters00 = removeRepeatedParameters(listOfParameters0);
       svs = getVariables(*e);
       listOfParameters = defineUndefinedParameters(listOfParameters00,*e,v,svs);

        //add events to a event happening
       for(vector<const_symbol_list*>::iterator p = listOfParameters.begin(); p != listOfParameters.end(); ++p)
       {
              const Action * anAction = new Action(v,*e,*p);
              eventsForMutexCheck.push_back(anAction);
              oldTriggeredEvents.push_back(anAction);
       }; //end of looping thro' parameters

       deleteParameters(listOfParameters);

   };
};

//trigger processes and events before the first happening in the plan
bool Events::triggerInitialEvents(Validator * v,double firstHappeningTime)
{
  if(!hasEvents()) return true;

  bool isOK = true;
  ActiveCtsEffects * ace = v->getActiveCtsEffects();
  ace->clearCtsEffects();  //clear cts effect given by the first happening

  //check for events and processes in the intitial state
  isOK = triggerDiscreteEvents(v,true);

  //update values before the first happening of the plan!
  if(ace->hasCtsEffects() && (isOK || ContinueAnyway))  //if processes have been activated in the intitial state then there may be events to trigger and we need to update values before the first happening
  {
     ace->setTime(firstHappeningTime);
     ace->setLocalUpdateTime(firstHappeningTime);
     isOK = triggerEventsOnInterval(v,true) && isOK;

     if(isOK || ContinueAnyway)
     {
        //execute cts update happening
        if(firstHappeningTime != 0)
        {
          if(Verbose) *report << "\n";
          ace->setLocalUpdateTime(firstHappeningTime - v->getState().getTime());
          const Happening * updateHap = ace->getCtsEffectUpdate();
          updateHap->adjustActiveCtsEffects(*ace);
          isOK =  v->executeHappening(updateHap) && isOK;
          ace->ctsEffectsProcessed = false; ace->addActiveFEs();//update cts effects
         };
     };

  };

  v->adjustActiveCtsEffects(ace);
  return isOK;
};

bool Events::triggerEventsOnInterval(Validator * v,bool init)
{
  bool isOK = true;


  do
  {
     isOK = triggerDiscreteEvents(v,init) && isOK; if(!isOK && !ContinueAnyway) return false;

     isOK = triggerContinuousEvents(v,init) && isOK; if(!isOK && !ContinueAnyway) return false;

  }while(ctsEventTriggered);

  return isOK;
};

bool checkPreconditionsAreNotSatisfied(const State * s,const vector<const Action *> & events)
{
  for(vector<const Action *>::const_iterator i = events.begin(); i != events.end(); ++i)
  {
//  cout << "Checking a precondition for "<< *i << "\n";
    if((*i)->getPrecondition()->evaluate(s))
    {
      if(LaTeX) *report << "\\> Event "<<*i<<" does not falsify its precondition!\\\\\n";
      else if(Verbose) cout << "Event "<<*i<<" does not falsify its precondition!\n";

      return false;
    };


  };

  return true;
};

//trigger events given by discrete change
bool Events::triggerDiscreteEvents(Validator * v,bool init)
{
      if(EventPNEJuddering) JudderPNEs = true;
      double theCurrentTime = v->getState().getTime();
      updateHappeningTime(theCurrentTime);
      bool isOK = true;

      vector<const_symbol_list*> listOfParameters;
      int numberofEvents = 0;
      //trigger events: build up list of events to be triggered then add to a happening and execute the happening
      //then test for more events
      bool eventTriggered;
      do{
           eventTriggered = false;
           vector<const Action*> acts;

           for(vector<event*>::iterator e = ungroundEvents.begin(); e != ungroundEvents.end(); ++e)
           {

              if(!init || (theCurrentTime != 0))
               {
                   listOfParameters = getParametersDiscreteFinal((*e)->precondition,*e,v->getState());
                }
               else
               {
                  listOfParameters = getParametersDiscreteInitialFinal((*e)->precondition,*e,v);

                  if(!listOfParameters.empty())
                  {
                    *report << "Event preconditions are not permitted to be satisfied in the initial state!\n";
                    if(!ContinueAnyway) return false;
                    isOK = false;
                  };
               };

               //add events to a event happening and remove repetitions
               for(vector<const_symbol_list*>::iterator p = listOfParameters.begin(); p != listOfParameters.end(); ++p)
               {
                    if(isTriggered(*e,*p)) isOK = false;
                    else
                    {
                      const Action * anAction = new Action(v,*e,*p); //include precondition to check for mutex events? yes
                      acts.push_back(anAction);
                      triggeredEvents.insert(anAction->getName0());
                      oldTriggeredEvents.push_back(anAction);
                    };

               }; //end of looping thro' parameters

               deleteParameters(listOfParameters);

           };    //end of looping thro' events

             //apply events as a happening to the state
            if(acts.size() != 0 && (isOK || ContinueAnyway))
            {
               if(numberofEvents > 1)
               {
                 *report << "Event cascades of this nature are not yet handled robustly!\n";
                 if(!ContinueAnyway) return false;
                 isOK = false;
               };

               if(Verbose) cout << "\n";
               Happening * eventHappening = new Happening(v,acts,true);
               isOK = v->executeHappening(eventHappening) && isOK;
               isOK = checkPreconditionsAreNotSatisfied(&(v->getState()),acts) && isOK;
               eventHappening->clearActions(); delete eventHappening;
               if(!isOK && !ContinueAnyway) return false;
               eventTriggered = true;
               numberofEvents = acts.size();

             };

      isOK = triggerDiscreteProcesses(v) && isOK;

      }while(eventTriggered); //if events were triggered this may result in more events triggering

      eventsForMutexCheck.clear();


      if(EventPNEJuddering) JudderPNEs = false;
     return isOK;
};

const StartAction * newStartProcessAction(process* pro, const_symbol_list* csl,Validator * v)
{
  effect_lists * elc = new effect_lists();

	for(pc_list<timed_effect*>::const_iterator i = pro->effects->timed_effects.begin();
			i != pro->effects->timed_effects.end();++i)
	{
    if((*i)->ts == E_CONTINUOUS)
    {
				insert_effects(elc,(*i)->effs);
		};
	};

  //create a StartAction here
  action * das = new safeaction(pro->name,pro->parameters,new conj_goal(new goal_list()),new effect_lists(),pro->symtab);
  return new StartAction(v,das,csl,new conj_goal(new goal_list()),elc,0.0,new goal_list(),vector<const CondCommunicationAction *>(),vector<const CondCommunicationAction *>());

};

const EndAction * newEndProcess(const StartAction * sa,const_symbol_list * csl,Validator * v)
{
   //create end action to end the process
    const operator_ * pro = sa->getAction();
    action * dae = new safeaction(pro->name,pro->parameters,new conj_goal(new goal_list()),new effect_lists(),pro->symtab);
    const EndAction * endAct = new EndAction(v,dae,csl,sa,0,new goal_list());
    return endAct;
};

//trigger processes given by discrete change
bool Events::triggerDiscreteProcesses(Validator * v)
{
   if(EventPNEJuddering) JudderPNEs = true;
   bool isOK = true;
   vector<const Action*> processes;
   double time = v->getState().getTime();

   //find processes that are ended
   if(time > 0)
   {
	   vector<const StartAction *> eraseElements;
	   for(map<const StartAction *, pair<const Proposition *,const_symbol_list *> >::iterator ap = activeProcesses.begin();
	        ap != activeProcesses.end() ; ++ap)
	   {
	      const_cast<Proposition *>(ap->second.first)->resetCtsFunctions(); //ensure we are checking the precondition at a point not on an interval

	      if(!(ap->second.first->evaluate(&v->getState())) && !(isProcessTriggered(ap->first)))
	      {
	          const EndAction * endAct = addEndProcess(processes,ap->first,ap->second.second,v);
	          oldTriggeredEvents.push_back(endAct);
	          ap->second.first->destroy(); delete ap->second.second; eraseElements.push_back(ap->first);
	          untriggeredProcesses.insert(ap->first->getName0());
	      };
	   };

	   for(vector<const StartAction *>::iterator ee = eraseElements.begin(); ee != eraseElements.end(); ++ee)
	       activeProcesses.erase(*ee);
   };

   for(vector<process*>::const_iterator p = ungroundProcesses.begin(); p != ungroundProcesses.end(); ++p)
   {
       //for every process check if it could be triggered in the initial state or later on
       //get list of parameters  (may be empty)
    	vector<const_symbol_list*> listOfParameters;
       if(time == 0.0)
       {
       	listOfParameters = getParametersDiscreteInitialFinal((*p)->precondition,*p,v);
       }
       else
       {
       	listOfParameters = getParametersDiscreteFinal((*p)->precondition,*p,v->getState()); //need to check for all changes after the event cascade, this is the best way, only consider the process if something has changed
	   };
	//cout << "Process " << (*p)->name->getName() << " with " << listOfParameters.size() << "\n";

        //add processes to an event happening and remove repetitions
        for(vector<const_symbol_list*>::iterator pa = listOfParameters.begin(); pa != listOfParameters.end(); ++pa)
        {
          if(!isProcessActive(*p,*pa) /*&& !isProcessUntriggered(*p,*pa)*/)  //no need to check if it has not been untriggered, because above we now check that the process is triggered by something that has changed
          {
             const StartAction * sa = newStartProcessAction(*p,*pa,v);
             processes.push_back(sa);
             //only add the precondition to the list of triggered processes, as we already know it to be satisfied when applied
             const Proposition * prop = v->pf.buildProposition((*p)->precondition,*(buildBindings(*p,**pa).copy(v)));
             activeProcesses[sa] = make_pair(prop,*pa);
             oldTriggeredEvents.push_back(sa);
             triggeredProcesses.insert(sa);
             //cout << "Triggered " << *sa << "\n";
          }
          else
            delete *pa;

        }; //end of looping thro' parameters


   };    //end of looping thro' processes

   //apply processes as a happening to the state
   if(processes.size() != 0)
   {
    if(Verbose) cout << "\n";
    ActiveCtsEffects * ace = v->getActiveCtsEffects();
    Happening * eventHappening = new Happening(v,processes,true);
    eventHappening->adjustActiveCtsEffects(*ace);
    isOK = v->executeHappening(eventHappening);
    eventHappening->clearActions(); delete eventHappening;
    if(!isOK && !ContinueAnyway) return false;

   };

  if(EventPNEJuddering) JudderPNEs = false;
  return isOK;
};

const vector<const_symbol_list*> getParametersDiscreteInitialFinal(goal * g,operator_ * op,Validator * v)
{
   vector<const_symbol_list *> aParameterList;
   const_symbol_list * blankcsl = newBlankConstSymbolList(op->parameters,v);
   aParameterList.push_back(blankcsl);
   const vector<const_symbol_list*> listOfParameters0 = getParametersList(g,op,v,aParameterList,false,true);
   delete blankcsl;
   const vector<const_symbol_list*> listOfParameters00 = removeRepeatedParameters(listOfParameters0);
   const set<var_symbol*> svs = getVariables(op);
   return defineUndefinedParameters(listOfParameters00,op,v,svs);
};

const vector<const_symbol_list*> getParametersDiscreteFinal(goal * g,operator_ * op, const State & s)
{
   //get list of parameters  (may be empty)
   const vector<const_symbol_list*> listOfParameters0 = getParametersDiscrete(g,op,s);
   const vector<const_symbol_list*> listOfParameters00 = removeRepeatedParameters(listOfParameters0);
   const set<var_symbol*> svs = getVariables(op);
   return defineUndefinedParameters(listOfParameters00,op,s.getValidator(),svs);
};

const vector<const_symbol_list*> getParametersCtsFinal(goal * g,operator_ * op,Validator * v)
{
   const vector<const_symbol_list*> listOfParameters0 = getParametersCts(g,op,v,false,false);
   const vector<const_symbol_list*> listOfParameters00 = removeRepeatedParameters(listOfParameters0);
   const set<var_symbol*> svs = getVariables(op);
   return defineUndefinedParameters(listOfParameters00,op,v,svs);
};

//trigger events given by ctsly changing PNEs
bool Events::triggerContinuousEvents(Validator * v,bool init)
{
      if(!init && v->isLastHappening()) return true;
      if(EventPNEJuddering) JudderPNEs = true;
      ctsEventTriggered = false;
      ActiveCtsEffects * ace = v->getActiveCtsEffects();
      ExecutionContext * ec = v->getExecutionContext();
      vector<const_symbol_list*> listOfParameters;

      double currentTime = v->getState().getTime();
      double nextTime;
      if(!init) nextTime = v->getNextHappeningTime(); else nextTime = v->getCurrentHappeningTime();
      double localTime = nextTime - currentTime;
      if(localTime == 0) return true;

      bool isOK = true;
      vector<const Action*> acts;
      vector<const Action*> processes;

      ace->setLocalUpdateTime(localTime);  //set value of LocalUpdateTime - time since last regular happening

      //process cts effects, build polys etc, conditional cts effects handled here also

      if(TestingPNERobustness) ace->addActiveFEs(true);
      else ace->addActiveFEs();


      if(!ace->hasCtsEffects()) return true;

      //get parameters for all events that may be triggered due to cts effects
      for(vector<event*>::iterator e = ungroundEvents.begin(); e != ungroundEvents.end(); ++e)
      {
             listOfParameters = getParametersCtsFinal((*e)->precondition,*e,v);

             for(vector<const_symbol_list*>::iterator p = listOfParameters.begin(); p != listOfParameters.end(); ++p)
             {
                  const Action * anAction = new Action(v,*e,*p);
                  const_cast<Proposition*>(anAction->getPrecondition())->setUpComparisons(ace);
                  acts.push_back(anAction);
             };//end of loop thro' parameters

             deleteParameters(listOfParameters);
      };

      //find out when events are first triggered, if at all
      vector<pair<const Action *,intervalEnd> > ctsTriggeredEvents;
      Intervals eventIntervals;
      for(vector<const Action*>::iterator e = acts.begin(); e != acts.end(); ++e)
      {
         try
         {
           eventIntervals = (*e)->getPrecondition()->getIntervals(&(v->getState()));
         }
         catch(BadAccessError & e)
         {
           //if a PNE is not defined, then no problem, the event is simply not triggered
         };

         if(!eventIntervals.intervals.empty())
         {
             ctsTriggeredEvents.push_back(make_pair(*e, eventIntervals.intervals.begin()->first));//intervals are constructed left to right so this is the minimum point
         }
         else
             delete *e;

      };

      //get start time for processes to trigger - add to list of actions
      map<const StartAction *, pair<const Proposition *,const_symbol_list *> > startProcesses;
      for(vector<process*>::iterator pro = ungroundProcesses.begin(); pro != ungroundProcesses.end(); ++pro)
      {
             listOfParameters = getParametersCtsFinal((*pro)->precondition,*pro,v);

             for(vector<const_symbol_list*>::iterator p = listOfParameters.begin(); p != listOfParameters.end(); ++p)
             {
                if(!isProcessActive(*pro,*p) /*&& !isProcessUntriggered(*pro,*p)*/)
                {
                    //create a StartAction here
                    const StartAction * sa = newStartProcessAction(*pro,*p,v);
                    const Proposition * prop = v->pf.buildProposition((*pro)->precondition,*(buildBindings(*pro,**p).copy(v)));
                    const_cast<Proposition*>(prop)->setUpComparisons(ace);
                    eventIntervals = prop->getIntervals(&(v->getState()));

                    if(!eventIntervals.intervals.empty())
                    {
                        intervalEnd startTime = make_pair(-1,true);

                         //do not trigger a process on an isolated point
                        for(vector< pair<intervalEnd,intervalEnd> >::const_iterator ei = eventIntervals.intervals.begin(); ei != eventIntervals.intervals.end();++ei)
                        {
                          if(ei->first.first != ei->second.first)
                          {
                            startTime = ei->first; break;
                          };
                        };

                        if(startTime.first != -1)
                        {
                             ctsTriggeredEvents.push_back(make_pair(sa, startTime));//intervals are constructed left to right so this is the minimum point
                             startProcesses[sa] = make_pair(prop,new const_symbol_list(**p));
                        };
                    }
                    else
                    {
                        delete sa;
                        prop->destroy();
                    };
                };
             };

             deleteParameters(listOfParameters);
      };

       map<const EndAction *,const StartAction*> endStartMap;
       //find processes that could be ended
       for(map<const StartAction *, pair<const Proposition *,const_symbol_list *> >::iterator ap = activeProcesses.begin(); ap != activeProcesses.end() ; ++ap)
       {
              const_cast<Proposition*>(ap->second.first)->setUpComparisons(ace);
              Intervals eventIntervals0 = ap->second.first->getIntervals(&(v->getState()));
              eventIntervals = setComplement(eventIntervals0,localTime);
              if(!(eventIntervals.intervals.empty()))
              {
                  const EndAction * endAct = newEndProcess(ap->first,ap->second.second,v);
                  endStartMap[endAct] = ap->first;

                  intervalEnd startTime = make_pair(-1,true);

                   //do not untrigger a process on an isolated point
                  for(vector< pair<intervalEnd,intervalEnd> >::const_iterator ei = eventIntervals.intervals.begin(); ei != eventIntervals.intervals.end();++ei)
                  {
                    if(ei->first.first != ei->second.first)
                    {
                      startTime = ei->first; break;
                    };
                  };

                  if((startTime.first != -1) && isProcessTriggered(ap->first))
                  {
                    ctsTriggeredEvents.push_back(make_pair(endAct, startTime));
                  };
              };
       };

      vector<const Action*> eventsToTrigger;
      CoScalar eventTriggerTime = 0;
      //extract the first group of events to be triggered (including processes)
      for(vector<pair<const Action *,intervalEnd> >::iterator i = ctsTriggeredEvents.begin(); i != ctsTriggeredEvents.end(); ++i)
      {
        if(eventsToTrigger.empty())
        {
           eventTriggerTime =  i->second.first;
           eventsToTrigger.push_back(i->first);
        }
        else if(i->second.first == eventTriggerTime)
        {
           eventsToTrigger.push_back(i->first);
        }
        else if(i->second.first < eventTriggerTime)
        {
           eventTriggerTime =  i->second.first;
           for(vector<const Action*>::iterator k = eventsToTrigger.begin(); k != eventsToTrigger.end(); ++k) delete *k;
           eventsToTrigger.clear();
           eventsToTrigger.push_back(i->first);
        }
        else
           delete i->first;

      };

      //trigger events
      Happening * eventHappening;
      updateHappeningTime(eventTriggerTime + currentTime);
      bool triggerEventsExist = !eventsToTrigger.empty();

       //remember the no loops rule, i.e. an event cannot be triggered more than once at a given time point, only the same time point as last trigger
       //loop thro' events check if they have already been triggered (unless a process)
       if(triggerEventsExist)
       {
            for(vector<const Action*>::const_iterator j = eventsToTrigger.begin(); j != eventsToTrigger.end(); ++j)
              if(isTriggered(*j))
              {
                if(!ContinueAnyway) return false;
                isOK = false;
              };
       };

      if(triggerEventsExist && isOK)
      {
        ctsEventTriggered = true;

        ace->setTime(eventTriggerTime + currentTime); //same time as event trigger time
        ace->setLocalUpdateTime(eventTriggerTime);

        if(eventTriggerTime != 0)
        {
          //execute invariant happening
          ec->setActiveCtsEffects(ace);
          ec->setTime(eventTriggerTime + currentTime); //same time as event trigger time
          if(ec->hasInvariants()) {if(Verbose) cout << "\n"; isOK = v->executeHappening(ec->getInvariants()) && isOK;};
          ec->setTime(nextTime); //same time as event trigger time

          //execute cts update happening
          if(Verbose) cout << "\n";
          isOK =  v->executeHappening(ace->getCtsEffectUpdate()) && isOK;
        };

        //execute event happening
        if(Verbose) cout << "\n";
        eventHappening = new Happening(v,eventsToTrigger,eventTriggerTime + currentTime,true);
        eventHappening->adjustContext(*ec);
        eventHappening->adjustActiveCtsEffects(*ace); //update cts effects for interval after event happening
        isOK =  v->executeHappeningCtsEvent(eventHappening) && isOK;

        lastHappeningTime = eventTriggerTime + currentTime;
        vector<const StartAction *> triggeredStartProcesses;
        vector<const EndAction *> triggeredEndProcesses;
        vector<const Action *> events;

        //add to list of triggered events
        for(vector<const Action*>::const_iterator j = eventsToTrigger.begin(); j != eventsToTrigger.end(); ++j)
        {
           (*j)->addTriggeredEvents(events,oldTriggeredEvents,triggeredStartProcesses,triggeredEndProcesses);
        };

        //check triggered events have falsified their preconditions and add to list of triggered events
        isOK = checkPreconditionsAreNotSatisfied(&(v->getState()),events) && isOK;
        for(vector<const Action *>::const_iterator te = events.begin(); te != events.end(); ++te) triggeredEvents.insert((*te)->getName0());

        //add triggered processes to list of triggered processes
        for(vector<const StartAction *>::const_iterator trs = triggeredStartProcesses.begin(); trs != triggeredStartProcesses.end(); ++trs)
        {
            pair<const Proposition *,const_symbol_list *> propcsl = startProcesses[*trs];
            activeProcesses[*trs] = make_pair(propcsl.first,propcsl.second);
            triggeredProcesses.insert(*trs);
            startProcesses.erase(*trs);
        };

        //delete unused startProcesses' props and csls
        for(map<const StartAction *, pair<const Proposition *,const_symbol_list *> >::iterator sp = startProcesses.begin(); sp != startProcesses.end(); ++sp)
        {
            sp->second.first->destroy(); //delete sp->second.first;
        };

        //remove untriggered processes from list of triggered processes
        for(vector<const EndAction *>::const_iterator trs = triggeredEndProcesses.begin(); trs != triggeredEndProcesses.end(); ++trs)
        {
             const StartAction * stAction = endStartMap[*trs];
             map<const StartAction *, pair<const Proposition *,const_symbol_list *> >::iterator ap = activeProcesses.find(stAction);
             if(ap != activeProcesses.end())
             {
               ap->second.first->destroy(); delete ap->second.second;
               activeProcesses.erase(ap);
               untriggeredProcesses.insert(stAction->getName0());
             };

        };

        eventHappening->clearActions(); delete eventHappening;

        //ready for triggering discrete events
        ace->setTime(nextTime);
        ace->setEventTime(eventTriggerTime + currentTime);
      }
      else
        for(vector<const Action*>::iterator k = eventsToTrigger.begin(); k != eventsToTrigger.end(); ++k) delete *k;



      if(EventPNEJuddering) JudderPNEs = false;
      return isOK;
};

//for discretly triggered events
const vector<const_symbol_list*> getParametersDiscrete(goal * g,operator_ * op, const State & s,bool neg)
{
      vector<const_symbol_list*> listOfparameters;

   ///Comparison
   if(const comparison * comp = dynamic_cast<const comparison*>(g))
	{
      //get list of PNEs, then make list of parameter lists from them
      set<const func_term*> pnes;
      getPNEs(comp->getLHS(),pnes);
      getPNEs(comp->getRHS(),pnes);

      set<const FuncExp *> changedPNEs = s.getChangedPNEs();

      for(set<const func_term*>::iterator pne = pnes.begin(); pne != pnes.end(); ++pne)
      {
         //has pne changed since last happening
         for(set<const FuncExp *>::const_iterator fe = changedPNEs.begin(); fe != changedPNEs.end(); ++fe)
         {
             if(((*fe)->getName()  == (*pne)->getFunction()->getName())
                   && (*fe)->checkConstantsMatch((*pne)->getArgs())) //+check consts match!
             {
                 const_symbol_list* parametersForEvent = new const_symbol_list();

                  for(var_symbol_list::iterator j = op->parameters->begin(); j != op->parameters->end(); ++j)
                  {
                      string para = getParameter(*fe,*j,*pne);

                      const_symbol * cs = para==""?0:s.getValidator()->getAnalysis()->const_tab.symbol_get(para);

                     parametersForEvent->push_back(cs);

                   }; //end of filling in parameters

                  //ground out undefineds here to produce list of parameter lists
                  //check the comp holds for each parameter list! use only parameters used in comp
                  set<var_symbol*> svs = getVariables(comp);
                  listOfparameters = defineUndefinedParametersPropVar(parametersForEvent,op,s.getValidator(),g,false,neg,svs);

             };

         };//end of loop for checking if pne could be a changed fe

      };  //end of looping thro' pnes in expression

      return listOfparameters;
	};

  //Literal
	if(const simple_goal * sg = dynamic_cast<const simple_goal*>(g))
	{
		 //handle derived predicates first!
     string literalName = sg->getProp()->head->getName();
     vector<const_symbol_list*> aParameterList;
     if(s.getValidator()->getDerivRules()->isDerivedPred(literalName))
     {
        aParameterList.push_back(newBlankConstSymbolList(op->parameters,s.getValidator()));
        return getParametersList(g,op,s.getValidator(),aParameterList,neg,true);
     };

      //check if literal is one that has changed since last happening
      //if so fill in parameters with one from a simple prop
      set<const SimpleProposition *> changedLiterals = s.getChangedLiterals();
      for(set<const SimpleProposition *>::const_iterator sp = changedLiterals.begin(); sp != changedLiterals.end(); ++sp)
      {
         if( ((*sp)->getPropName()  == literalName) && ((!neg && s.evaluate(*sp)) || (neg && !s.evaluate(*sp)))
             && (*sp)->checkConstantsMatch(sg->getProp()->args) )
         {
           const_symbol_list* parametersForEvent = new const_symbol_list();

           //fill in parameters for event
           for(var_symbol_list::iterator j = op->parameters->begin(); j != op->parameters->end(); ++j)
           {
              string para = getParameter(*sp,*j,sg);
              const_symbol * cs = para==""?0:s.getValidator()->getAnalysis()->const_tab.symbol_get(para);

              parametersForEvent->push_back(cs);
            }; //end of filling in parameters

           listOfparameters.push_back(parametersForEvent);
         };

      };

    return listOfparameters;
  };

  //Negation
	if(dynamic_cast<const neg_goal *>(g))
	{
    return getParametersDiscrete(const_cast<goal*>(dynamic_cast<const neg_goal *>(g)->getGoal()),op,s,!neg);
	};

    //Other
    listOfparameters = getParameters(g,op,s.getValidator(),true,neg);

    return listOfparameters;
};

goal * newQfiedGoal(const qfied_goal * qg,operator_ * op,Validator * v)
{
     //get list of instansigated parameters for the qfied prop, so for forall(?z ?y), we have a grounded list for every z? and ?y
     set<var_symbol*> svs = getVariables(qg);
     vector<const_symbol_list*> constantsList = defineUndefinedParameters(newBlankConstSymbolList(const_cast<var_symbol_list*>(qg->getVars()),v),const_cast<var_symbol_list*>(qg->getVars()),v,svs);

     //now create a conjunction or disjunction with the qfied variables substituted
       map<parameter_symbol*,parameter_symbol*> newvars;
       goal_list* theGoals = new goal_list();

       for(vector<const_symbol_list*>::iterator k = constantsList.begin(); k != constantsList.end(); ++k)
       {
          const goal * aGoal = copyGoal(qg->getGoal());

          //define mapping of parameter symbol to constant
          const_symbol_list::iterator consList = (*k)->begin();
          for(var_symbol_list::const_iterator i = qg->getVars()->begin(); i != qg->getVars()->end(); ++i)
          {
             newvars[const_cast<var_symbol*>(*i)] = *consList;
             consList++;
           };
          //add conjunct/disjunct with qfied variables substituted
          changeVars(const_cast<goal *>(aGoal),newvars);
          theGoals->push_back(const_cast<goal *>(aGoal));
       };

     deleteParameters(constantsList);

     goal * goalToCheck;
     if(qg->getQuantifier()==E_FORALL)
     {
       goalToCheck = new conj_goal(theGoals);
     }
     else
     {
       goalToCheck = new disj_goal(theGoals);
     };

     return goalToCheck;
};

//for ctsly triggered events
const vector<const_symbol_list*> getParametersCts(goal * g,operator_ * op,Validator * v,bool neg,bool atAPoint)
{
    vector<const_symbol_list*> listOfparameters;
    vector<const_symbol_list*> alistOfparameters;

    ///Comparison
    if(const comparison * comp = dynamic_cast<const comparison*>(g))
	{
      //get list of PNEs, then make list of parameter lists from them
      set<const func_term*> pnes;
      getPNEs(comp->getLHS(),pnes);
      getPNEs(comp->getRHS(),pnes);

      for(set<const func_term*>::iterator pne = pnes.begin(); pne != pnes.end(); ++pne)
      {
         //loop thro' ctsly changing PNEs
         for(map<const FuncExp *, ActiveFE*>::const_iterator fe = v->getActiveCtsEffects()->activeFEs.begin(); fe != v->getActiveCtsEffects()->activeFEs.end(); ++fe)
         {
             if(fe->first->getName()  == (*pne)->getFunction()->getName()
                 && fe->first->checkConstantsMatch((*pne)->getArgs()))
             {
                 const_symbol_list* parametersForEvent = new const_symbol_list();

                  //fill in parameters for event
                  for(var_symbol_list::iterator j = op->parameters->begin(); j != op->parameters->end(); ++j)
                  {
                      string para = getParameter(fe->first,*j,*pne);
                      const_symbol * cs = para==""?0:v->getAnalysis()->const_tab.symbol_get(para);

                     parametersForEvent->push_back(cs);
                   }; //end of filling in parameters

                   //any undefined parameters will be ground out and checked later, unless we are checking at a point for mutex checks with actions at the same time.
                   if(!atAPoint) listOfparameters.push_back(parametersForEvent);
                   else
                   {
                     set<var_symbol*> svs = getVariables(comp);
                     alistOfparameters = defineUndefinedParametersPropVar(parametersForEvent,op,v,g,false,neg,svs,true);

                     for(vector<const_symbol_list*>::const_iterator l = alistOfparameters.begin(); l != alistOfparameters.end(); ++l)
                              listOfparameters.push_back(*l);
                   };

             };

         };//end of loop for checking if pne could be a changed fe

      };  //end of looping thro' pnes in expression

      return listOfparameters;
	};


  //Literal
	if(dynamic_cast<const simple_goal*>(g))
	{
       //return an empty list, literals do not change ctsly, so cannot trigger a event by cts activity
		   return listOfparameters;
  };

  //Negation
	if(dynamic_cast<const neg_goal *>(g))
	{
    return getParametersCts(const_cast<goal*>(dynamic_cast<const neg_goal *>(g)->getGoal()),op,v,!neg,atAPoint);
	};

    //Others
    listOfparameters = getParameters(g,op,v,false,neg,atAPoint);

    return listOfparameters;
};

//for both discrete and cts
const vector<const_symbol_list*> getParameters(goal * g,operator_ * op,Validator * v,bool discrete,bool neg,bool atAPoint)
{
    vector<const_symbol_list*> listOfparameters;

  ///Conjunction
	if((!neg && (dynamic_cast<const conj_goal *>(g))) || (neg && (dynamic_cast<const disj_goal *>(g))))
	{
        const goal_list* goalList;
        if(dynamic_cast<const conj_goal *>(g))
        {
           goalList =  dynamic_cast<const conj_goal *>(g)->getGoals();
        }
        else
        {
           goalList =  dynamic_cast<const disj_goal *>(g)->getGoals();
        };

        //loop thro' conjuncts
        //get all parameters from each
        for(pc_list<goal*>::const_iterator i = goalList->begin(); i != goalList->end(); ++i)
			  {

           vector<const_symbol_list*> someParameters;
           if(discrete) someParameters = getParametersDiscrete(*i,op,v->getState(),neg);
           else someParameters = getParametersCts(*i,op,v,neg,atAPoint);


           //whittle down the list of parameters depending if they satisfy the rest of the conjunction
           vector<const_symbol_list*> alistOfparameters;
           if(!someParameters.empty()) alistOfparameters = getParametersList(g,op,v,someParameters,neg,discrete,atAPoint);
             deleteParameters(someParameters);

             for(vector<const_symbol_list*>::const_iterator l = alistOfparameters.begin(); l != alistOfparameters.end(); ++l)
                listOfparameters.push_back(*l);


       };

       return listOfparameters;
	};

   ///Disjunction
	if((!neg && (dynamic_cast<const disj_goal *>(g))) || (neg && (dynamic_cast<const conj_goal *>(g))))
	{
        const goal_list* goalList;
        if(dynamic_cast<const disj_goal *>(g))
        {
           goalList =  dynamic_cast<const disj_goal *>(g)->getGoals();
        }
        else
        {
           goalList =  dynamic_cast<const conj_goal *>(g)->getGoals();
        };
         //loop thro' disjuncts
        for(pc_list<goal*>::const_iterator i = goalList->begin(); i != goalList->end(); ++i)
			{
              vector<const_symbol_list*> someParameters;
              if(discrete) someParameters = getParametersDiscrete(*i,op,v->getState(),neg);
              else someParameters = getParametersCts(*i,op,v,neg,atAPoint);

              for(vector<const_symbol_list*>::const_iterator l = someParameters.begin(); l != someParameters.end(); ++l)
                   listOfparameters.push_back(*l);
              //All remaining undefineds are given by existing objects and could be anything since
              //remaining disjuncts need not be satisfied - thus leave as undefined
			};

       return listOfparameters;
	};


	if(const qfied_goal* qg = dynamic_cast<const qfied_goal*>(g))
	{
     goal * goalToCheck = newQfiedGoal(qg,op,v);

     if(discrete) listOfparameters = getParametersDiscrete(goalToCheck,op,v->getState(),neg);
     else listOfparameters = getParametersCts(goalToCheck,op,v,neg,atAPoint);

     delete goalToCheck;

     return listOfparameters;
	};

	if(const imply_goal * ig = dynamic_cast<const imply_goal*>(g))
	{
		neg_goal * ng = new neg_goal(const_cast<goal *>(ig->getAntecedent()));
		goal_list * gl = new goal_list();;
		goal * agoal = new goal(*const_cast<goal *>(ig->getConsequent()));
		gl->push_back(ng);
		gl->push_back(agoal);
		disj_goal * goalToCheck = new disj_goal(gl);

    if(discrete) listOfparameters = getParametersDiscrete(goalToCheck,op,v->getState(),neg);
    else listOfparameters = getParametersCts(goalToCheck,op,v,neg,atAPoint);
    gl->clear();
    ng->destroy();
    delete agoal;
    delete goalToCheck;


    return listOfparameters;
	};

      return listOfparameters;
};

void unionVariables(set<var_symbol*> & svs1,set<var_symbol*> & svs2)
{

  for(set<var_symbol*>::const_iterator i = svs2.begin(); i != svs2.end(); ++i)
  {
       svs1.insert(*i);
  };

};

//does the goal contain a negation or a comparsion?
bool containsNegationComp(goal * g,bool neg)
{

  if(dynamic_cast<const comparison*>(g))
	{
    return true;
	};

	if(const conj_goal * cg = dynamic_cast<const conj_goal *>(g))
	{
        for(pc_list<goal*>::const_iterator i = cg->getGoals()->begin(); i != cg->getGoals()->end(); ++i)
			{
				if(containsNegationComp(*i,neg)) return true;
			};
      return false;
	};


	if(const disj_goal * dg = dynamic_cast<const disj_goal*>(g))
	{
        for(pc_list<goal*>::const_iterator i = dg->getGoals()->begin(); i != dg->getGoals()->end(); ++i)
			{
				if(containsNegationComp(*i,neg)) return true;
			};
       return false;
	};

	if(const neg_goal * ng = dynamic_cast<const neg_goal *>(g))
	{
		return containsNegationComp(const_cast<goal*>(ng->getGoal()),!neg);
	};


	if(const imply_goal * ig = dynamic_cast<const imply_goal*>(g))
	{

		return containsNegationComp(const_cast<goal*>(ig->getAntecedent()),neg) ||
		    containsNegationComp(const_cast<goal*>(ig->getConsequent()),neg);
	};

	if(dynamic_cast<const simple_goal*>(g))
	{
    return neg;
  };

	if(const qfied_goal * qg = dynamic_cast<const qfied_goal*>(g))
	{
      return containsNegationComp(const_cast<goal*>(qg->getGoal()),neg);
	};

  return false;
};

const vector<const_symbol_list*> getCopyCSL(const vector<const_symbol_list*> & lop)
{
 vector<const_symbol_list*> listOfparameters;

 for(vector<const_symbol_list*>::const_iterator p = lop.begin(); p != lop.end(); ++p)
 {
    listOfparameters.push_back(new const_symbol_list(**p));
 };

 return listOfparameters;
};

//find the list of parameters lists that satisfy g from the given list of parameter lists lop (psi as in paper)
const vector<const_symbol_list*> getParametersList(goal * g,operator_ * op,Validator * v,const vector<const_symbol_list*> & lop,bool neg,bool discrete,bool atAPoint)
{
    vector<const_symbol_list*> listOfparameters;

  ///Comparison
   if(const comparison * comp = dynamic_cast<const comparison*>(g))
	{
      //if checking for ctsly triggered events we need to assume any comps with ctsly changing pnes are true
      //at this stage. The comps are checked later, so do not do any checking here, it is done later (which will also include any comps with no ctsly changing PNEs that may be given here)
      if(!discrete && !atAPoint)
      {
         //get list of PNEs, then make list of parameter lists from them
         set<const func_term*> pnes;
         getPNEs(comp->getLHS(),pnes);
         getPNEs(comp->getRHS(),pnes);
         bool hasCtsPNE = false;

         for(set<const func_term*>::iterator pne = pnes.begin(); pne != pnes.end(); ++pne)
         {
            //loop thro' ctsly changing PNEs
            for(map<const FuncExp *, ActiveFE*>::const_iterator fe = v->getActiveCtsEffects()->activeFEs.begin(); fe != v->getActiveCtsEffects()->activeFEs.end(); ++fe)
            {
                if(fe->first->getName()  == (*pne)->getFunction()->getName())
                {
                   hasCtsPNE = true;
                };

            };//end of loop for checking if pne could be a changed fe

         };  //end of looping thro' pnes in expression

         //return copy of lop
         if(hasCtsPNE)
         {
           return getCopyCSL(lop);
         };
      };

      //get list of unique parameters from g
      set<var_symbol*> svs = getVariables(comp->getLHS());
      set<var_symbol*> svs2 =getVariables(comp->getRHS());
      unionVariables(svs,svs2);

      //dont want to delete lop
      vector<const_symbol_list*> someParameters;
      for(vector<const_symbol_list*>::const_iterator p = lop.begin(); p != lop.end(); ++p)
      {
          someParameters.push_back(new const_symbol_list(**p));
      };
      vector<const_symbol_list*> listOfparameters = defineUndefinedParametersPropVar(someParameters,op,v,g,false,neg,svs,atAPoint);

      return listOfparameters;
	};

  //Literal
	if(const simple_goal * sg = dynamic_cast<const simple_goal*>(g))
	{

      //look in state, look up all simple props that are true with this name. Get a list of constant symbols
      //corresponding to the true literals. Create a list of partially ground parameters from these and lop
      string literalName = sg->getProp()->head->getName();
      bool isDerivedPred = v->getDerivRules()->isDerivedPred(literalName);

      LogicalState logState = v->getState().getLogicalState();
      if(!neg && !isDerivedPred)
      {
        for(LogicalState::const_iterator i = logState.begin(); i != logState.end(); ++i)
        {
            if(i->second && (i->first->getPropName() == literalName))
            {
                if(i->first->checkParametersConstantsMatch(sg->getProp()->args))
                {
                    addToListOfParameters(listOfparameters,lop,i->first->getConstants(op->parameters,sg->getProp()->args,v));
                }
            };
        };
      }
      else
      {
         //old style method for negations and derived predicates
         set<var_symbol*> svs = getVariables(sg);
          vector<const_symbol_list*> someParameters;
          for(vector<const_symbol_list*>::const_iterator p = lop.begin(); p != lop.end(); ++p)
          {
              someParameters.push_back(new const_symbol_list(**p));
          };
         listOfparameters = defineUndefinedParametersPropVar(someParameters,op,v,g,isDerivedPred,neg,svs);

      };
      return listOfparameters;
  };

  //Negation
	if(dynamic_cast<const neg_goal *>(g))
	{
    return getParametersList(const_cast<goal*>(dynamic_cast<const neg_goal *>(g)->getGoal()),op,v,lop,!neg,discrete,atAPoint);
	};

  ///Conjunction
	if((!neg && (dynamic_cast<const conj_goal *>(g))) || (neg && (dynamic_cast<const disj_goal *>(g))))
	{
        const goal_list* goalList;
        if(dynamic_cast<const conj_goal *>(g))
        {
           goalList =  dynamic_cast<const conj_goal *>(g)->getGoals();
        }
        else
        {
           goalList =  dynamic_cast<const disj_goal *>(g)->getGoals();
        };

        if(goalList->empty()) return getCopyCSL(lop);

        vector<const_symbol_list*> aListOfparameters;
        vector<const_symbol_list*> aListOfparametersTemp;
        for(vector<const_symbol_list*>::const_iterator p = lop.begin(); p != lop.end(); ++p) aListOfparametersTemp.push_back(new const_symbol_list(**p));


        vector<goal*> negCompGoals; //do negative ones (and comparisons) later as these often cause a lot of work, try to reduce list first
        //loop thro' conjuncts
        for(pc_list<goal*>::const_iterator i = goalList->begin(); i != goalList->end(); ++i)
			  {
           if(containsNegationComp(*i,neg)) negCompGoals.push_back(*i);
           else
           {
             //get the list of parameters for each conjunct and then pass on list for next conjunct
             if(!aListOfparametersTemp.empty())
             {
               aListOfparameters = getParametersList(*i,op,v,aListOfparametersTemp,neg,discrete,atAPoint);//cout << " size of temp = "<<aListOfparameters.size()<<"\n";
               deleteParameters(aListOfparametersTemp);
               aListOfparametersTemp = aListOfparameters;
             };
           };
        };

        for(vector<goal*>::const_iterator j = negCompGoals.begin(); j != negCompGoals.end(); ++j)
        {
           if(!aListOfparametersTemp.empty())
           {
             aListOfparameters = getParametersList(*j,op,v,aListOfparametersTemp,neg,discrete,atAPoint);//cout << " size of temp = "<<aListOfparameters.size()<<"\n";
             deleteParameters(aListOfparametersTemp);
             aListOfparametersTemp = aListOfparameters;
           };
        };

        //maybe be some undefined parameters left over

       return aListOfparameters;
	};

   ///Disjunction
	if((!neg && (dynamic_cast<const disj_goal *>(g))) || (neg && (dynamic_cast<const conj_goal *>(g))))
	{
        const goal_list* goalList;
        if(dynamic_cast<const disj_goal *>(g))
        {
           goalList =  dynamic_cast<const disj_goal *>(g)->getGoals();
        }
        else
        {
           goalList =  dynamic_cast<const conj_goal *>(g)->getGoals();
        };
         //loop thro' disjuncts
        for(pc_list<goal*>::const_iterator i = goalList->begin(); i != goalList->end(); ++i)
			{
              vector<const_symbol_list*> someParameters;
              someParameters = getParametersList(*i,op,v,lop,neg,discrete,atAPoint);
              for(vector<const_symbol_list*>::iterator j = someParameters.begin(); j != someParameters.end(); ++j)
                 listOfparameters.push_back(*j);

              //All remaining undefineds are given by existing objects and could be anything since
              //remaining disjuncts need not be satisfied - thus leave as undefined
			};

       return listOfparameters;
	};


	if(const qfied_goal* qg = dynamic_cast<const qfied_goal*>(g))
	{
     goal * goalToCheck = newQfiedGoal(qg,op,v);

     listOfparameters = getParametersList(goalToCheck,op,v,lop,neg,discrete,atAPoint);

     delete goalToCheck;

     return listOfparameters;
	};

	if(const imply_goal * ig = dynamic_cast<const imply_goal*>(g))
	{
		neg_goal * ng = new neg_goal(const_cast<goal *>(ig->getAntecedent()));
		goal_list * gl = new goal_list();;
		goal * agoal = new goal(*const_cast<goal *>(ig->getConsequent()));
		gl->push_back(ng);
		gl->push_back(agoal);
		disj_goal * goalToCheck = new disj_goal(gl);

    listOfparameters = getParametersList(goalToCheck,op,v,lop,neg,discrete,atAPoint);
    gl->clear();
    ng->destroy();
    delete agoal;
    delete goalToCheck;

    return listOfparameters;
	};

      return listOfparameters;
};

 //find vs in the simple_goal parameters, map this to the parameter instantigation in sp
string getParameter(const SimpleProposition * sp,var_symbol * vs,const simple_goal * sg)
{
  string parameterName = "";
  int parameterNo = 1;
  parameter_symbol_list::iterator p = sg->getProp()->args->begin();
  for(; p != sg->getProp()->args->end(); ++p)
  {
      if(vs == *p) break;
      ++parameterNo;
  };

  if( p != sg->getProp()->args->end())
  {
     parameterName = sp->getParameter(parameterNo);
  };

  return parameterName;
};

//find vs in the func_term parameters, map this to the parameter instantigation in fe
string getParameter(const FuncExp * fe,var_symbol * vs,const func_term * pne)
{
  string parameterName = "";
  int parameterNo = 1;
  parameter_symbol_list * psl =  const_cast<parameter_symbol_list*>(pne->getArgs());
  parameter_symbol_list::iterator p = psl->begin();
  for(; p != psl->end(); ++p)
  {
      if(vs == *p) break;
      ++parameterNo;
  };

  if( p != psl->end())
  {
     parameterName = fe->getParameter(parameterNo);
  };

  return parameterName;
};

bool undefinedParameterExists(const_symbol_list * csl)
{

     for(const_symbol_list::iterator cs =  csl->begin(); cs != csl->end(); ++cs)
     {
       if(!(*cs)) return true;
     };

     return false;
};

const set<var_symbol*> getVariables(const expression * e)
{
    set<var_symbol*> theVariables;

    if(dynamic_cast<const binary_expression*>(e))
	{
    const binary_expression * be = dynamic_cast<const binary_expression*>(e);
    theVariables = getVariables(be->getLHS());
    set<var_symbol*> svs2 =getVariables(be->getRHS());
    unionVariables(theVariables,svs2);
    return theVariables;
	};

  if(dynamic_cast<const uminus_expression*>(e))
	{
    const uminus_expression * ue = dynamic_cast<const uminus_expression*>(e);
    theVariables = getVariables(ue->getExpr());
    return theVariables;
	};

    if(const func_term * fe = dynamic_cast<const func_term*>(e))
	{
      parameter_symbol_list *param_list =   const_cast<parameter_symbol_list*>(fe->getArgs());

      for(parameter_symbol_list::iterator i = param_list->begin(); i != param_list->end();++i)
   	  {
          if(dynamic_cast<const var_symbol *>(*i))
   		  {
               theVariables.insert(const_cast<var_symbol*>(dynamic_cast<const var_symbol *>(*i)));
        };
       };
	};

  return theVariables;
};

const set<var_symbol*> getVariables(const simple_goal * sg)
{
    set<var_symbol*> theVariables;

    for(parameter_symbol_list::iterator i = sg->getProp()->args->begin(); i != sg->getProp()->args->end();++i)
   	  {
      if(dynamic_cast<const var_symbol *>(*i))
   		  {
               theVariables.insert(const_cast<var_symbol*>(dynamic_cast<const var_symbol *>(*i)));
           };
      };

    return theVariables;
};

const set<var_symbol*> getVariables(const qfied_goal * qg)
{
    set<var_symbol*> theVariables;

    for(var_symbol_list::const_iterator i = qg->getVars()->begin(); i != qg->getVars()->end();++i)
   	  {
         theVariables.insert(const_cast<var_symbol*>(*i));
      };

    return theVariables;
};

const set<var_symbol*> getVariables(const operator_ * op)
{
    set<var_symbol*> theVariables;

    for(var_symbol_list::const_iterator i = op->parameters->begin(); i != op->parameters->end();++i)
   	  {
          theVariables.insert(*i);
      };

    return theVariables;
};

//fill in parameters in lop with those in csl and return results, discard csl also
void addToListOfParameters(vector<const_symbol_list*> & vcsl,const vector<const_symbol_list*> & lop,const_symbol_list * csl)
{

  bool satisfied;

  for(vector<const_symbol_list*>::const_iterator i = lop.begin(); i != lop.end(); ++i)
  {
       const_symbol_list * aConstList = new const_symbol_list(**i);

       satisfied = true;
       const_symbol_list::const_iterator j = csl->begin();
       const_symbol_list::iterator l = aConstList->begin();
       for(const_symbol_list::const_iterator k = (*i)->begin(); k != (*i)->end(); ++k)
       {
        if((*j))
        {
            if(!(*k))
                 *l = *j;
             else if(*l != *j)
                 satisfied = false; //the literal(or other) is not satisfied by previously defined parameters so not add to list of parameters
        };

        ++j; ++l;
       };

       if(satisfied) vcsl.push_back(aConstList);
       else delete aConstList;


  };

  delete csl;

      //test
 /*      cout << " results\n";
       for(vector<const_symbol_list*>::const_iterator k = vcsl.begin(); k != vcsl.end() ; ++k)
       {
          for(const_symbol_list::iterator l = (*k)->begin(); l != (*k)->end(); ++l)
          {
            cout << (*l)->getName();

          };
          cout <<"\n";
       }; */
        //end test


};

//for function below
bool isInList(const set<var_symbol*> & svs, var_symbol * vs)
{
    set<var_symbol*>::const_iterator i = svs.find(vs);
    return (i != svs.end());
};

//returns new set of parameter lists based on csl, only defines undefined variables if they are in svs
const vector<const_symbol_list*> defineUndefinedParameters(const_symbol_list * csl,var_symbol_list* variables,Validator * vld,const set<var_symbol*> & svs)
{
 map<unsigned int,const_symbol_list*> definedLists;
 vector<const_symbol_list*> finalDefinedLists;

 if(csl->size() == 0 || !undefinedParameterExists(csl))
 {
   finalDefinedLists.push_back(new const_symbol_list(*csl)); delete csl;

   return finalDefinedLists;
 };

 definedLists[1] = new const_symbol_list(*csl);
 const_symbol_list* considerList = definedLists[1];
 unsigned int definedListNoConsider = 0;
 unsigned int definedListNoAdd = 1;

 //loop thro definedLists until every parameter is ground, adding to definedLists when grounding a variable and other undefineds remain

 int paraNumber;
 int paraNumber0;
 bool remainingParasDefined;

 do{
   ++definedListNoConsider;
   map<unsigned int,const_symbol_list*>::iterator dl = definedLists.find(definedListNoConsider);
   if(dl == definedLists.end()) break;
   considerList = dl->second;

   //choose the first undefined parameter
   const_symbol_list::iterator cs =  considerList->begin();
   var_symbol_list::iterator vs = variables->begin(); paraNumber = 1;
   for(; cs != considerList->end(); ++cs)
   {
     if(!(*cs) && isInList(svs,*vs)) break;
     ++vs; ++paraNumber;
   };

   if(cs == considerList->end()) {delete considerList; break;};

   //are remaining parameters defined in list?
   remainingParasDefined = true; var_symbol_list::iterator vs0 = variables->begin(); paraNumber0 = 1;
   for(const_symbol_list::iterator cs0 =  considerList->begin(); cs0 != considerList->end(); ++cs0)
   {
     if((!(*cs0)) && isInList(svs,*vs0) && (paraNumber0 > paraNumber)) remainingParasDefined = false;
     ++vs0; ++paraNumber0;
   };

   //define the parameter for every possibility and add to list of definedLists
  //cout << "Seeking range for " << **vs << "\n";
   vector<const_symbol *> vals = vld->range(*vs);
  //cout << "Got\n";
   for(vector<const_symbol *>::const_iterator obj = vals.begin(); obj != vals.end(); ++obj)
   {
      const_symbol_list* parameters = new const_symbol_list(*considerList);

      const_symbol_list::iterator j =  parameters->begin();
      for(int k = 1 ; k < paraNumber; ++k) {++j;};
      *j = *obj;

      if(!remainingParasDefined) definedLists[++definedListNoAdd] = parameters;
      else finalDefinedLists.push_back(parameters);
   };

   delete considerList;  //delete each consider list
 }while(true);//end of looping thro defined lists, for given parameter, move to next parameter if considered parameter set was last in list when chosen

 /*
 if(finalDefinedLists.empty())
 {
   finalDefinedLists.push_back(new const_symbol_list(*csl));
 };
 */
       //test
     /*  cout << finalDefinedLists.size() << " results\n";

       for(vector<const_symbol_list*>::const_iterator k = finalDefinedLists.begin(); k != finalDefinedLists.end() ; ++k)
       {
          for(const_symbol_list::iterator l = (*k)->begin(); l != (*k)->end(); ++l)
          {
            cout << (*l)->getName();

          };
          cout <<"\n";
       };  //end test
       */

 delete csl;
 return finalDefinedLists;
};


const vector<const_symbol_list*> defineUndefinedParameters(const vector<const_symbol_list*> & vcsl,operator_ * op,Validator * vld,const set<var_symbol*> & svs)
{
  vector<const_symbol_list*> definedLists;
  vector<const_symbol_list*> someDefinedLists;

  for(vector<const_symbol_list*>::const_iterator csl = vcsl.begin(); csl != vcsl.end(); ++csl)
  {
     someDefinedLists = defineUndefinedParameters(*csl,op->parameters,vld,svs);

     for(vector<const_symbol_list*>::const_iterator i = someDefinedLists.begin(); i != someDefinedLists.end(); ++i)
        definedLists.push_back(*i);

  };

  return definedLists;
};

//return parameter lists such that they satisfy the goal g, also delete reject candidate parameter lists!
const vector<const_symbol_list*> checkParametersProp(const vector<const_symbol_list*> & vcsl,operator_ * op,Validator * vld,goal * g,bool neg,bool dp,bool compError)
{
   vector<const_symbol_list*> definedLists;

   for(vector<const_symbol_list*>::const_iterator gp = vcsl.begin(); gp != vcsl.end(); ++gp)
   {
          bool propSatisfied;
          if(dynamic_cast<const simple_goal*>(g) && !dp)
          {
            const Proposition * prop = vld->pf.buildProposition(g,*(buildBindings(op,**gp).copy(vld)));
            propSatisfied = prop->evaluate(&vld->getState());
            prop->destroy();
          }
          else
          {
            const Proposition * prop = vld->pf.buildProposition(g,*(buildBindings(op,**gp).copy(vld)));
            try
            {
              if(compError) propSatisfied = prop->evaluateAtPointWithinError(&vld->getState());
              else propSatisfied = prop->evaluate(&vld->getState());
            }
            catch(BadAccessError & e)
            {
              if(!neg) propSatisfied = false;
              else propSatisfied = true; //if a PNE is not defined, then no problem, the event is simply not triggered
            };
            prop->destroy();
          };

          if((propSatisfied && !neg) || (!propSatisfied && neg) )
          {
              definedLists.push_back(*gp);
          }
          else
              delete *gp;

   };

   return definedLists;
};


//define undefined parameters in csl (but only if in svs!) and ensure they satisfy the goal of the event
const vector<const_symbol_list*> defineUndefinedParametersPropVar(const_symbol_list * csl,operator_ * op,Validator * vld,goal * g,bool dp,bool neg,const set<var_symbol*> & svs,bool compError)
{
   vector<const_symbol_list*> groundListOfParameters;

   //if there are variables to ground then ground 'em!
   if(!svs.empty()) groundListOfParameters = defineUndefinedParameters(csl,op->parameters,vld,svs);
   else
   {
     groundListOfParameters.push_back(new const_symbol_list(*csl));
     delete csl;
   };

   return checkParametersProp(groundListOfParameters,op,vld,g,neg,dp,compError);
};

const vector<const_symbol_list*> defineUndefinedParametersPropVar(const vector<const_symbol_list*> & vcsl,operator_ * op,Validator * vld,goal * g,bool dp,bool neg,const set<var_symbol*> & svs,bool compError)
{
  vector<const_symbol_list*> definedLists;
  vector<const_symbol_list*> someDefinedLists;

  for(vector<const_symbol_list*>::const_iterator csl = vcsl.begin(); csl != vcsl.end(); ++csl)
  {
     someDefinedLists = defineUndefinedParametersPropVar(*csl,op,vld,g,dp,neg,svs,compError);

     for(vector<const_symbol_list*>::const_iterator i = someDefinedLists.begin(); i != someDefinedLists.end(); ++i)
        definedLists.push_back(*i);
  };

  return definedLists;
};

//get pnes in expression e
void getPNEs(const expression * e,set<const func_term*> & pnes)
{

	if(const func_term * fexpression = dynamic_cast<const func_term *>(e))
	{
      pnes.insert(fexpression);
      return;
	};


	if(const binary_expression * bexp = dynamic_cast<const binary_expression *>(e))
	{
      getPNEs(bexp->getLHS(),pnes);
      getPNEs(bexp->getRHS(),pnes);
      return;
	};

	if(const uminus_expression * uexp = dynamic_cast<const uminus_expression *>(e))
	{
      getPNEs(uexp->getExpr(),pnes);
	};


};

//copy a goal including all subgoals
const goal * copyGoal(const goal * g)
{

	if(dynamic_cast<const neg_goal *>(g))
	{
		const goal * ng = (dynamic_cast<const neg_goal *>(g))->getGoal();

    return new neg_goal(const_cast<goal*>(copyGoal(ng)));
  };

	if(dynamic_cast<const imply_goal*>(g))
	{
		const imply_goal * ig = dynamic_cast<const imply_goal*>(g);

		neg_goal * ng = new neg_goal(const_cast<goal *>(copyGoal(ig->getAntecedent())));
		goal_list * gl = new goal_list();;
		goal * agoal = const_cast<goal*>(copyGoal(ig->getConsequent()));
		gl->push_back(ng);
		gl->push_back(agoal);
		const disj_goal * dg = new disj_goal(gl);

		return copyGoal(dg);
	};

	if(dynamic_cast<const conj_goal *>(g))
	{
		const conj_goal * cg = dynamic_cast<const conj_goal *>(g);
		goal_list * gl = new goal_list();
		for(pc_list<goal*>::const_iterator i = cg->getGoals()->begin(); i != cg->getGoals()->end(); ++i)
		{
			gl->push_back(const_cast<goal*>(copyGoal(*i)));
		};
		return new conj_goal(gl);
	};


	if(dynamic_cast<const disj_goal*>(g))
	{
		const disj_goal * dg = dynamic_cast<const disj_goal*>(g);
		goal_list * gl = new goal_list();
		for(pc_list<goal*>::const_iterator i = dg->getGoals()->begin(); i != dg->getGoals()->end(); ++i)
		{
				gl->push_back(const_cast<goal*>(copyGoal(*i)));
		};
    const disj_goal * d_goal = new disj_goal(gl);

		return d_goal;
	};


	if(dynamic_cast<const qfied_goal*>(g))
	{
		const qfied_goal * qg = dynamic_cast<const qfied_goal*>(g);

		const qfied_goal * ans;
    var_symbol_table * sym_tab = new var_symbol_table();
    for(map<string,var_symbol*>::const_iterator i = qg->getSymTab()->begin(); i != qg->getSymTab()->end(); ++i)
    {
       sym_tab->symbol_put(i->first);
    };

		if(qg->getQuantifier() == E_EXISTS)
		 	ans = new qfied_goal(E_EXISTS,new var_symbol_list(*const_cast<var_symbol_list*>(qg->getVars())),
		 				const_cast<goal*>(copyGoal(const_cast<goal*>(qg->getGoal()))),sym_tab);
		else
			ans = new qfied_goal(E_FORALL,new var_symbol_list(*const_cast<var_symbol_list*>(qg->getVars())),
						const_cast<goal*>(copyGoal(const_cast<goal*>(qg->getGoal()))),sym_tab);

		return ans;
	};

	if(dynamic_cast<const simple_goal*>(g))
	{
    const simple_goal * sg = dynamic_cast<const simple_goal*>(g);
    parameter_symbol_list* newpsl = new parameter_symbol_list();
    for(parameter_symbol_list::iterator i = sg->getProp()->args->begin();i != sg->getProp()->args->end();++i)
    {
       newpsl->push_back(*i);
    };
    const proposition * prop = new proposition(sg->getProp()->head,newpsl);
    const simple_goal * sim_goal = new simple_goal(const_cast<proposition*>(prop),sg->getPolarity());

		return sim_goal;
	};

   if(dynamic_cast<const comparison*>(g))
	{
     const comparison * comp = dynamic_cast<const comparison*>(g);
     const expression * e1 = copyExpression(comp->getLHS()); //copy
     const expression * e2 = copyExpression(comp->getRHS());

     return new comparison(comp->getOp(),const_cast<expression*>(e1),const_cast<expression*>(e2));
   };

	return g;
};

const expression * copyExpression(const expression * e)
{

  if(const div_expression * fexpression = dynamic_cast<const div_expression *>(e))
	{
		return new div_expression(const_cast<expression*>(copyExpression(fexpression->getLHS())),const_cast<expression*>(copyExpression(fexpression->getRHS())));
	};

	if(const minus_expression * fexpression = dynamic_cast<const minus_expression *>(e))
	{
		return new minus_expression(const_cast<expression*>(copyExpression(fexpression->getLHS())),const_cast<expression*>(copyExpression(fexpression->getRHS())));
	};

	if(const mul_expression * fexpression = dynamic_cast<const mul_expression *>(e))
	{
     return new mul_expression(const_cast<expression*>(copyExpression(fexpression->getLHS())),const_cast<expression*>(copyExpression(fexpression->getRHS())));
	};

	if(const plus_expression * fexpression = dynamic_cast<const plus_expression *>(e))
	{
		return new plus_expression(const_cast<expression*>(copyExpression(fexpression->getLHS())),const_cast<expression*>(copyExpression(fexpression->getRHS())));
	};

	if(const int_expression * fexpression = dynamic_cast<const int_expression*>(e))
	{
    int aNumber = int(fexpression->double_value());
		return new int_expression(aNumber);
	};

	if(const float_expression * fexpression = dynamic_cast<const float_expression*>(e))
	{
    NumScalar aNumber = fexpression->double_value();
		return new float_expression(aNumber);
	};

	if(const uminus_expression * fexpression = dynamic_cast<const uminus_expression*>(e))
	{
		return new uminus_expression(const_cast<expression*>(copyExpression(fexpression->getExpr())));
	};

	if(const func_term * fexpression = dynamic_cast<const func_term *>(e))
	{
    parameter_symbol_list* newpsl = new parameter_symbol_list();
    for(parameter_symbol_list::const_iterator i = fexpression->getArgs()->begin();i != fexpression->getArgs()->end();++i)
    {
       newpsl->push_back(*i);
    };

     return new func_term(const_cast<func_symbol*>(fexpression->getFunction()),newpsl);
  };

  if(dynamic_cast<const special_val_expr *>(e))
	{
      return e;
	};

  return e;
};

};
