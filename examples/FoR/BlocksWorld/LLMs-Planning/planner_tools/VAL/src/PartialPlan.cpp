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

#include "PartialPlan.h"
#include "SearchSpace.h"
#include "graphconstruct.h"

using namespace VAL;

namespace Planner {


PartialPlan::PartialPlan() :
	current(SearchSpace::instance().getVal(),current_analysis->the_problem->initial_state), 
	planHead(SearchSpace::instance().getVal(),current_analysis->the_domain->ops,SearchSpace::instance().getDummyPlan()),
	currentStep(planHead.begin()), 
	possibleActions(SearchSpace::instance().getPG().applicableActions(SearchSpace::instance().getVal(),&current)),
	relevantActions(SearchSpace::instance().getPG().relevantActions(SearchSpace::instance().getVal(),&current)),
	waited(false), pending(0)
{};


void PartialPlan::selfSchedule(Inst::ActEntry * ae)
{
	scheduledActions.push_back(ae);

	cout << "REMEMBER TO REMOVE MUTEX ACTIONS WHEN SCHEDULING AN ACTION\n";
	
};

// Assume that ae is one of the possibleActions
PartialPlan * PartialPlan::schedule(Inst::ActEntry * ae)
{
	PartialPlan * pp = new PartialPlan(*this);
	
	pp->selfSchedule(ae);

	return pp;
};

void PartialPlan::selfCommit()
{
	for(vector<Inst::ActEntry *>::const_iterator i = scheduledActions.begin();
				i != scheduledActions.end();++i)
	{
		Action * a = new Action(SearchSpace::instance().getVal(),(*i)->getIO()->forOp(),(*i)->getIO()->getEnv()->getCore());

		pending->inject(a);
	};
	current.progress(*currentStep);
	waited = false;
	scheduledActions.clear();
	possibleActions = SearchSpace::instance().getPG().applicableActions(SearchSpace::instance().getVal(),&current);
	relevantActions = SearchSpace::instance().getPG().relevantActions(SearchSpace::instance().getVal(),&current);
	pending = 0;
};

// This commits all the scheduled actions to be executed
PartialPlan * PartialPlan::commit()
{
	PartialPlan * pp = new PartialPlan(*this);
	pp->selfCommit();
	return pp;
};

// This action waits for t time (where t is upper-bounded by a time proposed by
// OPT++). Should only be executed when no actions are secheduled.
void PartialPlan::selfWait(double t)
{		
	pending = new Happening(SearchSpace::instance().getVal(),current.getTime()+t,vector<pair<double,Action *> >());
	planHead.addHappening(pending);

	while(!(++currentStep).isRegular())
	{
		current.progress(*currentStep);
	};
	possibleActions = SearchSpace::instance().getPG().applicableActions(SearchSpace::instance().getVal(),&current);
	relevantActions = SearchSpace::instance().getPG().relevantActions(SearchSpace::instance().getVal(),&current);
};

PartialPlan * PartialPlan::wait(double t)
{
	PartialPlan * pp = new PartialPlan(*this);
	pp->selfWait(t);
	waited = true;
	return pp;
};

double PartialPlan::timeToTrigger()
{
	ActiveCtsEffects * ace = currentStep.getActiveCtsEffects();
	ace->buildAFECtsFtns();
	cout << "Here are the active functions\n";
	for(map<const FuncExp *,ActiveFE *>::const_iterator i = ace->activeFEs.begin();
				i != ace->activeFEs.end();++i)
	{
		cout << *(i->first) << " = " << *(i->second->ctsFtn) << "\n";
	};
	cout << "Here are the relevant processes/events/actions:\n";
	for(vector<Inst::ActEntry *>::const_iterator i = relevantActions.begin();
			i != relevantActions.end();++i)
	{
		cout << **i << "\n";
	};
	return 0.0;
};

void PartialPlan::initialWait(double t)
{
	pending = new Happening(SearchSpace::instance().getVal(),current.getTime()+t,vector<pair<double,Action *> >());
	planHead.addHappening(pending);
	currentStep = planHead.begin();
	//Verbose = true;
	SearchSpace::instance().getVal()->resetStep(currentStep);
	SearchSpace::instance().getVal()->getEvents().triggerInitialEvents(SearchSpace::instance().getVal(),t);
cout << "Now got APs: " << SearchSpace::instance().getVal()->getActiveCtsEffects()->ctsEffects << "\n";
	current = SearchSpace::instance().getVal()->getState();
	currentStep = SearchSpace::instance().getVal()->recoverStep();
	possibleActions = SearchSpace::instance().getPG().applicableActions(SearchSpace::instance().getVal(),&current);
	relevantActions = SearchSpace::instance().getPG().relevantActions(SearchSpace::instance().getVal(),&current);
};



void PartialPlan::write(ostream & o) const
{
	o << "[APs: ";
	for(map<const FuncExp *,ActiveFE *>::const_iterator i = currentStep.getActiveCtsEffects()->activeFEs.begin();
			i != currentStep.getActiveCtsEffects()->activeFEs.end();++i)
	{
		o << *(i->first) << " " << *(i->second->ctsFtn) << "\n";
	};

	o << currentStep.getActiveCtsEffects()->ctsEffects << "\nInvs: " <<
			currentStep.getExecutionContext()->invariants << "\nPlanHead: ";
	planHead.show(o);
	o << "\nState: " << current << "\nApplicable actions: ";
	for(vector<Inst::ActEntry *>::const_iterator i = possibleActions.begin();
			i != possibleActions.end();++i)
	{
		o << **i << " ";
	};
	o << "]\n\n";
};

bool LongPlanHead::operator()(const PartialPlan * pp1,const PartialPlan * pp2) const
{
	return pp1->length() > pp2->length();
};


};
