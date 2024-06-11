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

#ifndef __PARTIALPLAN
#define __PARTIALPLAN

#include "State.h"
#include "Plan.h"
#include "Validator.h"
#include <iostream>


namespace Inst {
class ActEntry;
};

namespace Planner {



class PartialPlan {
private:
	VAL::State current;
// This contains the active process effects and some information about the 
// active duratives. 
	VAL::Plan planHead;
	VAL::Plan::const_iterator currentStep;

// Note: Not sure, yet, whether the mismatch between instantiatedOp structures and
// Action structures will cause a problem. We might need a translation between them,
// or, better still, a quick way to interpret a FastEnvironment as an Environment.

// possibleActions will be created on first entry to a state after an epsilon wait
// or a long wait (or for the initial state). After an action is selected we will 
// need to filter all mutex choices out of the collection, but it is not renewed
// until a wait.
	vector<Inst::ActEntry *> possibleActions;
	vector<Inst::ActEntry *> relevantActions;

	vector<Inst::ActEntry *> scheduledActions;

	bool waited;
	VAL::Happening * pending;

	void selfSchedule(Inst::ActEntry *);
	void selfCommit();
	void selfWait(double t);
public:
	PartialPlan();
	void write(std::ostream & o) const;
	int length() const {return planHead.length();};

	PartialPlan * schedule(Inst::ActEntry *);
	PartialPlan * commit();
	PartialPlan * wait(double t);
	bool hasWaited() const {return waited;};
	bool canCommit() const {return !scheduledActions.empty();};
	void initialWait(double t);

	double timeToTrigger();
};

struct PartialPlanOrder {
	virtual ~PartialPlanOrder() {};
	virtual bool operator()(const PartialPlan * pp1,const PartialPlan * pp2) const = 0;
};

struct PartialPlanOrderer {
	PartialPlanOrder * ppo;
	PartialPlanOrderer(PartialPlanOrder * p) : ppo(p) {};
	PartialPlanOrderer() : ppo(0) {};
	bool operator()(const PartialPlan * pp1,const PartialPlan * pp2) const
	{
		return ppo->operator()(pp1,pp2);
	};
};

struct LongPlanHead : public PartialPlanOrder {
	bool operator()(const PartialPlan * pp1,const PartialPlan * pp2) const;
};

inline ostream & operator<<(ostream & o,const PartialPlan & pp)
{
	pp.write(o);
	return o;
};

};

#endif

