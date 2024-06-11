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

#include "SearchSpace.h"

namespace Planner {

SearchSpace::SearchSpace() :
	derivRules(VAL::current_analysis->the_domain->drvs,VAL::current_analysis->the_domain->ops),
	val(&derivRules,0.001,*VAL::theTC,VAL::current_analysis->the_domain->ops,VAL::current_analysis->the_problem->initial_state,
				&dummyPlan,VAL::current_analysis->the_problem->metric,true,true,0,0),
	gf(new Inst::GraphFactory()), 
	pg(gf), ppq(0), hasOrder(false), myPPO(0)
{
	pg.extendToGoals();
};


void SearchSpace::findPlan()
{
	if(!hasOrder)
	{
		cout << "You have to set the ordering for Partial Plans before searching for a plan\n";
		exit(0);
	};
	PartialPlan * pp = new PartialPlan();
	cout << "Our first partial plan:\n" << *pp << "\n";
	ppq->push(pp);

	cout << "Let's embark on our search for a plan....\n";
	cout << "Our story begins with\n" << *(ppq->top()) << "\n";

	cout << "First we wait epsilon\n";
	pp->initialWait(val.getTolerance());
	ppq->pop();
	ppq->push(pp);
	cout << "Now we have:\n" << *pp << "\n";
	pp->timeToTrigger();
};
	
};
