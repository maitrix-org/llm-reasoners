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

#ifndef __SEARCHSPACE
#define __SEARCHSPACE

#include "Validator.h"
#include "ptree.h"
#include "graphconstruct.h"
#include "TIM.h"
#include "PartialPlan.h"

#include <queue>

namespace Planner {

// This is a singleton object....

class SearchSpace {
private:
// We need all these things to build a Validator object that can then be used to 
// support the simulation process for execution of Happenings.
	VAL::DerivationRules derivRules;
	VAL::plan dummyPlan;
	VAL::Validator val;

	Inst::GraphFactory * gf;
	Inst::PlanGraph pg;

	typedef std::priority_queue<PartialPlan *,vector<PartialPlan *>,PartialPlanOrderer> PPQueue;
	PPQueue * ppq;

	bool hasOrder;
	PartialPlanOrder * myPPO;
	
	SearchSpace();
	SearchSpace(const SearchSpace &);
	
public:
	static SearchSpace & instance() 
	{
		static SearchSpace ssp;
		return ssp;
	};

	~SearchSpace()
	{
		delete myPPO;
	};
	
	void setOrdering(PartialPlanOrder * ppo)
	{
		if(hasOrder)
		{
			PPQueue * ppq1 = new PPQueue(PartialPlanOrderer(ppo));
			
			while(!ppq->empty())
			{
				ppq1->push(ppq->top());
				ppq->pop();
			};
			delete ppq;
			ppq = ppq1;
			delete myPPO;
			myPPO = ppo;
		}
		else
		{
			ppq = new PPQueue(PartialPlanOrderer(ppo));
			myPPO = ppo;
			hasOrder = true;
		};
	};
	
	VAL::Validator * getVal() {return &val;};
	VAL::plan * getDummyPlan() {return &dummyPlan;};
	Inst::PlanGraph & getPG() {return pg;};


	void findPlan();
};



};



#endif
