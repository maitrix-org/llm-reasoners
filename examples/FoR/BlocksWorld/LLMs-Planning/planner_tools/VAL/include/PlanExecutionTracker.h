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

#ifndef __PET
#define __PET

#include <map>

#include "State.h"
#include "Plan.h"
#include "VisitController.h"
#include "Proposition.h"
#include "GoalHypSpace.h"

namespace VAL {

class ActionLinker : public VisitController {
private:
	map<const SimpleProposition *,const Happening *> & links;
	map<const SimpleProposition *,const Happening *> & supps;
	const Action * act;
	Validator * vld;
	set<const SimpleProposition *> & ignores;
	const Happening * hap;
	
public:
	ActionLinker(map<const SimpleProposition *,const Happening *> & l,
				map<const SimpleProposition *,const Happening *> & sp,
				const Action * a,Validator * v,set<const SimpleProposition *> & ins,
				const Happening * h) : 
		links(l), supps(sp), act(a), vld(v), ignores(ins), hap(h) {};

	virtual void visit_simple_goal(simple_goal * s) 
	{
		if(s->getPolarity() == E_POS)
		{
			const SimpleProposition * sp = vld->pf.buildLiteral(s->getProp(),act->getBindings());
			if(ignores.find(sp) == ignores.end() && links.find(sp) != links.end())
			{
//				supps[sp] = hap;
			};
		}
	};
	virtual void visit_qfied_goal(qfied_goal *) {};
	virtual void visit_conj_goal(conj_goal * c) 
	{
		for(goal_list::const_iterator i = c->getGoals()->begin();
			i != c->getGoals()->end();++i)
		{
			(*i)->visit(this);
		};
	};
	virtual void visit_disj_goal(disj_goal * d) 
	{
		for(goal_list::const_iterator i = d->getGoals()->begin();
			i != d->getGoals()->end();++i)
		{
			(*i)->visit(this);
		};
	};
	virtual void visit_timed_goal(timed_goal * t) 
	{
		t->getGoal()->visit(this);
	};
	virtual void visit_imply_goal(imply_goal *) {};
	virtual void visit_neg_goal(neg_goal * ng) 
	{
	// This should invert - support is the other way round!
		ng->getGoal()->visit(this);
	};
};

class PlanExecutionTracker : public StateObserver, public GoalHypothesisSpace {
private:
	vector<State> states;
	set<const SimpleProposition *> trs;
	map<const SimpleProposition *,const Happening *> links;
	map<const SimpleProposition *,const Happening *> supps;
	Validator * vld;
	
public:
	PlanExecutionTracker(const State & s,Validator * v) : 
		links(), supps(), vld(v)
	{
		states.push_back(s);
	}
	void notifyChanged(const State * s,const Happening * h)
	{
		std::cout << "****** State changed\n" << "Applied: " << *h << "\n";
		
		set<const SimpleProposition *> ignores;
		set<const SimpleProposition *> sc = s->getChangedLiterals();
		for(set<const SimpleProposition *>::const_iterator i = sc.begin();i != sc.end();++i)
		{
			if(s->evaluate(*i))
			{
				bool wasTrue = false;
				for(vector<State>::iterator j = states.begin();j != states.end();++j)
				{
					if(j->evaluate(*i)) 
					{
						wasTrue = true;
						break;
					}
				}
				if(!wasTrue)
				{
					std::cout << **i << " now true for the first time\n";
					trs.insert(*i);
					links[*i] = h;
				}
			}
			else
			{
				if(trs.find(*i) != trs.end())
				{
					std::cout << **i << " now consumed\n";
					trs.erase(*i);
					std::cout << "I suspect that " << *(links[*i]) << " was executed to enable " << *h << "\n";
					ignores.insert(*i);
					supps[*i] = h;
				}
			}
		};
		const vector<const Action *> * acts = h->getActions();
		for(vector<const Action *>::const_iterator a = acts->begin(); a != acts->end(); ++a)
		{
			ActionLinker al(links,supps,*a,vld,ignores,h);
			(*a)->getAction()->precondition->visit(&al);
		}
		states.push_back(*s);
	};
	void write(std::ostream & o) const
	{
		o << "The final achievements are:\n";

		for(set<const SimpleProposition*>::const_iterator i = trs.begin(); i != trs.end(); ++i)
		{
			o << **i << "\n";
			if(supps.find(*i) == supps.end())
			{
				o << "A final goal?\n";
			}
		}
	};
};



}
#endif
