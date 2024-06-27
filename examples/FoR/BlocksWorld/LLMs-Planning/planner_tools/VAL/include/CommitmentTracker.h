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

#ifndef __ComT
#define __ComT

#include <map>
#include <set>

#include "State.h"
#include "Plan.h"
#include "VisitController.h"
#include "Proposition.h"
#include "Agents.h"

namespace VAL {

class ActionLinker : public VisitController {
private:
	map<const SimpleProposition *,const Happening *> & links;
	map<const SimpleProposition *,set<pair<const Happening *,const Action *> > > & actSupps;
	map<const Happening *,map<const Action*,vector<int> > > & agLs;
	const Action * act;
	Validator * vld;
	set<const SimpleProposition *> & ignores;
	const Happening * hap;

	void whichGroups(Agents & as)
	{
		agLs[hap][act] = as.whichGroups(act->getBindings());
	};
	
public:
	ActionLinker(map<const SimpleProposition *,const Happening *> & l,
				map<const SimpleProposition *,set<pair<const Happening *,const Action *>  > > & asp,
				map<const Happening *,map<const Action*,vector<int> > > & ags,
				const Action * a,Validator * v,set<const SimpleProposition *> & ins,
				const Happening * h,Agents & as) : 
		links(l), actSupps(asp), agLs(ags), act(a), vld(v), ignores(ins), hap(h) {whichGroups(as);};

	virtual void visit_simple_goal(simple_goal * s) 
	{
		if(s->getPolarity() == E_POS)
		{
			const SimpleProposition * sp = vld->pf.buildLiteral(s->getProp(),act->getBindings());
			if(ignores.find(sp) == ignores.end() && links.find(sp) != links.end())
			{
				actSupps[sp].insert(make_pair(hap,act));
				std::cout << *sp << " is required to support " << act->getName() << " at " << hap->getTime() << "\n";
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

	virtual void visit_comparison(comparison * cmp)
	{
		std::cout << act->getName() << " has a numeric precondition - not yet handling these\n";
	}
	
};

class CommitmentTracker : public StateObserver {
private:
	vector<State> states;
	map<const Happening *,int> happenings;
	set<const SimpleProposition *> trs;
	map<const SimpleProposition *,const Happening *> links;
	map<const SimpleProposition *,const Happening *> supps;
	map<const SimpleProposition *,set<pair<const Happening *,const Action *> > > actSupps;
	map<const Happening *,map<const Action*,vector<int> > > agentLinks;
	Validator * vld;
	Agents agents;
	
public:
	CommitmentTracker(const State & s,Validator * v,Agents & as) : 
		links(), supps(), vld(v), agents(as)
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
				}
				links[*i] = h;
			}
			else
			{
				if(trs.find(*i) != trs.end())
				{
					std::cout << **i << " now consumed\n";
					trs.erase(*i);
					std::cout << "I suspect that " << *(links[*i]) << " was executed to enable " << *h << "\n";
					const set<const Action *> & ss = states[happenings[links[*i]]].whatDidThis(*i);
					if(ss.empty()) std::cout << "!!!!Nothing Here!!!\n";
					for(set<const Action*>::const_iterator ac = ss.begin();ac != ss.end();++ac)
					{
						std::cout << "It was " << (*ac)->getName() << " that was applied at " <<  links[*i]->getTime() << " to achieve " << **i << "\n";
					}

					
					for(map<const Action *,vector<int> >::iterator xx = agentLinks[links[*i]].begin();
							xx != agentLinks[links[*i]].end();++xx)
					{
						if(xx->second.size() == 1)
						{
							std::cout << "Perhaps a commitment by " << agents.show(xx->second[0]) << " in " << xx->first->getName() << "\n";

							
						}
						else if(xx->second.size() > 1)
						{
							std::cout << "Handover between agents in action: " << xx->first->getName() << "\n";
						}
					}
					
					ignores.insert(*i);
					supps[*i] = h;
				}
			}
		};
		const vector<const Action *> * acts = h->getActions();
		actSupps.clear();
		for(vector<const Action *>::const_iterator a = acts->begin(); a != acts->end(); ++a)
		{
			ActionLinker al(links,actSupps,agentLinks,*a,vld,ignores,h,agents);
			(*a)->getAction()->precondition->visit(&al);
			
		}

		for(map<const SimpleProposition *,set<pair<const Happening *,const Action *> > >::const_iterator i = actSupps.begin();i != actSupps.end();++i)
		{
			for(set<pair<const Happening *,const Action *> >::const_iterator j = i->second.begin();j != i->second.end();++j)
			{
				std::cout << *(i->first) << " supports " << j->second->getName() << " at " << j->first->getTime() << " and was achieved by:\n";
				const set<const Action *> & ss = states[happenings[links[i->first]]].whatDidThis(i->first);
				if(ss.empty()) std::cout << "!!!!Nothing Here!!!\n";
				for(set<const Action*>::const_iterator ac = ss.begin();ac != ss.end();++ac)
				{
					std::cout << "-- " << (*ac)->getName() << " that was started at "
					 <<  (*ac)->startOfAction()->getPlanStep()->start_time << " and achieved the support at " << links[i->first]->getTime() << "\n";
				}
			}
		}
		states.push_back(*s);
		happenings[h] = states.size()-1;
	};

	
	void conclude()
	{
		
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
