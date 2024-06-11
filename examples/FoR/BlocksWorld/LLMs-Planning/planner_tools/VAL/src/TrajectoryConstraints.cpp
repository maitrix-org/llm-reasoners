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

#include "TrajectoryConstraints.h"
#include "Proposition.h"
#include "VisitController.h"
#include "Validator.h"


namespace VAL {

int PreferenceMonitor::id = 0;
std::set<int> PreferenceMonitor::done;

#define MESSAGE(x) if(LaTeX) { *report << x << "\\\\\n"; } else if(Verbose) { cout << x << "\n";}

bool PreferenceMonitor::violationSerious(const State * s) const
{
	if(done.find(myId) == done.end())
	{
		AdvicePropositionConj a;
		vld->countViolation(s,name,&a);
		done.insert(myId);
	};
	return false;
};

class TrajectoryConstraintsMonitor::CollectProps : public VisitController {
private:
	bool isPref;
	string name;
	bool share;
	bool done;

	Environment bs;

	MonitorOwner mono;
	
public:
	
	TrajectoryConstraintsMonitor & tjm;

	CollectProps(TrajectoryConstraintsMonitor & tcm) : 
		isPref(false), share(false), tjm(tcm)
	{};
	
	void visit_qfied_goal(qfied_goal * qg) 
	{
		if(qg->getQuantifier() == E_EXISTS) 
		{
			std::cerr << "Existentially quantified preference or constraint!\n";
			exit(-1);
		};
		if(isPref) 
		{
			share = true;
			done = false;
		};
		// Need to construct Environments for the quantified expression
		vector<unsigned int> cs;
		vector<vector<const_symbol *> > vs;
		vector<const var_symbol *> vrs;
		for(var_symbol_list::const_iterator i = qg->getVars()->begin();i != qg->getVars()->end();++i)
		{
			vs.push_back(tjm.vld->range(*i));
			cs.push_back(0);
			vrs.push_back(*i);
			bs[*i] = vs[vs.size()-1][0];
//			cout << **i << " = " << *bs[*i] << "\n";
		};
		unsigned int j = 0;
		while(j < vs.size())
		{
			for(unsigned int i = 0; i <= j;++i)
			{
				bs[vrs[i]] = vs[i][cs[i]];
//				cout << *vrs[i] << " = ";
				
//				cout << *bs[vrs[i]] << "\n";
			};
			qg->getGoal()->visit(this);

			j = 0;
			++cs[0];
			while(j < vs.size() && cs[j] == vs[j].size())
			{
				cs[j] = 0;
				++j;
				++cs[j];
			};
		};
				 

		bs.clear();
		share = false;
	};
	
	void visit_conj_goal(conj_goal * cg) 
	{
		for(goal_list::const_iterator i = cg->getGoals()->begin();i != cg->getGoals()->end();++i)
		{
			(*i)->visit(this);
		};
	};

	void visit_preference(preference * p)
	{
		isPref = true;
		name = p->getName();
		p->getGoal()->visit(this);
		isPref = false;
	};
	
	void visit_constraint_goal(constraint_goal * cg) 
	{
		Environment * e1 = bs.empty()?0:bs.copy(tjm.vld);
		const Proposition * p = !e1?tjm.vld->pf.buildProposition(cg->getRequirement()):
									tjm.vld->pf.buildProposition(cg->getRequirement(),*e1,true,0);
		e1 = bs.empty()?0:bs.copy(tjm.vld);
		const Proposition * q = cg->getTrigger()?(!e1?tjm.vld->pf.buildProposition(cg->getTrigger()):
														tjm.vld->pf.buildProposition(cg->getTrigger(),*e1,true,0)):0;

//		cout << "Got " << *p << "\n";
		tjm.allProps.push_back(p);
		if(q) tjm.allProps.push_back(q);
		switch(cg->getCons())
		{
			case E_ATEND:
			{
				PropMonitor pm(p);
				if(isPref) 
				{
					if(share && done)
					{
						passOn(pm,mono);
					}
					else
					{
						pm.setPreference(tjm.vld,name);
						if(share)
						{
							passOn(mono,pm);
							done = true;
						};
					};
				};
				tjm.atEnd.push_back(pm);
			};
				break;
			case E_ALWAYS:
			{
				PropMonitor pm(p);
				if(isPref) 
				{
					if(share && done)
					{
						passOn(pm,mono);
					}
					else
					{
						pm.setPreference(tjm.vld,name);
						if(share)
						{
							passOn(mono,pm);
							done = true;
						};
					};
				};
				tjm.always.push_back(pm);
			};
				break;
			case E_SOMETIME:
			{
				PropMonitor pm(p);
				if(isPref) 
				{
					if(share && done)
					{
						passOn(pm,mono);
					}
					else
					{
						pm.setPreference(tjm.vld,name);
						if(share)
						{
							passOn(mono,pm);
							done = true;
						};
					};
				};
				tjm.sometime.push_back(pm);
			};
				break;
			case E_WITHIN:
			{
				Deadlined d(cg->getDeadline(),p);
				if(isPref)
				{
					if(share && done)
					{
						passOn(d,mono);
					}
					else
					{
						d.setPreference(tjm.vld,name);
						if(share)
						{
							passOn(mono,d);
							done = true;
						};
					};
				};
				tjm.within.push_back(d);
			};
				break;
			case E_ATMOSTONCE:
			{
				PropMonitor pm(p);
				if(isPref) 
				{
					if(share && done)
					{
						passOn(pm,mono);
					}
					else
					{
						pm.setPreference(tjm.vld,name);
						if(share)
						{
							passOn(mono,pm);
							done = true;
						};
					};
				};
				tjm.atMostOnce.push_back(pm);
			};
				break;
			case E_SOMETIMEAFTER:
			{
				PropositionPair pp(q,p);
				if(isPref)
				{
					if(share && done)
					{
						passOn(pp,mono);
					}
					else
					{
						pp.setPreference(tjm.vld,name);
						if(share)
						{
							passOn(mono,pp);
							done = true;
						};
					};
				};
				tjm.sometimeAfter.push_back(pp);
			};
				break;
			case E_SOMETIMEBEFORE:
			{
				PropositionPair pp(q,p);
				if(isPref)
				{
					if(share && done)
					{
						passOn(pp,mono);
					}
					else
					{
						pp.setPreference(tjm.vld,name);
						if(share)
						{
							passOn(mono,pp);
							done = true;
						};
					};
				};
				tjm.sometimeBefore.push_back(pp);
			};
				break;
			case E_ALWAYSWITHIN:
			{
				TriggeredDeadlined td(q,Deadlined(cg->getDeadline(),p));
				if(isPref)
				{
					if(share && done)
					{
						passOn(td,mono);
					}
					else
					{
						td.setPreference(tjm.vld,name);
						if(share)
						{
							passOn(mono,td);
							done = true;
						};
					};
				};
				tjm.alwaysWithin.push_back(td);
			};
				break;
			case E_HOLDDURING:
			{
				Window w(make_pair(cg->getFrom(),cg->getDeadline()),p);
				if(isPref)
				{
					if(share && done)
					{
						passOn(w,mono);
					}
					else
					{
						w.setPreference(tjm.vld,name);
						if(share)
						{
							passOn(mono,w);
							done = true;
						};
					};
				};
				tjm.holdDuring.push_back(w);
			};
				break;
			case E_HOLDAFTER:
			{
				Deadlined d(cg->getDeadline(),p);
				if(isPref)
				{
					if(share && done)
					{
						passOn(d,mono);
					}
					else
					{
						d.setPreference(tjm.vld,name);
						if(share)
						{
							passOn(mono,d);
							done = true;
						};
					};
				};
				tjm.holdAfter.push_back(d);
			};
				break;
			default:
				break;
		};
	};
};


TrajectoryConstraintsMonitor::TrajectoryConstraintsMonitor(Validator * v,con_goal * cg1,con_goal * cg2) :
	isActive(false), vld(v)
{
// Set up the initial propositions.
	CollectProps cp(*this);
	if(cg1) 
	{
		isActive = true;
		cg1->visit(&cp);
	}
	if(cg2) 
	{
		isActive = true;
		cg2->visit(&cp);
	};
};

TrajectoryConstraintsMonitor::~TrajectoryConstraintsMonitor()
{
	for(Propositions::iterator i = allProps.begin();i != allProps.end();++i)
	{
		(*i)->destroy();
	};
};

bool TrajectoryConstraintsMonitor::checkAtState(const State & s) 
{
	if(!isActive) return true;
	MESSAGE("Checking trajectory constraints at " << s.getTime());

	/*if(LaTeX)
	{
		*report << "Checking trajectory constraints at " << s.getTime() << "\\\\\n"
	}
	else if(Verbose) 
	{
		cout << "Checking trajectory constraints at " << s.getTime() << "\n";
	};*/

	// Before always to allow checking of condition from current state.
	Deadlines d;
	for(Deadlines::const_iterator i = holdAfter.begin();i != holdAfter.end();++i)
	{
		if(i->first <= s.getTime())
		{
			MESSAGE("Now requiring " << *(i->second) << " to remain true");
			PropMonitor pm(i->second);
			passOn(pm,*i);
			always.push_back(pm);
		}
		else
		{
			d.push_back(*i);
		};
	};
	holdAfter.swap(d);
	
	for(Propositions::const_iterator i = always.begin();i != always.end();++i)
	{
		if(always.size() < 10) {MESSAGE("Checking " << **i);};
		if(!(*i)->evaluate(&s))
		{
			*report << "Failure to satisfy: " << **i << "\n";
			if(i->violationSerious(&s)) return false;
		};
	};

	// Check these first so that triggered deadlines get tested against current state
	for(TriggeredDeadlines::const_iterator i = alwaysWithin.begin();i != alwaysWithin.end();++i)
	{
		if((i->first)->evaluate(&s))
		{
			MESSAGE("Triggered deadline goal " << *(i->second.second) << " by " 
					<< s.getTime()+(i->second.first));
			Deadlined dl(s.getTime()+(i->second.first),i->second.second);
			passOn(dl,*i);
			within.push_back(dl);
		};
	};
	
	d.clear();
	for(Deadlines::const_iterator i = within.begin();i != within.end();++i)
	{
		if(within.size() < 10)
			{MESSAGE("Checking deadline goal: " << *(i->second) << " by " << i->first);};
		if(s.getTime() > i->first)
		{
			MESSAGE("Failed to meet deadline on " << *(i->second) << " (" << 
				i->first << ")");
			//vld->getErrorLog().addGoal(i->second,&s);
			if(i->violationSerious(&s)) return false;
		}
		else if((i->second)->evaluate(&s))
		{
			MESSAGE("Satisfied " << *(i->second) << " at " << s.getTime());
		}
		else
		{
			d.push_back(*i);
		};
	};
	within.swap(d);
	
	// Do before sometime to allow the sometime check to handle triggered goals in
	// current state.
	for(PropositionPairs::const_iterator i = sometimeAfter.begin();i != sometimeAfter.end();++i)
	{
		if((i->first)->evaluate(&s))
		{
			MESSAGE("Triggered a requirement for " << *(i->second) << " after " <<
					*(i->first) << " at " << s.getTime());
			PropMonitor pm(i->second);
			passOn(pm,*i);
			sometime.push_back(pm);
		};
	};

	Propositions ps;
	for(Propositions::const_iterator i = sometime.begin();i != sometime.end();++i)
	{
		if((*i)->evaluate(&s))
		{
			MESSAGE("Satisfied " << **i << " at " << s.getTime());
		}
		else
		{
			ps.push_back(*i);
		};
	};
	sometime.swap(ps);

	ps.clear();
	for(Propositions::const_iterator i = atMostOnce.begin();i != atMostOnce.end();++i)
	{
		if((*i)->evaluate(&s))
		{
			MESSAGE("First satisfied " << **i << " at " << s.getTime());
			currently.push_back(*i);
		}
		else
		{
			ps.push_back(*i);
		};
	};
	atMostOnce.swap(ps);

	ps.clear();
	for(Propositions::const_iterator i = currently.begin();i != currently.end();++i)
	{
		if((*i)->evaluate(&s))
		{
			ps.push_back(*i);
		}
		else
		{
			MESSAGE(**i << " stops being true at " << s.getTime());
			never.push_back(*i);
		};
	};
	currently.swap(ps);

	for(Propositions::const_iterator i = never.begin();i != never.end();++i)
	{
		if((*i)->evaluate(&s))
		{
			MESSAGE("Constraint violated by satisfaction of " << **i 
						<< " at " << s.getTime());
			if(i->violationSerious(&s)) return false;
		};
	};

	Windows w;
	for(Windows::const_iterator i = holdDuring.begin(); i != holdDuring.end();++i)
	{
		if(i->first.first <= s.getTime() && i->first.second > s.getTime())
		{
			MESSAGE("In window for " << *(i->second));
			if(!(i->second)->evaluate(&s))
			{
				MESSAGE("Condition failed");
				if(i->violationSerious(&s)) return false;
			};
		}
		else if(i->first.second > s.getTime())
		{
			w.push_back(*i);
		};
	};
	holdDuring.swap(w);

	PropositionPairs pps;
	for(PropositionPairs::const_iterator i = sometimeBefore.begin();i != sometimeBefore.end();++i)
	{
		if(!((i->second)->evaluate(&s)))
		{
			pps.push_back(*i);
		};
		if((i->first)->evaluate(&s))
		{
			MESSAGE(*(i->first) << " satisfied without " << *(i->second) 
					<< " sometime before");
			if(i->violationSerious(&s)) return false;
		};
	};
	sometimeBefore.swap(pps);
	
	return true;
};

bool TrajectoryConstraintsMonitor::checkFinalState(const State & s)
{
	if(!isActive) return true;
	if(!checkAtState(s)) return false;

	for(Propositions::const_iterator i = atEnd.begin();i != atEnd.end();++i)
	{
		if(!(*i)->evaluate(&s))
		{
			MESSAGE("Final requirement " << **i << " violated");
			if(i->violationSerious(&s)) return false;
		};
	};

	if(!sometime.empty())
	{
		bool serious = false;
		MESSAGE("Outstanding requirements unsatisfied during plan:");
		for(Propositions::const_iterator i = sometime.begin();i != sometime.end();++i)
		{
			MESSAGE("\t" << **i);
			serious |= i->violationSerious(&s);
		};
		if(serious) return false;
	};

	if(!within.empty())
	{
		bool serious = false;
		MESSAGE("Outstanding deadlines to be met:");
		for(Deadlines::const_iterator i = within.begin();i != within.end();++i)
		{
			MESSAGE("\t" << *(i->second) << " by " << i->first);
			serious |= i->violationSerious(&s);
		};
		if(serious) return false;
	};
	
	return true;
};

};
