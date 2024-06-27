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

#ifndef __HOWANALYSER
#define __HOWANALYSER

#include "ptree.h"
#include "VisitController.h"
#include <iostream>
#include "TypedAnalyser.h"
#include "TimSupport.h"
#include <set>

using std::set;
using std::cout;
using namespace TIM;

namespace VAL {

class AbstractGraph;

class HWWAction {
private:
	typedef map<vector<int>,vector<extended_pred_symbol *> > ArgSets;
	ArgSets argSets;
public:
	HWWAction(operator_ * op) 
	{};

	void add(const vector<int> & v,extended_pred_symbol * pd)
	{
		argSets[v].push_back(pd);
	};

	void write(ostream & o) const
	{
		for(ArgSets::const_iterator i = argSets.begin();i != argSets.end();++i)
		{
			o << "{";
			for(unsigned int j = 0;j < i->first.size();++j)
			{
				o << (i->first)[j] << " ";
			};
			o << "}";
		};
	};

	void analyse(action * a)
	{
		for(ArgSets::const_iterator i = argSets.begin();i != argSets.end();++i)
		{
			cout << "Argument set: {";
			for(unsigned int j = 0;j < i->first.size();++j)
			{
				cout << (i->first)[j] << " ";
			};
			cout << "}\n";
			conj_goal * cg = dynamic_cast<conj_goal *>(a->precondition);
			if(!cg)
			{
				cout << "Complex precondition\n";
				continue;
			};
			for(goal_list::const_iterator g = cg->getGoals()->begin();g != cg->getGoals()->end();++g)
			{
				const simple_goal * sg = dynamic_cast<const simple_goal *>(*g);
				if(!sg)
				{
					cout << **g << " is not a simple goal\n";
					continue;
				};
				cout << "Going to analyse: ";
				EPS(sg->getProp()->head)->writeName(cout);
				cout << "\n";
				
			};
		};
	};
};

inline ostream & operator << (ostream & o,const HWWAction & a)
{
	a.write(o);
	return o;
};

class HowAnalyser : public VisitController {
private:
	map<operator_*,HWWAction *> acts;

	set<extended_pred_symbol *> epss;
	AbstractGraph * ag;
	
public:
	HowAnalyser();

	void completeGraph();
	
	virtual void visit_action(action * a);

	virtual void visit_effect_lists(effect_lists * el)
	{
		for(pc_list<simple_effect *>::iterator i = el->add_effects.begin();
				i != el->add_effects.end();++i)
		{
			(*i)->visit(this);
		};

		for(pc_list<assignment *>::iterator i = el->assign_effects.begin();
				i != el->assign_effects.end();++i)
		{
			(*i)->visit(this);
		};

		for(pc_list<timed_effect *>::iterator i = el->timed_effects.begin();
				i != el->timed_effects.end();++i)
		{
			(*i)->visit(this);
		};
	};

	virtual void visit_simple_effect(simple_effect * se);
	virtual void visit_timed_effect(timed_effect * te)
	{
		cout << "Timed initial literal: not yet handled\n";
	};

	virtual void visit_assignment(assignment * a)
	{
		cout << "Initial assignment value: not yet considered\n";
	};
	
	virtual void visit_pred_decl(pred_decl * p)
	{
		if(dynamic_cast<TIMpred_decl*>(p) != 0) return;

		cout << "\nVisiting " << p->getPred()->getName() << "\n";
		
		holding_pred_symbol * hps = HPS(p->getPred());
		if(hps)
		{
			for(holding_pred_symbol::PIt i = hps->pBegin();i != hps->pEnd();++i)
			{
				(*i)->writeName(cout); 
				cout << ": ";
				if((*i)->isStatic()) 
				{
					cout << "Static (or wrong type level)";
				}
				else if((*i)->decays()) 
				{
					cout << "Decays";
				}
				else 
				{
					cout << (*i)->addsEnd()-(*i)->addsBegin() << " choices:";
					for(extended_pred_symbol::OpProps::const_iterator j = (*i)->addsBegin();
							j != (*i)->addsEnd();++j)
					{
						if(acts.find(j->op) == acts.end())
						{
							acts.insert(make_pair(j->op,new HWWAction(j->op)));
						};
						int pc = 0;
						vector<int> ags;
						cout << "\n\t" << *(j->op->name) << "(";
					 	for(var_symbol_list::const_iterator k = j->op->parameters->begin();
					 			k != j->op->parameters->end();++k,++pc)
					 	{
					 		unsigned int ac = 0;
					 		bool d = false;
					 		parameter_symbol_list::const_iterator x = j->second->args->begin();
					 		for(ac=0;ac < j->second->args->size();++x,++ac)
					 		{
					 			if(*x == *k)
					 			{
					 				cout << "P[" << ac << "] ";
					 				ags.push_back(pc);
					 				d = true;
					 			};
					 		};
					 		if(!d) 
					 		{
								cout << "{" << theTC->range(*k).size() << " choices of ";
								
								if((*k)->type) 
								{
									cout << (*k)->type->getName();
									
									vector<const pddl_type *> ls = theTC->leaves((*k)->type);
									if(!ls.empty())
									{
										cout << " [";
										for(vector<const pddl_type *>::const_iterator x = ls.begin();
												x != ls.end();++x)
										{
											cout << (*x)->getName() << " ";
										};
										cout << "]";
									};
								}
								else
								{
									cout << "?";
								};
								cout << "} ";
								
					 		};
					 	};
					 	cout << ")";
					 	acts[j->op]->add(ags,*i);
					};
				};
				cout << "\n";
				TIMpredSymbol * tps = TPS(*i);
				
				if(tps && !(*i)->isStatic())
				{
					bool something = false;
					cout << "TIM says:\n";
					for(int a = 0;a < tps->arity();++a)
					{
						if(tps->property(a)->isSingleValued())
						{
							cout << *(tps->property(a)) << " is ";
							cout << "single valued\n";
							something = true;
						}
						
					};
					if(!something) 
					{
						cout << "nothing interesting here\n";
					};
				};
			};
		};	
	};
};

};

#endif
