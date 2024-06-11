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

#include <cstdio>
#include <iostream>
#include <fstream>
#include "ptree.h"
#include <FlexLexer.h>
#include "TIM.h"
#include "instantiation.h"
#include "SimpleEval.h"
#include "DebugWriteController.h"
#include "typecheck.h"


using std::ifstream;
using std::cerr;

using namespace VAL;

namespace Inst {

bool varFree(const VAL::parameter_symbol_list * pl)
{
	for(VAL::parameter_symbol_list::const_iterator i = pl->begin();i != pl->end();++i)
	{
		if(!dynamic_cast<const VAL::const_symbol *>(*i)) return false;
	};
	return true;
}


ostream & operator<<(ostream & o,const instantiatedOp & io)
{
	io.write(o);
	return o;
};

ostream & operator<<(ostream & o,const PNE & io)
{
	io.write(o);
	return o;
};

ostream & operator<<(ostream & o,const Literal & io)
{
	io.write(o);
	return o;
};

void instantiatedOp::writeAll(ostream & o) 
{
	instOps.write(o);
};


OpStore instantiatedOp::instOps;
map<VAL::pddl_type *,vector<VAL::const_symbol*> > instantiatedOp::values;


void instantiatedOp::instantiate(const VAL::operator_ * op,const VAL::problem * prb,VAL::TypeChecker & tc)
{
	FastEnvironment e(static_cast<const id_var_symbol_table*>(op->symtab)->numSyms());
	vector<vector<VAL::const_symbol*>::const_iterator> vals(op->parameters->size());
	vector<vector<VAL::const_symbol*>::const_iterator> starts(op->parameters->size());
	vector<vector<VAL::const_symbol*>::const_iterator> ends(op->parameters->size());
	vector<VAL::var_symbol *> vars(op->parameters->size());
	int i = 0;
	int c = 1;
	for(var_symbol_list::const_iterator p = op->parameters->begin();
			p != op->parameters->end();++p,++i)
	{
		if(values.find((*p)->type) == values.end()) 
		{
			values[(*p)->type] = tc.range(*p);
		};
		vals[i] = starts[i] = values[(*p)->type].begin();
		ends[i] = values[(*p)->type].end();
		if(ends[i]==starts[i]) return;
		e[(*p)] = *(vals[i]);
		vars[i] = *p;
		c *= values[(*p)->type].size();
	};
//	cout << c << " candidates to consider\n";
	if(!i)
	{
		SimpleEvaluator se(&e);
		op->visit(&se);
		if(!se.reallyFalse())
		{
			FastEnvironment * ecpy = e.copy();
			instantiatedOp * o = new instantiatedOp(op,ecpy);
			if(instOps.insert(o))
			{
				delete o;
			};
				
		};
		return;
	};
	--i;
	while(vals[i] != ends[i])
	{
		if(!TIM::selfMutex(op,makeIterator(&e,op->parameters->begin()),
						makeIterator(&e,op->parameters->end())))
		{
			
			SimpleEvaluator se(&e);
			const_cast<VAL::operator_*>(op)->visit(&se);
			if(!se.reallyFalse())
			{
				FastEnvironment * ecpy = e.copy();
				instantiatedOp * o = new instantiatedOp(op,ecpy);
				if(instOps.insert(o))
				{
					delete o;
				};
			};
		};
/*
 *		else
		{
			cout << "Killed\n" << op->name->getName() << "(";
			for(var_symbol_list::const_iterator a = op->parameters->begin();
					a != op->parameters->end();++a)
			{
				cout << e[*a]->getName() << " ";
			};
			cout << ")\n";
		};
*/
		int x = 0;
		++vals[0];
		if(vals[0] != ends[0]) e[vars[0]] = *(vals[0]);
		while(x < i && vals[x] == ends[x])
		{
			vals[x] = starts[x];
			e[vars[x]] = *(vals[x]);
			++x;
			++vals[x];
			if(vals[x] != ends[x]) e[vars[x]] = *(vals[x]);
		};
	};
};

LiteralStore instantiatedOp::literals;
PNEStore instantiatedOp::pnes;



class Collector : public VisitController {
private:
	bool adding;
	const VAL::operator_ * op;
	FastEnvironment * fe;
	LiteralStore & literals;
	PNEStore & pnes;
public:
	Collector(const VAL::operator_ * o,FastEnvironment * f,LiteralStore & l,PNEStore & p) :
		adding(true), op(o), fe(f), literals(l), pnes(p)
	{};
	
	virtual void visit_simple_goal(simple_goal * p) 
	{
		Literal * l = new Literal(p->getProp(),fe);
		if(literals.insert(l))
		{
			delete l;
		};
	};
	virtual void visit_qfied_goal(qfied_goal * p) 
	{p->getGoal()->visit(this);};
	virtual void visit_conj_goal(conj_goal * p) 
	{p->getGoals()->visit(this);};
	virtual void visit_disj_goal(disj_goal * p) 
	{p->getGoals()->visit(this);};
	virtual void visit_timed_goal(timed_goal * p) 
	{p->getGoal()->visit(this);};
	virtual void visit_imply_goal(imply_goal * p) 
	{
		p->getAntecedent()->visit(this);
		p->getConsequent()->visit(this);
	};
	virtual void visit_neg_goal(neg_goal * p) 
	{
		p->getGoal()->visit(this);
	};
	virtual void visit_preference(preference * p)
	{
		p->getGoal()->visit(this);
	};
	virtual void visit_simple_effect(simple_effect * p) 
	{
		Literal * l = new Literal(p->prop,fe);
		if(literals.insert(l))
		{
			delete l;
		};
	};
	virtual void visit_constraint_goal(constraint_goal * cg)
	{
		if(cg->getRequirement())
		{
			cg->getRequirement()->visit(this);
		};
		if(cg->getTrigger())
		{
			cg->getTrigger()->visit(this);
		};
	};
	virtual void visit_forall_effect(forall_effect * p) 
	{
		p->getEffects()->visit(this);
	};
	virtual void visit_cond_effect(cond_effect * p) 
	{
		p->getCondition()->visit(this);
		p->getEffects()->visit(this);
	};
	virtual void visit_timed_effect(timed_effect * p) 
	{
		p->effs->visit(this);
	};
	virtual void visit_timed_initial_literal(timed_initial_literal * p)
	{
		p->effs->visit(this);
	};
	virtual void visit_effect_lists(effect_lists * p) 
	{
		p->add_effects.pc_list<simple_effect*>::visit(this);
		p->forall_effects.pc_list<forall_effect*>::visit(this);
		p->cond_effects.pc_list<cond_effect*>::visit(this);
		p->timed_effects.pc_list<timed_effect*>::visit(this);
		bool whatwas = adding;
		adding = !adding;
		p->del_effects.pc_list<simple_effect*>::visit(this);
		adding = whatwas;
		p->assign_effects.pc_list<assignment*>::visit(this);
	};
	virtual void visit_operator_(VAL::operator_ * p) 
	{
		adding = true;
		p->effects->visit(this);
	};
	virtual void visit_action(VAL::action * p)
	{
		visit_operator_(static_cast<VAL::operator_*>(p));
	};
	virtual void visit_durative_action(VAL::durative_action * p) 
	{
		visit_operator_(static_cast<VAL::operator_*>(p));
	};
	virtual void visit_problem(VAL::problem * p) 
	{
		p->initial_state->visit(this);
		p->the_goal->visit(this);
		if(p->constraints)
			p->constraints->visit(this);
// Maybe should visit the domain constraints, too?
	};

	virtual void visit_assignment(assignment * a) 
	{
		const func_term * ft = a->getFTerm();
		PNE * pne = new PNE(ft,fe);
		if(pnes.insert(pne))
		{
			delete pne;
		};
	};
};

void instantiatedOp::createAllLiterals(VAL::problem * p) 
{
	Collector c(0,0,literals,pnes);
	p->visit(&c);
	for(OpStore::iterator i = instOps.begin(); i != instOps.end(); ++i)
	{
		(*i)->collectLiterals();
	};
};



void instantiatedOp::collectLiterals()
{
	Collector c(op,env,literals,pnes);
	op->visit(&c);
};

void instantiatedOp::writeAllLiterals(ostream & o) 
{
	literals.write(o);
};

void instantiatedOp::writeAllPNEs(ostream & o)
{
	pnes.write(o);
};

VAL::const_symbol * const getConst(string name) 
{
	return current_analysis->const_tab.symbol_get(name);
};

VAL::const_symbol * const getConst(char * name) 
{
	return current_analysis->const_tab.symbol_get(name);
};

// Added by AMC to test whether a goal may be satisfied by the effects
// of an InstantiatedOp

bool instantiatedOp::isGoalMetByOp(const Literal * lit)
{
  effect_lists * effs = op->effects;

  return isGoalMetByEffect(effs, lit);
};

bool instantiatedOp::isGoalMetByEffect(const VAL::effect_lists * effs, const Literal * lit)
{
  using VAL::pc_list;

  for(pc_list<VAL::simple_effect*>::const_iterator i = effs->add_effects.begin(); i != effs->add_effects.end(); ++i)
    {
	if (isGoalMetByEffect(*i, lit)) return true;
    };
  for(pc_list<VAL::forall_effect*>::const_iterator i = effs->forall_effects.begin(); i != effs->forall_effects.end();++i)
    {
      if (isGoalMetByEffect(*i, lit)) return true;
    };
  for(pc_list<VAL::cond_effect*>::const_iterator i = effs->cond_effects.begin(); i != effs->cond_effects.end();++i)
    {
      if (isGoalMetByEffect(*i, lit)) return true;
    };
  for(pc_list<VAL::cond_effect*>::const_iterator i = effs->cond_effects.begin(); i != effs->cond_effects.end();++i)
    {
      if (isGoalMetByEffect(*i, lit)) return true;
    };
  for (pc_list<VAL::timed_effect*>::const_iterator i = effs->timed_effects.begin(); i != effs->timed_effects.end(); ++i)
    {
      if (isGoalMetByEffect(*i, lit)) return true;
    };
  return false;
};

bool instantiatedOp::isGoalMetByEffect(VAL::simple_effect * seff, const Literal * lit)
{
  Literal l (seff->prop,env);
  Literal * lt = instantiatedOp::getLiteral(&l);
  //  std::cout <<"Simple effect: " << (*lt) << "\n";
  return (lit==lt);
};

bool instantiatedOp::isGoalMetByEffect(VAL::forall_effect * fleff, const Literal * lit)
{
  if (isGoalMetByEffect(fleff->getEffects(), lit)) return true;
  else return false;
};

bool instantiatedOp::isGoalMetByEffect(VAL::cond_effect * ceff, const Literal * lit)
{
  if (isGoalMetByEffect(ceff->getEffects(), lit)) return true;
  else return false;
};

bool instantiatedOp::isGoalMetByEffect(VAL::timed_effect * teff, const Literal * lit)
{
  if (isGoalMetByEffect(teff->effs, lit)) return true;
  else return false;
};



};
