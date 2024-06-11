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

#ifndef __ANALYSER
#define __ANALYSER

#include <vector>
#include <iostream>

using std::vector;
using std::ostream;

#include "ptree.h"
#include "VisitController.h"

namespace VAL {

class extended_pred_symbol : public pred_symbol {
private:
	int initialState;
	int goalState;
	vector<operator_*> preconds;
	vector<operator_*> adds;
	vector<operator_*> dels;
	
	vector<derivation_rule*> derivedPreconds;
	vector<derivation_rule*> derivedAdds;

public:
	extended_pred_symbol(const string & nm) : pred_symbol(nm), 
			initialState(0), goalState(0), preconds(),
			adds(), dels(), derivedPreconds(), derivedAdds() {};

	void setGoal() {++goalState;};
	void setInitial() {++initialState;};
	int isGoal() const {return goalState;};
	int isInitial() const {return initialState;};

	bool isStatic() const {return adds.empty() && derivedAdds.empty() && dels.empty();};
	bool decays() const {return adds.empty() && derivedAdds.empty() && !dels.empty();};
	
	void addPre(operator_ * o) {preconds.push_back(o);};
	void addPre(derivation_rule * o) {derivedPreconds.push_back(o);};

	void addAdd(operator_ * o) {adds.push_back(o);};
	void addAdd(derivation_rule * o) {derivedAdds.push_back(o);};

	void addDel(operator_ * o) {dels.push_back(o);};

	void write(ostream & o) const
	{
		o << "\nReport for: " << getName() << "\n---------\n";
		o << "Initial: " << initialState << " Goal: " << goalState <<
			"\nPreconditions:\n";
		for(vector<operator_*>::const_iterator i = preconds.begin();
				i != preconds.end();++i)
		{
			if(*i) o << "\t" << (*i)->name->getName() << "\n";
		};
		if(isStatic()) 
		{
			o << "Proposition is static\n";
			return;
		};
		if(decays())
		{
			o << "Proposition decays only\n";
		}
		else
		{
			o << "Adds:\n";
			for(vector<operator_*>::const_iterator i = adds.begin();
					i != adds.end();++i)
			{
				if(*i) o << "\t" << (*i)->name->getName() << "\n";
			};
		};
		o << "Dels:\n";
		for(vector<operator_*>::const_iterator i = dels.begin();
				i != dels.end();++i)
		{
			if(*i) o << "\t" << (*i)->name->getName() << "\n";
		};
	};
	void display(int i) const
	{
		write(cout);
	};
	void visit(VisitController * v) const
	{
		write(cout);
	};
};

#define EPS(x) static_cast<VAL::extended_pred_symbol*>(const_cast<VAL::pred_symbol*>(x))


class Analyser : public VisitController {
private:
	bool initially;
	bool finally;
	bool adding;
	operator_ * op;
	derivation_rule * drv;
public:
	Analyser() : initially(false), finally(false), adding(true), op(0), drv(0) {};
	virtual void visit_pred_decl(pred_decl * p) 
	{
		p->getPred()->visit(this);
	};
	virtual void visit_simple_goal(simple_goal * p) 
	{
		if(finally) 
		{
			EPS(p->getProp()->head)->setGoal();
		}
		else
		{
			if (op) EPS(p->getProp()->head)->addPre(op);
			if (drv) EPS(p->getProp()->head)->addPre(drv);
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
	virtual void visit_simple_effect(simple_effect * p) 
	{
		if(initially)
		{
			EPS(p->prop->head)->setInitial();
		}
		else
		{
			if(adding)
			{
				EPS(p->prop->head)->addAdd(op);
			}
			else
			{
				EPS(p->prop->head)->addDel(op);
			};
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
	};
	virtual void visit_derivation_rule(derivation_rule * r)
	{
		drv = r;
		adding = true;
		r->get_body()->visit(this);
		EPS(r->get_head()->head)->addAdd(drv);
		drv = 0;
	};
	virtual void visit_operator_(operator_ * p) 
	{
		op = p;
		adding = true;
		p->precondition->visit(this);
		p->effects->visit(this);
		op = 0;
	};
	virtual void visit_action(action * p)
	{
		visit_operator_(static_cast<operator_*>(p));
	};
	virtual void visit_durative_action(durative_action * p) 
	{
		visit_operator_(static_cast<operator_*>(p));
	};
	virtual void visit_problem(problem * p) 
	{
		initially = true;
		p->initial_state->visit(this);
		initially = false;
		finally = true;
		if(p->the_goal) p->the_goal->visit(this);
		finally = false;
	};

	virtual void visit_domain(domain * p) 
	{
		visit_operator_list(p->ops);
		if (p->drvs) visit_derivations_list(p->drvs);
	};
};

};

#endif
