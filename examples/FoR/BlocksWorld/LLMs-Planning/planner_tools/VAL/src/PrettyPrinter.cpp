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

#include "PrettyPrinter.h"
#include "Utils.h"
#include "ptree.h"
#include <set>
#include <algorithm>
#include "FuncExp.h"


using std::set;
using std::cerr;
using std::for_each;
using std::find;

//#define set std::set
namespace VAL {

void PrettyPrinter::write_symbol(ostream & o,const symbol * p)
{
	o << p->getName();
};

void PrettyPrinter::write_const_symbol(ostream & o,const const_symbol * p)
{
	o << p->getName();
	if(showType && p->type) 
	{
		showType = false;
		o << " - " << *(p->type);
		showType = true;
	};
};

void PrettyPrinter::write_var_symbol(ostream & o,const var_symbol * p)
{
	o << "?" << p->getName();
	if(showType && p->type) 
	{
		showType = false;
		o << " - " << *(p->type);
		showType = true;
	};
};

void PrettyPrinter::write_pddl_typed_symbol(ostream & o,const pddl_typed_symbol * p)
{
	o << p->getName();
	if(showType && p->type) 
	{
		showType = false;
		o << " - " << *(p->type);
		showType = true;
	};
};

void PrettyPrinter::write_plus_expression(ostream & o,const plus_expression * p)
{
	o << "(+ " << *(p->getLHS()) << " " << *(p->getRHS()) << ")";
};

void PrettyPrinter::write_minus_expression(ostream & o,const minus_expression * p)
{
	o << "(- " << *(p->getLHS()) << " " << *(p->getRHS()) << ")";
};

void PrettyPrinter::write_mul_expression(ostream & o,const mul_expression * p)
{
	o << "(* " << *(p->getLHS()) << " " << *(p->getRHS()) << ")";
};

void PrettyPrinter::write_div_expression(ostream & o,const div_expression * p)
{
	o << "(/ " << *(p->getLHS()) << " " << *(p->getRHS()) << ")";
};

void PrettyPrinter::write_uminus_expression(ostream & o,const uminus_expression * p)
{
	o << "(- " << *(p->getExpr()) << ")";
};

void PrettyPrinter::write_int_expression(ostream & o,const int_expression * p)
{
	o << (int) p->double_value();
};

void PrettyPrinter::write_float_expression(ostream & o,const float_expression * p)
{
	o << p->double_value();
};


void PrettyPrinter::write_special_val_expr(ostream & o,const special_val_expr * p)
{
	switch(p->getKind())
	{
		case E_HASHT:
			o << "#t";
			break;
		case E_DURATION_VAR:
		case E_TOTAL_TIME:
		default:
			cerr << "Not handling duration variables or total time value\n";
			break;
	};
};

void PrettyPrinter::write_func_term(ostream & o,const func_term * p)
{
	o << "(";
	p->getFunction()->func_symbol::write(o);
	for(parameter_symbol_list::const_iterator i = p->getArgs()->begin();i != p->getArgs()->end();++i)
	{
		o << " ";
		if(dynamic_cast<var_symbol*>(*i)) 
		{
			dynamic_cast<var_symbol*>(*i)->var_symbol::write(o);
		}
		else
		{
			dynamic_cast<const_symbol*>(*i)->const_symbol::write(o);
		};
	};
	o << ")";
};

void PrettyPrinter::write_assignment(ostream & o,const assignment * p)
{

 	o << "(";
	switch(p->getOp())
	{
		case E_ASSIGN:
			if(inInitial) 
			{
				o << "= ";
			}
			else 
			{
				o << "assign ";
			};
			break;
		case E_INCREASE:
			o << "increase ";
			break;
		case E_DECREASE:
			o << "decrease ";
			break;
		case E_SCALE_UP:
			o << "scale-up ";
			break;
		case E_SCALE_DOWN:
			o << "scale-down ";
			break;
		default:
			break;
	};
	p->getFTerm()->write(o);
	o << " ";
	p->getExpr()->write(o);
	o << ")";

};

struct Writer {
	ostream & o;

	Writer(ostream & oo) : o(oo) {};

	template<class typ>
	void operator()(const typ * p) {o << " "; p->typ::write(o);};
};

struct NotWriter {
	ostream & o;

	NotWriter(ostream & oo) : o(oo) {};

	template<class typ>
	void operator()(const typ * p) {o << " (not "; p->typ::write(o);o << ")";};
};

void PrettyPrinter::write_goal_list(ostream & o,const goal_list * p)
{
	for(goal_list::const_iterator i = p->begin();i != p->end();++i)
	{
		o << " " << **i;
	};
};

void PrettyPrinter::write_simple_goal(ostream & o,const simple_goal * p)
{
	if(p->getPolarity()==E_NEG) 
	{
		o << "(not " << *(p->getProp()) << ")";
	}
	else
	{
		o << *(p->getProp());
	};
};

void PrettyPrinter::write_qfied_goal(ostream & o,const qfied_goal * p)
{
	switch(p->getQuantifier())
	{
		case E_FORALL:
			o << "(forall (";
			break;
		default:
			o << "(exists (";
			break;
	};
	showType = true;
	for(var_symbol_list::const_iterator i = p->getVars()->begin();i != p->getVars()->end();++i)
	{
		o << **i << " ";
	};
	showType = false;
	o << ") " << *(p->getGoal()) << ")";
};

void PrettyPrinter::write_conj_goal(ostream & o,const conj_goal * p)
{
	o << "(and " << *(p->getGoals()) << ")";
	
};

void PrettyPrinter::write_disj_goal(ostream & o,const disj_goal * p)
{
	o << "(or " << *(p->getGoals()) << ")";
};

void PrettyPrinter::write_timed_goal(ostream & o,const timed_goal * p)
{
	switch(p->getTime())
	{
		case E_AT_END:
			o << "(at end ";
			break;
		case E_AT_START:
			o << "(at start ";
			break;
		default:
			o << "(over all ";
			break;
	};
	o << *(p->getGoal()) << ")";
	
};

void PrettyPrinter::write_imply_goal(ostream & o,const imply_goal * p)
{
	o << "(implies " << *(p->getAntecedent()) << " " << *(p->getConsequent()) << ")";
};

void PrettyPrinter::write_neg_goal(ostream & o,const neg_goal * p)
{
	o << "(not " << *(p->getGoal()) << ")";
};

void PrettyPrinter::write_comparison(ostream & o,const comparison * p)
{
	o << "(";
	switch(p->getOp())
	{
		case E_GREATER:
			o << "> ";
			break;
		case E_GREATEQ:
			o << ">= ";
			break;
		case E_LESS:
			o << "< ";
			break;
		case E_LESSEQ:
			o << "<= ";
			break;
		case E_EQUALS:
			o << "= ";
			break;
		default:
			break;
	};
	o << *(p->getLHS()) << " " << *(p->getRHS()) << ")";
};

void PrettyPrinter::write_proposition(ostream & o,const proposition * p)
{
	o << "(" << p->head->getName();
	for(parameter_symbol_list::const_iterator i = p->args->begin();i != p->args->end();++i)
	{
		o << " ";
		if(dynamic_cast<var_symbol*>(*i)) 
		{
			dynamic_cast<var_symbol*>(*i)->var_symbol::write(o);
		}
		else
		{
			dynamic_cast<const_symbol*>(*i)->const_symbol::write(o);
		};
	};
	o << ")";
};

void PrettyPrinter::write_pred_decl(ostream & o,const pred_decl * p)
{
	o << "(";
	p->getPred()->symbol::write(o);
	p->getArgs()->var_symbol_list::write(o);
	o << ")";
};

void PrettyPrinter::write_func_decl(ostream & o,const func_decl * p)
{
	o << "(";
	p->getFunction()->symbol::write(o);
	p->getArgs()->var_symbol_list::write(o);
	o << ")";
};

void PrettyPrinter::write_simple_effect(ostream & o,const simple_effect * p)
{
	o << *(p->prop);
};

void PrettyPrinter::write_forall_effect(ostream & o,const forall_effect * p)
{
	o << "(forall (";
	for(var_symbol_table::const_iterator i = p->getVars()->begin();i != p->getVars()->end();++i)
	{
		o << "?" << (i->first) << " ";
	};
	o << ") ";
	// We need to add the type conditions to the effect. These are preconditions of a 
	// conditional effect. If the effect is already conditional then we simply add the
	// preconditions, but otherwise we need to create a new conditional effect.  
	
	if(p->getEffects()->cond_effects.empty())
	{
		o << "(when ";
		bool tt = (p->getVars()->size() > 1);
		if(tt) o << "(and ";
		for(var_symbol_table::const_iterator i = p->getVars()->begin();i != p->getVars()->end();++i)
		{
			if(i->second->type)
			{
				o << "(" << i->second->type->getName() << " ?" << (i->first) << ") ";
			}
			else
			{
				o << "(or ";
				for(pddl_type_list::const_iterator j = i->second->either_types->begin();j != i->second->either_types->end();++j)
				{
					o << " (";
					(*j)->type->symbol::write(o);
					o << " " << i->first << ")";
				};
				o << ") ";
			};
		};
		if(tt) o << ") ";
		o << *(p->getEffects()) << ")";
		
	}
	else
	{
		if(p->getEffects()->cond_effects.size()==1 && 
			p->getEffects()->add_effects.empty() &&
			p->getEffects()->del_effects.empty() &&
			p->getEffects()->forall_effects.empty() &&
			p->getEffects()->timed_effects.empty())
		{
			o << "(when (and ";
			for(var_symbol_table::const_iterator i = p->getVars()->begin();i != p->getVars()->end();++i)
			{
				if(i->second->type)
				{
					o << "(" << i->second->type->getName() << " ?" << (i->first) << ") ";
				}
				else
				{
					o << "(or ";
					for(pddl_type_list::const_iterator j = i->second->either_types->begin();j != i->second->either_types->end();++j)
					{
						o << " (";
						(*j)->type->symbol::write(o);
						o << " " << i->first << ")";
					};
					o << ")";
				};
			};
			if(const conj_goal * cg = dynamic_cast<const conj_goal*>(p->getEffects()->cond_effects.front()->getCondition()))
			{
				o << *(cg->getGoals());
			}
			else
			{
				o << *(p->getEffects()->cond_effects.front()->getCondition());
			};
			o << ") " << *(p->getEffects()->cond_effects.front()->getEffects()) << ")";
		}
		else
		{
			cerr << "Complex quantified/conditional effect not yet handled!\n";
		};
	};
};

void PrettyPrinter::write_cond_effect(ostream & o,const cond_effect * p)
{
	o << "(when " << *(p->getCondition()) << " " << *(p->getEffects()) << ")";
};

void PrettyPrinter::write_timed_effect(ostream & o,const timed_effect * p)
{
	 o << *(p->effs);
};

void PrettyPrinter::write_effect_lists(ostream & o,const effect_lists * p)
{
	bool tt = (p->add_effects.size() + p->del_effects.size() + p->forall_effects.size() +
		p->cond_effects.size() + p->assign_effects.size() + p->timed_effects.size() > 1);
		

	if(tt || firstCall) o << "(and";
	bool f = firstCall;
	firstCall = false;
	for_each(p->add_effects.begin(),p->add_effects.end(),Writer(o));
	for_each(p->del_effects.begin(),p->del_effects.end(),NotWriter(o));
	for_each(p->forall_effects.begin(),p->forall_effects.end(),Writer(o));
	for_each(p->cond_effects.begin(),p->cond_effects.end(),Writer(o));
	for_each(p->assign_effects.begin(),p->assign_effects.end(),Writer(o));
	for_each(p->timed_effects.begin(),p->timed_effects.end(),Writer(o));
	firstCall = f;
	if(tt || firstCall) o << ")";
};

void PrettyPrinter::write_operator_(ostream & o,const operator_ * p)
{
};

void PrettyPrinter::write_action(ostream & o,const action * p)
{
	o << "(:action " << p->name->getName() << "\n :parameters (";
	p->parameters->var_symbol_list::write(o);
	showType = false;
	o << ")\n :precondition\n\t(and ";
	if(conj_goal * cg = dynamic_cast<conj_goal*>(p->precondition))
	{
		o << *(cg->getGoals());
	}
	else
	{
		o << *(p->precondition);
	};
	o << ")\n :effect\n\t";
	p->effects->effect_lists::write(o);
	o << ")\n\n";
	showType = true;
};

void PrettyPrinter::write_event(ostream & o,const event * p)
{
	o << "(:event " << p->name->getName() << "\n :parameters (";
	p->parameters->var_symbol_list::write(o);
	showType = false;
	o << ")\n :precondition\n\t(and ";
	if(conj_goal * cg = dynamic_cast<conj_goal*>(p->precondition))
	{
		o << *(cg->getGoals());
	}
	else
	{
		o << *(p->precondition);
	};
	o << ")\n :effect\n\t";
	p->effects->effect_lists::write(o);
	o << ")\n\n";
	showType = true;
};

void PrettyPrinter::write_process(ostream & o,const process * p)
{
	o << "(:process " << p->name->getName() << "\n :parameters (";
	p->parameters->var_symbol_list::write(o);
	showType = false;
	o << ")\n :precondition\n\t(and ";
	if(conj_goal * cg = dynamic_cast<conj_goal*>(p->precondition))
	{
		o << *(cg->getGoals());
	}
	else
	{
		o << *(p->precondition);
	};
	o << ")\n :effect\n\t";
	p->effects->effect_lists::write(o);
	o << ")\n\n";
	showType = true;
};

void collect_symbols(var_symbol_list & vs,const expression * e)
{
	if(const func_term * f = dynamic_cast<const func_term*>(e))
	{
		for(parameter_symbol_list::const_iterator i = f->getArgs()->begin();
				i != f->getArgs()->end();++i)
		{
			if(var_symbol* v = const_cast<var_symbol*>(dynamic_cast<const var_symbol*>(*i)))
			{
				if(find(vs.begin(),vs.end(),v)==vs.end()) vs.push_back(v);
			}
			else
			{
				cerr << "Handle constants in duration constraints manually, please\n";
				exit(1);
			};
		};
	}
	else
	{
		if(const binary_expression * b = dynamic_cast<const binary_expression*>(e))
		{
			collect_symbols(vs,b->getLHS());
			collect_symbols(vs,b->getRHS());
		}
		else if(const uminus_expression * u = dynamic_cast<const uminus_expression*>(e))
		{
			collect_symbols(vs,u->getExpr());
		};
	};
};
		
void PrettyPrinter::write_durative_action(ostream & o,const durative_action * p)
{
	o << "(:durative-action " << p->name->getName() << "\n :parameters (";
	p->parameters->var_symbol_list::write(o);
	showType = false;
	o << " :duration ...\n";
	o << ")\n :condition\n\t(and ";
	if(conj_goal * cg = dynamic_cast<conj_goal*>(p->precondition))
	{
		o << *(cg->getGoals());
	}
	else
	{
		o << *(p->precondition);
	};
	o << ")\n :effect\n\t";
	p->effects->effect_lists::write(o);
	o << ")\n\n";
	showType = true;	
};



void PrettyPrinter::write_domain(ostream & o,const domain * p)
{
	string s = pddl_req_flags_string(p->req ^ E_DURATIVE_ACTIONS);
	o << "(define (domain " << p->name << ")\n(:requirements " << s << ":continuous-effects)\n";
	if(p->types)
	{
		o << "(:types " << *(p->types) << ")\n";
	};
	o << "(:predicates\n\t";
	if(p->predicates)
	{
		for_each(p->predicates->begin(),p->predicates->end(),Writer(o));
	};
//	PredWriter pw(o);
//	p->ops->visit(&pw);
	o << ")\n";


	if(p->functions) 
	{
		o << "(:functions\n\t";
		for_each(p->functions->begin(),p->functions->end(),Writer(o));
		o << ")\n";
	};

	if(p->constants)
	{
		o << "(:constants\n\t";
		for_each(p->constants->begin(),p->constants->end(),Writer(o));
		o << ")\n";
	};
	o << "\n";
	
	p->ops->write(o);

	o << ")\n";
	
};

void PrettyPrinter::write_metric_spec(ostream & o,const metric_spec * p)
{
/*
 	switch(p->opt)
	{
		case E_MAXIMIZE:
			o << "(:metric maximize ";
			break;
		case E_MINIMIZE:
			o << "(:metric minimize ";
			break;
		default:
			break;
	};
	o << *(p->expr) << ")\n";
*/
};

void PrettyPrinter::write_length_spec(ostream & o,const length_spec * p)
{
	// Ignore this anyway.
};

void PrettyPrinter::write_problem(ostream & o,const problem * p)
{
	o << "(define (problem " << p->name << ")\n\t(:domain " << p->domain_name << ")\n\t(:objects";
	for_each(p->objects->begin(),p->objects->end(),Writer(o));
	o << ")\n\t(:init ";
	inInitial = true;
	showType = false;
	for_each(p->initial_state->add_effects.begin(),p->initial_state->add_effects.end(),Writer(o));
	for_each(p->initial_state->assign_effects.begin(),p->initial_state->assign_effects.end(),Writer(o));
	inInitial = false;
	
	o << ")\n\t(:goal " << *(p->the_goal) << ")\n";
	if(p->metric) o << *(p->metric);
	o << ")\n";
};


void PrettyPrinter::write_plan_step(ostream & o,const plan_step * p)
{
	cerr << "Type stripping is not an appropriate operation for plans!\n";
};





/*
string duration_expression::createAll(State & s)
{
	stringstream dur;
	
	if(const num_expression * val = dynamic_cast<const num_expression*>(exp))
	{
		dur << nm << " = " << val->double_value() << "\n";
		return dur.str();
	};
	
	Environment bs;
	if(!bindAll(bs,s,exp)) return "";
	recordOne(dur,s,bs);
	done = false;
	while(nextBinding(bs,s,exp))
	{
		done = false;
		if(s.safeBinding(bs,exp)) recordOne(dur,s,bs);
	};
	return dur.str();
};
*/

};

