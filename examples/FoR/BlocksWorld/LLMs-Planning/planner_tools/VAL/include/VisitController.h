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

#ifndef __VISITORCONTROLLER
#define __VISITORCONTROLLER

#include "ptree.h"

namespace VAL {

struct VisitController {
	virtual ~VisitController() {};

	virtual void visit_symbol(symbol *) {};
	virtual void visit_pred_symbol(pred_symbol *) {};
	virtual void visit_func_symbol(func_symbol *) {};
	virtual void visit_const_symbol(const_symbol *) {};
	virtual void visit_class_symbol(const_symbol *) {};
	virtual void visit_var_symbol(var_symbol *) {};
	virtual void visit_pddl_typed_symbol(pddl_typed_symbol *) {};
	virtual void visit_plus_expression(plus_expression *) {};
	virtual void visit_minus_expression(minus_expression *) {};
	virtual void visit_mul_expression(mul_expression *) {};
	virtual void visit_div_expression(div_expression *) {};
	virtual void visit_uminus_expression(uminus_expression *) {};
	virtual void visit_int_expression(int_expression *) {};
	virtual void visit_float_expression(float_expression *) {};
	virtual void visit_special_val_expr(special_val_expr *) {};
	virtual void visit_violation_term(violation_term *) {};
	virtual void visit_func_term(func_term *) {};
	virtual void visit_class_func_term(class_func_term *) {};
	virtual void visit_assignment(assignment *) {};
	virtual void visit_goal_list(goal_list * p) {p->pc_list<goal*>::visit(this);};
	virtual void visit_constraint_goal(constraint_goal *) {};
	virtual void visit_preference(preference *) {};
	virtual void visit_simple_goal(simple_goal *) {};
	virtual void visit_qfied_goal(qfied_goal *) {};
	virtual void visit_conj_goal(conj_goal *) {};
	virtual void visit_disj_goal(disj_goal *) {};
	virtual void visit_timed_goal(timed_goal *) {};
	virtual void visit_imply_goal(imply_goal *) {};
	virtual void visit_neg_goal(neg_goal *) {};
	virtual void visit_comparison(comparison *) {};
	virtual void visit_proposition(proposition *) {};
	virtual void visit_pred_decl_list(pred_decl_list * p) {p->pc_list<pred_decl*>::visit(this);};
	virtual void visit_func_decl_list(func_decl_list * p) {p->pc_list<func_decl*>::visit(this);};
	virtual void visit_pred_decl(pred_decl *) {};
	virtual void visit_func_decl(func_decl *) {};
	virtual void visit_simple_effect(simple_effect *) {};
	virtual void visit_forall_effect(forall_effect *) {};
	virtual void visit_cond_effect(cond_effect *) {};
	virtual void visit_timed_effect(timed_effect *) {};
	virtual void visit_timed_initial_literal(timed_initial_literal *) {};
	virtual void visit_effect_lists(effect_lists *) {};
	virtual void visit_operator_list(operator_list * p) {p->pc_list<operator_*>::visit(this);};
	virtual void visit_derivations_list(derivations_list * d) {d->pc_list<derivation_rule*>::visit(this);};
	virtual void visit_derivation_rule(derivation_rule * d) {};
	virtual void visit_operator_(operator_ *) {};
	virtual void visit_action(action *) {};
	virtual void visit_event(event *) {};
	virtual void visit_process(process *) {};
	virtual void visit_durative_action(durative_action *) {};
	virtual void visit_class_def(class_def *) {};
	virtual void visit_domain(domain *) {};
	virtual void visit_metric_spec(metric_spec *) {};
	virtual void visit_length_spec(length_spec *) {};
	virtual void visit_problem(problem *) {};
	virtual void visit_plan_step(plan_step *) {};
	
	virtual void visit_symbol(const symbol * s) {visit_symbol(const_cast<symbol *>(s));};
	virtual void visit_pred_symbol(const pred_symbol * s) {visit_pred_symbol(const_cast<pred_symbol *>(s));};
	virtual void visit_func_symbol(const func_symbol * s) {visit_func_symbol(const_cast<func_symbol *>(s));};
	virtual void visit_const_symbol(const const_symbol * s) {visit_const_symbol(const_cast<const_symbol *>(s));};
	virtual void visit_class_symbol(const class_symbol * s) {visit_class_symbol(const_cast<class_symbol *>(s));};
	virtual void visit_var_symbol(const var_symbol * s) {visit_var_symbol(const_cast<var_symbol *>(s));};
	virtual void visit_pddl_typed_symbol(const pddl_typed_symbol * s) {visit_pddl_typed_symbol(const_cast<pddl_typed_symbol *>(s));};
	virtual void visit_plus_expression(const plus_expression * s) {visit_plus_expression(const_cast<plus_expression *>(s));};
	virtual void visit_minus_expression(const minus_expression * s) {visit_minus_expression(const_cast<minus_expression *>(s));};
	virtual void visit_mul_expression(const mul_expression * s) {visit_mul_expression(const_cast<mul_expression *>(s));};
	virtual void visit_div_expression(const div_expression * s) {visit_div_expression(const_cast<div_expression *>(s));};
	virtual void visit_uminus_expression(const uminus_expression * s) {visit_uminus_expression(const_cast<uminus_expression *>(s));};
	virtual void visit_int_expression(const int_expression * s) {visit_int_expression(const_cast<int_expression *>(s));};
	virtual void visit_float_expression(const float_expression * s) {visit_float_expression(const_cast<float_expression *>(s));};
	virtual void visit_special_val_expr(const special_val_expr * s) {visit_special_val_expr(const_cast<special_val_expr *>(s));};
	virtual void visit_violation_term(const violation_term * v) {visit_violation_term(const_cast<violation_term *>(v));};
	virtual void visit_func_term(const func_term * s) {visit_func_term(const_cast<func_term *>(s));};
	virtual void visit_class_func_term(const class_func_term * s) {visit_class_func_term(const_cast<class_func_term *>(s));};
	virtual void visit_assignment(const assignment * s) {visit_assignment(const_cast<assignment *>(s));};
	virtual void visit_goal_list(const goal_list * p) {visit_goal_list(const_cast<goal_list*>(p));};
	virtual void visit_constraint_goal(const constraint_goal * cg) {visit_constraint_goal(const_cast<constraint_goal *>(cg));};
	virtual void visit_preference(const preference * p) {visit_preference(const_cast<preference *>(p));};
	virtual void visit_simple_goal(const simple_goal * s) {visit_simple_goal(const_cast<simple_goal *>(s));};
	virtual void visit_qfied_goal(const qfied_goal * s) {visit_qfied_goal(const_cast<qfied_goal *>(s));};
	virtual void visit_conj_goal(const conj_goal * s) {visit_conj_goal(const_cast<conj_goal *>(s));};
	virtual void visit_disj_goal(const disj_goal * s) {visit_disj_goal(const_cast<disj_goal *>(s));};
	virtual void visit_timed_goal(const timed_goal * s) {visit_timed_goal(const_cast<timed_goal *>(s));};
	virtual void visit_imply_goal(const imply_goal * s) {visit_imply_goal(const_cast<imply_goal *>(s));};
	virtual void visit_neg_goal(const neg_goal * s) {visit_neg_goal(const_cast<neg_goal *>(s));};
	virtual void visit_comparison(const comparison * s) {visit_comparison(const_cast<comparison *>(s));};
	virtual void visit_proposition(const proposition * s) {visit_proposition(const_cast<proposition *>(s));};
	virtual void visit_pred_decl_list(const pred_decl_list * p) {visit_pred_decl_list(const_cast<pred_decl_list*>(p));};
	virtual void visit_func_decl_list(const func_decl_list * p) {visit_func_decl_list(const_cast<func_decl_list*>(p));};
	virtual void visit_pred_decl(const pred_decl * s) {visit_pred_decl(const_cast<pred_decl *>(s));};
	virtual void visit_func_decl(const func_decl * s) {visit_func_decl(const_cast<func_decl *>(s));};
	virtual void visit_simple_effect(const simple_effect * s) {visit_simple_effect(const_cast<simple_effect *>(s));};
	virtual void visit_forall_effect(const forall_effect * s) {visit_forall_effect(const_cast<forall_effect *>(s));};
	virtual void visit_cond_effect(const cond_effect * s) {visit_cond_effect(const_cast<cond_effect *>(s));};
	virtual void visit_timed_effect(const timed_effect * s) {visit_timed_effect(const_cast<timed_effect *>(s));};
	virtual void visit_timed_initial_literal(const timed_initial_literal * s) {visit_timed_initial_literal(const_cast<timed_initial_literal *>(s));};
	virtual void visit_effect_lists(const effect_lists * s) {visit_effect_lists(const_cast<effect_lists *>(s));};
	virtual void visit_operator_list(const operator_list * p) {visit_operator_list(const_cast<operator_list*>(p));};
	virtual void visit_derivations_list(const derivations_list * d) {visit_derivations_list(const_cast<derivations_list*>(d));};
	virtual void visit_derivation_rule(const derivation_rule * s) {visit_derivation_rule(const_cast<derivation_rule*>(s));};
	virtual void visit_operator_(const operator_ * s) {visit_operator_(const_cast<operator_ *>(s));};
	virtual void visit_action(const action * s) {visit_action(const_cast<action *>(s));};
	virtual void visit_event(const event * s) {visit_event(const_cast<event *>(s));};
	virtual void visit_process(const process * s) {visit_process(const_cast<process *>(s));};
	virtual void visit_durative_action(const durative_action * s) {visit_durative_action(const_cast<durative_action *>(s));};
	virtual void visit_class_def(const class_def * s) {visit_class_def(const_cast<class_def *>(s));};
	virtual void visit_domain(const domain * s) {visit_domain(const_cast<domain *>(s));};
	virtual void visit_metric_spec(const metric_spec * s) {visit_metric_spec(const_cast<metric_spec *>(s));};
	virtual void visit_length_spec(const length_spec * s) {visit_length_spec(const_cast<length_spec *>(s));};
	virtual void visit_problem(const problem * s) {visit_problem(const_cast<problem *>(s));};
	virtual void visit_plan_step(const plan_step * s) {visit_plan_step(const_cast<plan_step *>(s));};

};

};


#endif
