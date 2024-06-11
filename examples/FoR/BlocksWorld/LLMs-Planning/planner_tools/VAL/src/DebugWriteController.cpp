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

#include <iostream>
#include "DebugWriteController.h"

namespace VAL {

void DebugWriteController::write_symbol(ostream & o,const symbol * p)
{
	p->display(indent);
};

void DebugWriteController::write_const_symbol(ostream & o,const const_symbol * p)
{
	p->display(indent);
};

void DebugWriteController::write_class_symbol(ostream & o,const class_symbol * p)
{
	p->display(indent);
};

void DebugWriteController::write_var_symbol(ostream & o,const var_symbol * p)
{
	p->display(indent);
};

void DebugWriteController::write_pddl_typed_symbol(ostream & o,const pddl_typed_symbol * p)
{
	p->display(indent);
};

void DebugWriteController::write_plus_expression(ostream & o,const plus_expression * p)
{
	p->display(indent);
};

void DebugWriteController::write_minus_expression(ostream & o,const minus_expression * p)
{
	p->display(indent);
};

void DebugWriteController::write_mul_expression(ostream & o,const mul_expression * p)
{
	p->display(indent);
};

void DebugWriteController::write_div_expression(ostream & o,const div_expression * p)
{
	p->display(indent);
};

void DebugWriteController::write_uminus_expression(ostream & o,const uminus_expression * p)
{
	p->display(indent);
};

void DebugWriteController::write_int_expression(ostream & o,const int_expression * p)
{
	p->display(indent);
};

void DebugWriteController::write_float_expression(ostream & o,const float_expression * p)
{
	p->display(indent);
};

void DebugWriteController::write_special_val_expr(ostream & o,const special_val_expr * p)
{
	p->display(indent);
};

void DebugWriteController::write_func_term(ostream & o,const func_term * p)
{
	p->display(indent);
};

void DebugWriteController::write_class_func_term(ostream & o,const class_func_term * p)
{
	p->display(indent);
};

void DebugWriteController::write_assignment(ostream & o,const assignment * p)
{
	p->display(indent);
};

void DebugWriteController::write_goal_list(ostream & o,const goal_list * p)
{
	p->display(indent);
};

void DebugWriteController::write_simple_goal(ostream & o,const simple_goal * p)
{
	p->display(indent);
};

void DebugWriteController::write_qfied_goal(ostream & o,const qfied_goal * p)
{
	p->display(indent);
};

void DebugWriteController::write_conj_goal(ostream & o,const conj_goal * p)
{
	p->display(indent);
};

void DebugWriteController::write_disj_goal(ostream & o,const disj_goal * p)
{
	p->display(indent);
};

void DebugWriteController::write_timed_goal(ostream & o,const timed_goal * p)
{
	p->display(indent);
};

void DebugWriteController::write_imply_goal(ostream & o,const imply_goal * p)
{
	p->display(indent);
};

void DebugWriteController::write_neg_goal(ostream & o,const neg_goal * p)
{
	p->display(indent);
};

void DebugWriteController::write_comparison(ostream & o,const comparison * p)
{
	p->display(indent);
};

void DebugWriteController::write_proposition(ostream & o,const proposition * p)
{
	p->display(indent);
};

void DebugWriteController::write_pred_decl_list(ostream & o,const pred_decl_list * p)
{
	p->display(indent);
};

void DebugWriteController::write_func_decl_list(ostream & o,const func_decl_list * p)
{
	p->display(indent);
};

void DebugWriteController::write_pred_decl(ostream & o,const pred_decl * p)
{
	p->display(indent);
};

void DebugWriteController::write_func_decl(ostream & o,const func_decl * p)
{
	p->display(indent);
};

void DebugWriteController::write_simple_effect(ostream & o,const simple_effect * p)
{
	p->display(indent);
};

void DebugWriteController::write_forall_effect(ostream & o,const forall_effect * p)
{
	p->display(indent);
};

void DebugWriteController::write_cond_effect(ostream & o,const cond_effect * p)
{
	p->display(indent);
};

void DebugWriteController::write_timed_effect(ostream & o,const timed_effect * p)
{
	p->display(indent);
};

void DebugWriteController::write_timed_initial_literal(ostream & o,const timed_initial_literal * p)
{
	p->display(indent);
};

void DebugWriteController::write_effect_lists(ostream & o,const effect_lists * p)
{
	p->display(indent);
};

void DebugWriteController::write_operator_list(ostream & o,const operator_list * p)
{
	p->display(indent);
};

void DebugWriteController::write_derivations_list(ostream & o,const derivations_list * d)
{
	d->display(indent);
};

void DebugWriteController::write_derivation_rule(ostream & o,const derivation_rule * d)
{
	d->display(indent);
};

void DebugWriteController::write_operator_(ostream & o,const operator_ * p)
{
	p->display(indent);
};

void DebugWriteController::write_action(ostream & o,const action * p)
{
	p->display(indent);
};

void DebugWriteController::write_event(ostream & o,const event * p)
{
	p->display(indent);
};

void DebugWriteController::write_process(ostream & o,const process * p)
{
	p->display(indent);
};

void DebugWriteController::write_durative_action(ostream & o,const durative_action * p)
{
	p->display(indent);
};

void DebugWriteController::write_class_def(ostream & o,const class_def * p)
{
	p->display(indent);
};

void DebugWriteController::write_domain(ostream & o,const domain * p)
{
	p->display(indent);
};

void DebugWriteController::write_metric_spec(ostream & o,const metric_spec * p)
{
	p->display(indent);
};

void DebugWriteController::write_length_spec(ostream & o,const length_spec * p)
{
	p->display(indent);
};

void DebugWriteController::write_problem(ostream & o,const problem * p)
{
	p->display(indent);
};

void DebugWriteController::write_plan_step(ostream & o,const plan_step * p)
{
	p->display(indent);
};

void DebugWriteController::write_preference(ostream & o,const preference * p)
{
	p->display(indent);
};

void DebugWriteController::write_constraint_goal(ostream & o,const constraint_goal * cg)
{
	cg->display(indent);
};

void DebugWriteController::write_violation_term(ostream & o,const violation_term * vt)
{
	vt->display(indent);
};

};

