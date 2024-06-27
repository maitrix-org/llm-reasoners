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

#include "SimpleEval.h"
#include "TypedAnalyser.h"
#include "instantiation.h"
#include "typecheck.h"

using namespace VAL;

namespace Inst {

const bool simpleEvalDebug = false;

IState InitialStateEvaluator::initState;
IState0Arity InitialStateEvaluator::init0State;


void InitialStateEvaluator::setInitialState()
{
	initState.clear();
	init0State.clear();
	
	for(pc_list<simple_effect*>::const_iterator i = 
				current_analysis->the_problem->initial_state->add_effects.begin();
				i != current_analysis->the_problem->initial_state->add_effects.end();++i)
	{
		if((*i)->prop->args->begin()==(*i)->prop->args->end())
		{
			// Arity 0...
			init0State.insert((*i)->prop->head);

		}
		else
		{
			initState[(*i)->prop->head].push_back((*i)->prop->args);
		};
	};
};

void InitialStateEvaluator::evaluateSimpleGoal(FastEnvironment * f,simple_goal * s)
{
	extended_pred_symbol * eps = EPS(s->getProp()->head);
	if(eps->appearsStatic())
	{
		if(!eps->isCompletelyStatic(f,s->getProp()))
		{
//			cout << s->getProp()->head->getName() << " is a faker\n";
			unknownTrue = true;
			unknownFalse = true;
			return;
		};
		
		if (simpleEvalDebug) cout << s->getProp()->head->getName() << " is static\n";

		unknownTrue = false;
		unknownFalse = false;

		//eps = eps->getPrimitive(f,s->getProp());

		if(eps->contains(f,s->getProp()))
		{
			valueTrue = true;
			valueFalse = false;
		}
		else
		{
			valueTrue = (init0State.find(s->getProp()->head) != init0State.end());
			valueFalse = !valueTrue;
		}
		if(s->getPolarity() == E_NEG)
		{
			const bool vt = valueTrue;
			valueTrue = valueFalse;
			valueFalse = vt;
		};

		return;
	}
	else if(eps->cannotIncrease())
	{
//		cout << "Got one that cannot increase " << *eps << "\n";
		if(s->getPolarity() == E_NEG)
		{
			valueTrue = !valueTrue;
			valueFalse = !valueFalse;
			unknownTrue = true;
			unknownFalse = false;
			return;
		}
		unknownTrue = false;
		unknownFalse = false;
		if(eps->contains(f,s->getProp()))
		{
			valueTrue = true;
			valueFalse = false;
			return;
		}; 
		valueTrue = (init0State.find(s->getProp()->head) != init0State.end());
		valueFalse = !valueTrue;
		return;
		
	}
	unknownTrue = true;
	unknownFalse = true;
};


bool partialMatch(const VAL::const_symbol * x,const VAL::const_symbol * y)
{
	return x==y || x==0 || y==0;
};

void SimpleEvaluator::visit_preference(preference * p)
{};

bool SimpleEvaluator::equiv(const parameter_symbol_list * s,const parameter_symbol_list * p)
{
	parameter_symbol_list::const_iterator y = p->begin();
	for(parameter_symbol_list::const_iterator x = s->begin();x != s->end();++x,++y)
	{
		if((*f)[*x] != *y) return false;
	};
	return true;
};
					
void SimpleEvaluator::visit_simple_goal(simple_goal * s)
{
	if(EPS(s->getProp()->head)->getParent() == this->equality)
	{
//	cout << "Got equality\n";
		unknownTrue = false;
		unknownFalse = false;
		valueTrue = ((*f)[s->getProp()->args->front()] == 
						(*f)[s->getProp()->args->back()]);
		valueFalse = !valueTrue;
		
		if(s->getPolarity() == E_NEG)
		{
			const bool vt = valueTrue;
			valueTrue = valueFalse;
			valueFalse = vt;
		};
		return;
	};
	primev->evaluateSimpleGoal(f,s);
	if (simpleEvalDebug) {
		if (!unknownTrue && valueTrue) {
			cout << "\t\tValue of fact known to be true\n";
		}
		if (!unknownFalse && valueFalse) {
			cout << "\t\tValue of fact known to be false\n";
		}
		if (unknownTrue || unknownFalse) {
			cout << "\t\tValue of fact unknown\n";
		}
	}
};


void SimpleEvaluator::visit_qfied_goal(qfied_goal * p)
{

	vector<vector<VAL::const_symbol*>::const_iterator> vals(p->getVars()->size());
	vector<vector<VAL::const_symbol*>::const_iterator> starts(p->getVars()->size());
	vector<vector<VAL::const_symbol*>::const_iterator> ends(p->getVars()->size());
	vector<VAL::var_symbol *> vars(p->getVars()->size());
	FastEnvironment fe(*f);
	fe.extend(vars.size());
	int i = 0;
	int c = 1;
	for(var_symbol_list::const_iterator pi = p->getVars()->begin();
			pi != p->getVars()->end();++pi,++i)
	{
		if(instantiatedOp::getValues().find((*pi)->type) == instantiatedOp::getValues().end()) 
		{
			instantiatedOp::getValues()[(*pi)->type] = tc->range(*pi);
		};
		vals[i] = starts[i] = instantiatedOp::getValues()[(*pi)->type].begin();
		ends[i] = instantiatedOp::getValues()[(*pi)->type].end();
		if(ends[i]==starts[i]) return;
		fe[(*pi)] = *(vals[i]);
		vars[i] = *pi;
		c *= instantiatedOp::getValues()[(*pi)->type].size();
	};

	
	valueTrue = (p->getQuantifier() == VAL::E_FORALL);
	valueFalse = !valueTrue;
	unknownTrue = false;
	unknownFalse = false;

	bool uTrue = false;
	bool uFalse = false;

	--i;
	while(vals[i] != ends[i])
	{
// This is inefficient because it creates a copy of the environment even if the copy is never used.
// In practice, this should not be a problem because a quantified effect presumably uses the variables
// it quantifies.
		FastEnvironment * const ecpy = f;
		FastEnvironment toPass(fe);
		f = &toPass;
		p->getGoal()->visit(this);
		
		if (p->getQuantifier() == VAL::E_FORALL) {
;			if(reallyFalse()) {
				if (simpleEvalDebug) cout << "Contradictory child of forall\n";
				return;
			}
			uTrue = uTrue || unknownTrue;
			uFalse = uFalse || unknownFalse;
		} else {
			if(reallyTrue()) {
				if (simpleEvalDebug) cout << "Tautologous child of exists\n";
				return;
			}
			uTrue = uTrue || unknownTrue;
			uFalse = uFalse || unknownFalse;
		}
		f = ecpy;

		int x = 0;
		++vals[0];
		if(vals[0] != ends[0]) fe[vars[0]] = *(vals[0]);
		while(x < i && vals[x] == ends[x])
		{
			vals[x] = starts[x];
			fe[vars[x]] = *(vals[x]);
			++x;
			++vals[x];
			if(vals[x] != ends[x]) fe[vars[x]] = *(vals[x]);
		};
	};
	unknownTrue = uTrue;
	unknownFalse = uFalse;
};

void SimpleEvaluator::visit_conj_goal(conj_goal * c)
{
	if (simpleEvalDebug) cout << "And...\n";
	bool uTrue = false;
	bool uFalse = false;

	unknownTrue = false;
	unknownFalse = false;
	valueTrue = true;
	valueFalse = false;
	for(goal_list::const_iterator i = c->getGoals()->begin();
		i != c->getGoals()->end();++i)
	{
		(*i)->visit(this);
		if(reallyFalse()) {
			if (simpleEvalDebug) cout << "Contradictory child of and\n";
			return;
		}
		uTrue = uTrue || unknownTrue;
		uFalse = uFalse || unknownFalse;
	};
	unknownTrue = uTrue;
	unknownFalse = uFalse;
        
        if (simpleEvalDebug) {
            if (!unknownTrue && valueTrue) {
                cout << "\t\tValue of AND known to be true\n";
            }
            if (!unknownFalse && valueFalse) {
                cout << "\t\tValue of AND known to be false\n";
            }
            if (unknownTrue) {                
                cout << "\t\tValue of AND might be true\n";
            }
            if (unknownFalse) {                
                cout << "\t\tValue of AND might be false\n";
            }

        }
};
	
void SimpleEvaluator::visit_disj_goal(disj_goal * d)
{
	if (simpleEvalDebug) cout << "Or...\n";
	bool uTrue = false;
	bool uFalse = false;

	unknownTrue = false;
	unknownFalse = false;
	valueTrue = false;
	valueFalse = true;

	for(goal_list::const_iterator i = d->getGoals()->begin();
		i != d->getGoals()->end();++i)
	{
		(*i)->visit(this);
		if(reallyTrue()) {
			if (simpleEvalDebug) cout << "Tautologous child of or\n";
			return;
		}
		uTrue = uTrue || unknownTrue;
		uFalse = uFalse || unknownFalse;
	};
	unknownTrue = uTrue;
	unknownFalse = uFalse;
};

void SimpleEvaluator::visit_timed_goal(timed_goal * t)
{
	t->getGoal()->visit(this);
};

void SimpleEvaluator::visit_imply_goal(imply_goal * ig)
{
	if (simpleEvalDebug) cout << "Implies...\n";
	ig->getAntecedent()->visit(this);
	if(unknownTrue || unknownFalse) {
		if (simpleEvalDebug) cout << "Implication with an unknown antecedent\n";
		unknownTrue = true;
		unknownFalse = true;
		return;
	}
	if(valueTrue)
	{
		if (simpleEvalDebug) cout << "Antecedent tautologous, checking consequent\n";
		ig->getConsequent()->visit(this);
	}
	else
	{
		if (simpleEvalDebug) cout << "Antecedent contradictory, ex falso quodlibet\n";
		valueTrue = true;
		valueFalse = false;
	}
};

void SimpleEvaluator::visit_neg_goal(neg_goal * ng)
{
	if (simpleEvalDebug) cout << "Negating...\n";
	ng->getGoal()->visit(this);
	if(!unknownTrue && !unknownFalse)
	{
		const bool vt = valueTrue;
		valueTrue = valueFalse;
		valueFalse = vt;
	} else {
		unknownTrue = true;
		unknownFalse = true;
	}

	if (simpleEvalDebug) {
		if (valueTrue) {
			cout << "Now cast as true\n";
		} else if (valueFalse) {
			cout << "Now cast as false\n";
		}
	}
};

void SimpleEvaluator::visit_event(event * op)
{
	op->precondition->visit(this);
};

void SimpleEvaluator::visit_process(process * op)
{
	op->precondition->visit(this);
};


void SimpleEvaluator::visit_comparison(comparison * c)
{
//	unknown = true;
//	return;
	
	isFixed = false;
	undefined = false;
	isDuration = false;
	c->getLHS()->visit(this);
	if(undefined) 
	{
		unknownTrue = false;
		valueTrue = false;
		unknownFalse = false;
		valueFalse = false;

		return;
	};
	if(isDuration)
	{
		valueTrue = true;
		unknownTrue = false;
		valueFalse = false;
		unknownFalse = false;
		return;
	};
	bool lhsFixed = isFixed;
	double lhsval = nvalue;
	//bool lhsDur = isDuration;
	
	isDuration = false;
	c->getRHS()->visit(this);
	if(undefined)
	{
		unknownTrue = valueTrue = false;
		unknownFalse = valueFalse = false;
		return;
	};
	
	isFixed &= lhsFixed;
	if(isFixed)
	{
		unknownTrue = false;
		unknownFalse = false;
		switch(c->getOp())
		{
			case E_GREATER:
				valueTrue = (lhsval > nvalue);  // I think this is a problem case if 
											// we are comparing with ?duration in the
											// special duration field.... 
				break;
			case E_GREATEQ:
				valueTrue = (lhsval >= nvalue);
				break;
			case E_LESS:
				valueTrue = (lhsval < nvalue);
				break;
			case E_LESSEQ:
				valueTrue = (lhsval <= nvalue);
				break;
			default: // E_EQUALS
				valueTrue = (lhsval == nvalue);
		};
		valueFalse = !valueTrue;
	}
	else
	{
		unknownTrue = true;
		unknownFalse = true;
	};
};

void SimpleEvaluator::visit_action(action * op)
{
	if (op->precondition) {
            if (simpleEvalDebug) cout << "Visiting operator preconditions\n";
            op->precondition->visit(this);
            if (simpleEvalDebug) {
                if(reallyTrue()) {
                    cout << "Preconditions are really true\n";
                }
                if (reallyFalse()) {
                    cout << "Preconditions are really false\n";
                }                
            }
        }
};

void SimpleEvaluator::visit_derivation_rule(derivation_rule * drv)
{
	if (drv->get_body()) drv->get_body()->visit(this);
};

void SimpleEvaluator::visit_durative_action(durative_action * da)
{
	if(da->precondition) da->precondition->visit(this);
	if(unknownTrue || valueTrue)
	{
		da->dur_constraint->visit(this);
	};
};



void SimpleEvaluator::visit_plus_expression(plus_expression * s)
{
	s->getLHS()->visit(this);
	double x = nvalue;
	bool lisFixed = isFixed;
	s->getRHS()->visit(this);
	nvalue += x;
	isFixed &= lisFixed;
};

void SimpleEvaluator::visit_minus_expression(minus_expression * s)
{
	s->getLHS()->visit(this);
	double x = nvalue;
	bool lisFixed = isFixed;
	s->getRHS()->visit(this);
	nvalue -= x;
	isFixed &= lisFixed;
};

void SimpleEvaluator::visit_mul_expression(mul_expression * s)
{
	s->getLHS()->visit(this);
	double x = nvalue;
	bool lisFixed = isFixed;
	s->getRHS()->visit(this);
	nvalue *= x;
	isFixed &= lisFixed;
};

void SimpleEvaluator::visit_div_expression(div_expression * s)
{
	s->getRHS()->visit(this);
	double x = nvalue;
	bool risFixed = isFixed;
	s->getLHS()->visit(this);
	isFixed &= risFixed;
	if(x != 0)
	{
		nvalue /= x;
	};
	if(isFixed && x == 0)
	{
		//cout << "Division by zero!\n";
		undefined = true;
	};
};

void SimpleEvaluator::visit_uminus_expression(uminus_expression * s)
{
	s->getExpr()->visit(this);
};

void SimpleEvaluator::visit_int_expression(int_expression * s)
{
	isFixed = true;
	nvalue = s->double_value();
};

void SimpleEvaluator::visit_float_expression(float_expression * s)
{
	isFixed = true;
	nvalue = s->double_value();
};

void SimpleEvaluator::visit_special_val_expr(special_val_expr * s)
{
	if(s->getKind() == E_DURATION_VAR) isDuration = true;
	isFixed = true; // Possibly inappropriate...
};

void SimpleEvaluator::visit_func_term(func_term * s)
{
	extended_func_symbol * efs = EFT(s->getFunction());
	//cout << "Eval: " << s->getFunction()->getName() << "\n";
	if(efs->isStatic())
	{
		isFixed = true;
		pair<bool,double> pv = efs->getInitial(makeIterator(f,s->getArgs()->begin()),
						makeIterator(f,s->getArgs()->end()));
		if(pv.first)
		{
			nvalue = pv.second;
			//cout << "Value is " << nvalue << "\n";
		}
		else
		{
			undefined = true;
			//cout << "Undefined\n";
		};
	}
	else
	{
		isFixed = false;
		//cout << "Variable\n";
	};
};

};
