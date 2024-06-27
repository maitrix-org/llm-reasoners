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

#ifndef __SIMPLE_EVALUATOR
#define __SIMPLE_EVALUATOR
#include "VisitController.h"
#include "FastEnvironment.h"
#include <map>
#include <vector>
#include <set>
#include "ptree.h"

namespace VAL {

	class TypeChecker;

};

namespace Inst {

typedef std::set<VAL::pred_symbol *> IState0Arity;
typedef std::map<VAL::pred_symbol *,vector<VAL::parameter_symbol_list*> > IState;



class PrimitiveEvaluator {
protected:
	bool & valueTrue;
	bool & unknownTrue;
	bool & valueFalse;
	bool & unknownFalse;

public:
	PrimitiveEvaluator(bool & xt,bool & yt, bool & xf,bool & yf) : valueTrue(xt), unknownTrue(yt), valueFalse(xf), unknownFalse(yf) {};
	virtual ~PrimitiveEvaluator() {};
	virtual void evaluateSimpleGoal(VAL::FastEnvironment * f,VAL::simple_goal * s)
	{
		unknownTrue = true;
		unknownFalse = true;
	};
};

template<typename PE>
class PrimitiveEvaluatorConstructor {
public:
	PrimitiveEvaluator * operator()(bool & v,bool & u, bool & w, bool & x)
	{
		return new PE(v,u,w,x);
	}
};


class InitialStateEvaluator : public PrimitiveEvaluator {
protected: 
	friend class ParameterDomainConstraints;
	friend class LitStoreEvaluator;
	static IState initState;
	static IState0Arity init0State;
public:
	InitialStateEvaluator(bool & v,bool & u, bool & w, bool & x):
		PrimitiveEvaluator(v,u,w,x)
	{};
	static void setInitialState();
	virtual void evaluateSimpleGoal(VAL::FastEnvironment * f,VAL::simple_goal * s);
};

typedef PrimitiveEvaluatorConstructor<InitialStateEvaluator> ISC;

class SimpleEvaluator : public VAL::VisitController {
protected:
	bool valueTrue;
	bool unknownTrue;
	bool valueFalse;
	bool unknownFalse;

	VAL::TypeChecker * tc;
	VAL::FastEnvironment * f;
	

	VAL::pred_symbol * const equality;
	
	bool isFixed;
	bool undefined;
	double nvalue; // Used for numeric values.
	bool isDuration;

	PrimitiveEvaluator * const primev;
	
public:

	template<typename PEC>
	SimpleEvaluator(VAL::TypeChecker * const tcIn, VAL::FastEnvironment * const ff,PrimitiveEvaluatorConstructor<PEC> pec) : valueTrue(true), unknownTrue(false), valueFalse(false), unknownFalse(false), tc(tcIn), f(ff),
		equality(VAL::current_analysis->pred_tab.symbol_probe("=")),
		primev(pec(valueTrue,unknownTrue,valueFalse,unknownFalse)) 
	{};

	~SimpleEvaluator() {delete primev;};
	
	bool reallyTrue() const
	{
		return !unknownTrue && valueTrue;
	};
	bool reallyFalse() const
	{
		//return !unknownTrue && !valueTrue;
		return !unknownFalse && valueFalse;
	};

	void prepareForVisit(VAL::FastEnvironment * const ff) {
		f = ff;
		valueTrue = true;
		unknownTrue = false;
		valueFalse = false;
		unknownFalse = false;
	}

	static void setInitialState() {InitialStateEvaluator::setInitialState();};
	
	virtual void visit_simple_goal(VAL::simple_goal *);
	virtual void visit_qfied_goal(VAL::qfied_goal *);
	virtual void visit_conj_goal(VAL::conj_goal *);
	virtual void visit_disj_goal(VAL::disj_goal *);
	virtual void visit_timed_goal(VAL::timed_goal *);
	virtual void visit_imply_goal(VAL::imply_goal *);
	virtual void visit_neg_goal(VAL::neg_goal *);
	virtual void visit_comparison(VAL::comparison *);
	virtual void visit_preference(VAL::preference *);
	virtual void visit_event(VAL::event * e);
    virtual void visit_process(VAL::process * p);
	virtual void visit_action(VAL::action * o);
	virtual void visit_derivation_rule(VAL::derivation_rule * o);
	virtual void visit_durative_action(VAL::durative_action * da);
	bool equiv(const VAL::parameter_symbol_list *,const VAL::parameter_symbol_list *);

	virtual void visit_plus_expression(VAL::plus_expression * s);
	virtual void visit_minus_expression(VAL::minus_expression * s);
	virtual void visit_mul_expression(VAL::mul_expression * s);
	virtual void visit_div_expression(VAL::div_expression * s);
	virtual void visit_uminus_expression(VAL::uminus_expression * s);
	virtual void visit_int_expression(VAL::int_expression * s);
	virtual void visit_float_expression(VAL::float_expression * s);
	virtual void visit_special_val_expr(VAL::special_val_expr * s);
	virtual void visit_func_term(VAL::func_term * s);



};


};

#endif
