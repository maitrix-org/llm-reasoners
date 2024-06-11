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

#ifndef __EVALUATOR
#define __EVALUATOR
#include "VisitController.h"
#include <map>
#include <vector>
#include <set>
#include "ptree.h"
#include "Environment.h"

namespace VAL {
class State;
class FastEnvironment;
class Validator;

};

namespace Inst {

class instantiatedOp;

class Evaluator : public VAL::VisitController {
protected:
	VAL::Validator * vld;

	bool value;
	VAL::Environment env;
	VAL::FastEnvironment * f;

	const VAL::State * state;
	
	VAL::pred_symbol * equality;

	bool ignoreMetrics;
	bool context;
	
public:

	static void setInitialState();
	
	Evaluator(VAL::Validator * v,const VAL::State * s,Inst::instantiatedOp * op,bool im = false);

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
	virtual void visit_durative_action(VAL::durative_action * da);
	bool operator()() {return value;};
};

};

#endif
