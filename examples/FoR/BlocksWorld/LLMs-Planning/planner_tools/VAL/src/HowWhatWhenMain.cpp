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
#include "ptree.h"
#include "TIM.h"
#include "HowAnalyser.h"
#include "FuncAnalysis.h"
#include "AbstractGraph.h"

using namespace TIM;
using namespace VAL;
//using namespace Inst;



int main(int argc,char * argv[])
{
	FAverbose = false;
	performTIMAnalysis(&argv[1]);

	HowAnalyser ha;

	current_analysis->the_domain->predicates->visit(&ha);
	current_analysis->the_domain->ops->visit(&ha);
	current_analysis->the_problem->initial_state->visit(&ha);
	ha.completeGraph();
};
