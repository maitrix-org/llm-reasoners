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
#include "instantiation.h"
#include "SimpleEval.h"
#include "DebugWriteController.h"
#include "typecheck.h"
#include "TIM.h"

using std::ifstream;
using std::cerr;

using namespace TIM;
using namespace Inst;
using namespace VAL;

int main(int argc,char * argv[])
{
	performTIMAnalysis(&argv[1]);

	SimpleEvaluator::setInitialState();
    for(operator_list::const_iterator os = current_analysis->the_domain->ops->begin();
    			os != current_analysis->the_domain->ops->end();++os)
    {
    	cout << (*os)->name->getName() << "\n";
    	instantiatedOp::instantiate(*os,current_analysis->the_problem,*theTC);
    	cout << instantiatedOp::howMany() << " so far\n";
    };
    instantiatedOp::createAllLiterals(current_analysis->the_problem,theTC);
    instantiatedOp::filterOps(theTC);
    cout << instantiatedOp::howMany() << "\n";
    instantiatedOp::writeAll(cout);

	cout << "\nList of all literals:\n";
    
    instantiatedOp::writeAllLiterals(cout);

}
