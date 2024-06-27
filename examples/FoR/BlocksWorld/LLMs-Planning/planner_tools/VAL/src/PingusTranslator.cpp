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
#include <fstream>
#include "ptree.h"
#include "TIM.h"
#include "FuncAnalysis.h"
#include "PinguPlanGenerator.h"
#include "FlexLexer.h"

using namespace std;
using namespace TIM;
using namespace VAL;
//using namespace Inst;

namespace VAL {

extern yyFlexLexer * yfl;
}
extern int yyparse();

plan * getPlan(char * name)
{
  plan * the_plan = 0;

  ifstream planFile(name);
  if(!planFile)
    {
      cout << "Bad plan file!\n";
      return the_plan;
    };


  yfl = new yyFlexLexer(&planFile,&cout);
  yyparse();
  delete yfl;

  the_plan = dynamic_cast<plan*>(top_thing);

  return the_plan;

};


int main(int argc,char * argv[])
{
	FAverbose = false;
	performTIMAnalysis(&argv[1]);

	plan * thePlan = getPlan(argv[3]);
	PinguPlanGen ppg(argv[4]);

	current_analysis->the_problem->initial_state->visit(&ppg);
	cout << "(pingus-plan\n(actions\n";
	thePlan->visit(&ppg);
	cout << "))\n";
};
