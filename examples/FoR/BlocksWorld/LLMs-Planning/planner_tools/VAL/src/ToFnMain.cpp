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

#include "ToFunction.h"
#include "FastEnvironment.h"
#include "SASActions.h"
#include "instantiation.h"
#include "SimpleEval.h"

using namespace SAS;

int main(int argc,char * argv[])
{
	performTIMAnalysis(&argv[1]);
	use_sasoutput = true;
	FunctionStructure fs;
	fs.normalise();
	fs.initialise();

	fs.processActions();
	fs.buildLayers();
	
    fs.setUpInitialState();
    int level = 0;
    while(fs.growOneLevel())
    {
    	++level;
    	cout << "Built level: " << level << "\n";
    };
};
