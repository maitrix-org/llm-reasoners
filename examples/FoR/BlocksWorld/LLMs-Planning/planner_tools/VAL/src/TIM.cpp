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

#include "FastEnvironment.h"
#include "TimSupport.h"
#include <cstdio>
#include <iostream>
#include <fstream>
#include "ptree.h"
#include "FlexLexer.h"
#include "TypedAnalyser.h"
#include "TIM.h"


extern int yyparse();
extern int yydebug;

using std::ifstream;
using std::ofstream;
using std::ostream;
using std::cerr;

namespace VAL {

bool Verbose = false;
ostream * report = &cout;
parse_category* top_thing=NULL;

analysis* current_analysis;

yyFlexLexer* yfl;
TypeChecker * theTC;
extern bool FAverbose;

int PropInfo::x = 0;

};

char * current_filename;
using namespace VAL;

namespace TIM {


TIMAnalyser * TA;



void performTIMAnalysis(char * argv[])
{
    current_analysis = new analysis;
    IDopTabFactory * fac = new IDopTabFactory;
    current_analysis->setFactory(fac);
    current_analysis->pred_tab.replaceFactory<holding_pred_symbol>();
    current_analysis->func_tab.replaceFactory<extended_func_symbol>();
    current_analysis->const_tab.replaceFactory<TIMobjectSymbol>();
    current_analysis->op_tab.replaceFactory<TIMactionSymbol>();
    current_analysis->setFactory(new TIMfactory());
    auto_ptr<EPSBuilder> eps(new specEPSBuilder<TIMpredSymbol>());
    Associater::buildEPS = eps;
    
    ifstream* current_in_stream;
    yydebug=0; // Set to 1 to output yacc trace 

    yfl= new yyFlexLexer;

    // Loop over given args

	for(int i = 0;i < 2;++i)
	{
		current_filename= argv[i];
	//	cout << "File: " << current_filename << '\n';
		current_in_stream = new ifstream(current_filename);
		if (current_in_stream->bad())
		{
		    // Output a message now
		    cerr << "Failed to open ";
		    if (i) {
			cerr << "problem";
		    } else {
			cerr << "domain";
		    }
		    cerr << " file " << current_filename << "\n";
		    exit(0);
		    // Log an error to be reported in summary later
		    line_no= 0;
		    log_error(E_FATAL,"Failed to open file");
		}
		else
		{
		    line_no= 1;

		    // Switch the tokeniser to the current input stream
		    yfl->switch_streams(current_in_stream,&cout);
		    yyparse();

		    // Output syntax tree
		    //if (top_thing) top_thing->display(0);
		}
		delete current_in_stream;
    }
    // Output the errors from all input files
    if(current_analysis->error_list.errors) {
	cerr << "Critical Errors Encountered in Domain/Problem File\n";
	cerr << "--------------------------------------------------\n\n";
        cerr << "Due to critical errors in the supplied domain/problem file, the planner\n";
	cerr << "has to terminate.  The errors encountered are as follows:\n";
    	current_analysis->error_list.report();
	exit(0);
    } else if (current_analysis->error_list.warnings) {
        cout << "Warnings encountered when parsing Domain/Problem File\n";
	cerr << "-----------------------------------------------------\n\n";
        cerr << "The supplied domain/problem file appear to violate part of the PDDL\n";
        cerr << "language specification.  Specifically:\n";
    	current_analysis->error_list.report();
	cerr << "\nThe planner will continue, but you may wish to fix your files accordingly\n";
    }

    delete yfl;

    DurativeActionPredicateBuilder dapb;
    current_analysis->the_domain->visit(&dapb);

	theTC = new TypeChecker(current_analysis);
    	if (!theTC->typecheckDomain()) {
		cerr << "Type Errors Encountered in Domain File\n";
		cerr << "--------------------------------------\n\n";
		cerr << "Due to type errors in the supplied domain file, the planner\n";
		cerr << "has to terminate.  The log of type checking is as follows:\n\n";
		Verbose = true;
		theTC->typecheckDomain();
		exit(0);
	}
	if (!theTC->typecheckProblem()) {
		cerr << "Type Errors Encountered in Problem File\n";
		cerr << "---------------------------------------\n\n";
		cerr << "Due to type errors in the supplied problem file, the planner\n";
		cerr << "has to terminate.  The log of type checking is as follows:\n\n";
		Verbose = true;
		theTC->typecheckProblem();
		exit(0);
	}
    TypePredSubstituter a;
    current_analysis->the_problem->visit(&a);
   	current_analysis->the_domain->visit(&a); 

   	Analyser aa(dapb.getIgnores());
   	current_analysis->the_problem->visit(&aa);
   	current_analysis->the_domain->visit(&aa);

//    current_analysis->the_domain->predicates->visit(&aa);

	if(FAverbose && current_analysis->the_domain->functions)
		current_analysis->the_domain->functions->visit(&aa);
    TA = new TIMAnalyser(*theTC,current_analysis);
    current_analysis->the_domain->visit(TA);
    current_analysis->the_problem->visit(TA);
    for_each(current_analysis->the_domain->ops->begin(),
    			current_analysis->the_domain->ops->end(),completeMutexes);
	TA->checkSV();
    dapb.reverse();
    current_analysis->the_domain->visit(&dapb);
    for(vector<durative_action*>::iterator i = aa.getFixedDAs().begin();
    		i != aa.getFixedDAs().end();++i)
    {
    	(static_cast<TIMactionSymbol*>((*i)->name))->assertFixedDuration();
    };
}

};

