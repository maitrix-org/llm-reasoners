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

/*-----------------------------------------------------------------------------
  VAL - The Automatic Plan Validator for PDDL+

  $Date: 2009-02-05 10:50:26 $
  $Revision: 1.2 $

  Maria Fox, Richard Howey and Derek Long - PDDL+ and VAL
  Stephen Cresswell - PDDL Parser

  maria.fox@cis.strath.ac.uk
  derek.long@cis.strath.ac.uk
  stephen.cresswell@cis.strath.ac.uk
  richard.howey@cis.strath.ac.uk

  By releasing this code we imply no warranty as to its reliability
  and its use is entirely at your own risk.

  Strathclyde Planning Group
  http://planning.cis.strath.ac.uk
 ----------------------------------------------------------------------------*/
#include <string>
#include "State.h"
#include "Plan.h"
#include "Validator.h"
#include "typecheck.h"
#include "RobustAnalyse.h"

#include <cstdio>
#include <iostream>
#include <fstream>
#include <time.h>
#include "ptree.h"
#include "FlexLexer.h"
#include "Utils.h"

#include "LaTeXSupport.h"
#include "main.h"

using std::ifstream;
using std::ofstream;
using std::cerr;
using std::cout;
using std::for_each;
using std::copy;

//#define vector std::vector

extern int yyparse();
extern int yydebug;

namespace VAL {

parse_category* top_thing = NULL;

analysis an_analysis;
analysis* current_analysis;

yyFlexLexer* yfl;
int Silent;
int errorCount;
bool Verbose;
bool ContinueAnyway;
bool ErrorReport;
bool InvariantWarnings;
bool LaTeX;

ostream * report = &cout;


};

char * current_filename;

typedef map<double,vector<pair<string,vector<double> > > > Ranking;

using namespace VAL;

void usage()
{
	cout << "VAL: The PDDL+ plan validation tool\n"
             << "Version 4: Validates continuous effects, events and processes.\n"
             << "\nAuthors: Derek Long, Richard Howey, Stephen Cresswell and Maria Fox\n"
             << "https:://github/KCL-Planning/VAL\n\n"
             << "Usage: validate [options] domainFile problemFile planFile1 ...\n"
		     << "Options:\n    -t <n>     -- Set tolerance to (float) value of n.\n"
		     << "    -r <n> <p> <m> -- Analyse the plan for its robustness, each action timestamp to within a (float) value of n, each PNE to within a (float value) of p, for m test plans.\n"
		     << "    -ra <p>    -- Calculate robustness of plan with respect to varying action timestamps, whilst varying PNEs to within a (float value) of p (default p = 0).\n"
		     << "    -rp <n>    -- Calculate robustness of plan with respect to varying PNEs, whilst varying action timestamps to within a (float value) of n (default n = 0).\n"
		     << "    -rm <x>    -- Set metric for robustness testing: x = m, maximum; x = a, accumulative; x = d, delay. (default x = m).\n"
		     << "    -rd <x>    -- Set distribution for robustness testing: x = u, uniform; x = n, normal; x = p, psuedo-normal. (default x = u).\n"
		     << "    -j         -- When varying the values of PNEs also vary for event preconditions. (default = false)\n"
		     << "    -v         -- Verbose reporting of plan check progress.\n"
		     << "    -l         -- Verbose LaTeX reporting of plan check progress.\n"
		     << "    -a         -- Do not output plan repair advice when Verbose is on.\n"
		     << "    -g         -- Use graphplan length where no metric specified.\n"
		     << "    -h         -- Print this message.\n"
		     << "    -p <n> <m> -- Number of pages for LaTeX Gantt chart (n = across time axis, m = across rows).\n"
		     << "    -o  ... -o -- Objects (and/or types of) to be tracked on LaTeX Gantt chart.\n"
		     << "    -q <n>     -- Number of points(10-878) used to draw LaTeX graphs of PNEs (default = 500).\n"
           << "    -d         -- Do not check set of derived predicates for stratification.\n"
		     << "    -c         -- Continue executing plan even if an action precondition is unsatisfied.\n"
 		     << "    -e         -- Produce error report for the full plan, and try to repair it.\n"
		     << "    -i         -- Warn if invariants with continuous effects cannot be checked.\n"
	     <<         "    -s         -- Silent mode: output is generated only when errors occur\n"
	     << "    -S         -- Silent mode with values: outputs only plan values in order (failed for bad plan)\n"
			 << "    -m         -- Use makespan as metric for temporal plans (overrides any other metric).\n"
	     << "    -L         -- Add step length as metric (in addition to any other metric).\n"
			 << "    -f <file>  -- LaTeX report will be stored in file 'file.tex'\n"
		     << "Multiple plan file arguments can be appended for checking.\n\n";

};


plan * getPlan(int & argc,char * argv[],int & argcount,TypeChecker & tc,vector<string> & failed,string & name)
{
     plan * the_plan;

		if(LaTeX)
		{
			latex.LaTeXPlanReportPrepare(argv[argcount]);
		}
		else
			if(!Silent) cout << "Checking plan: " << argv[argcount] << "\n";

	    ifstream planFile(argv[argcount++]);
	    if(!planFile)
	    {
	    	failed.push_back(name);
	    	*report << "Bad plan file!\n";
	    	the_plan = 0; return the_plan;
	    };

	    yfl = new yyFlexLexer(&planFile,&cout);
	    yyparse();
	    delete yfl;

		the_plan = dynamic_cast<plan*>(top_thing);

	    if(!the_plan || !tc.typecheckPlan(the_plan))
	    {
	    	failed.push_back(name);

	    	if(Silent < 2) *report << "Bad plan description!\n";
	    	if(Silent > 1) *report << "failed\n";
	    	delete the_plan;
	    	the_plan = 0; return the_plan;
	    };

		if(the_plan->getTime() >= 0) {name += " - Planner run time: "; name += toString(the_plan->getTime());};

    return the_plan;

};

vector<plan_step *> getTimedInitialLiteralActions()
{

  vector<plan_step *> timedIntitialLiteralActions;

    if(an_analysis.the_problem->initial_state->timed_effects.size() != 0)
      {
          int count = 1;
           for(pc_list<timed_effect*>::const_iterator e = an_analysis.the_problem->initial_state->timed_effects.begin(); e != an_analysis.the_problem->initial_state->timed_effects.end(); ++e)
           {
                  operator_symbol * timed_initial_lit = an_analysis.op_tab.symbol_put("Timed Initial Literal Action "+ toString(count++));

                  action  * timed_initial_lit_action = new action(timed_initial_lit,new var_symbol_list(),new conj_goal(new goal_list()),(*e)->effs,new var_symbol_table());

                  plan_step * a_plan_step =  new plan_step(timed_initial_lit,new const_symbol_list());
                  a_plan_step->start_time_given = true;
                  a_plan_step->start_time = dynamic_cast<const timed_initial_literal *>(*e)->time_stamp;

                  a_plan_step->duration_given = false;

                  timedIntitialLiteralActions.push_back(a_plan_step);
                  an_analysis.the_domain->ops->push_back(timed_initial_lit_action);
           };
      };

  return timedIntitialLiteralActions;
};

void deleteTimedIntitialLiteralActions(vector<plan_step *> tila)
{
  for(vector<plan_step *>::iterator i = tila.begin(); i != tila.end(); ++i)
  {
    delete *i;
  };
};

//execute all the plans in the usual manner without robustness checking
void executePlans(int & argc,char * argv[],int & argcount,TypeChecker & tc,const DerivationRules * derivRules,double tolerance,bool lengthDefault,bool giveAdvice)
{
  Ranking rnk;
  Ranking rnkInv;
  vector<string> failed;
  vector<string> queries;

	while(argcount < argc)
	{
      string name(argv[argcount]);

      plan * the_plan = getPlan(argc,argv,argcount,tc,failed,name);
      if(the_plan == 0) continue;

      plan * copythe_plan = new plan(*the_plan);
      plan * planNoTimedLits = new plan();
      vector<plan_step *> timedInitialLiteralActions = getTimedInitialLiteralActions();
      double deadLine = 101;

        //add timed initial literals to the plan from the problem spec
       for(vector<plan_step *>::iterator ps = timedInitialLiteralActions.begin(); ps != timedInitialLiteralActions.end(); ++ps)
       {
          the_plan->push_back(*ps);
       };

       //add actions that are not to be moved to the timed intitial literals otherwise to the plan to be repaired
       //i.e. pretend these actions are timed initial literals
       for(pc_list<plan_step*>::const_iterator i = copythe_plan->begin(); i != copythe_plan->end(); ++i)
       {
              planNoTimedLits->push_back(*i);
       };

       copythe_plan->clear(); delete copythe_plan;

       PlanRepair pr(timedInitialLiteralActions,deadLine,derivRules,tolerance,tc,an_analysis.the_domain->ops,
	    			an_analysis.the_problem->initial_state,
	    			the_plan,planNoTimedLits,an_analysis.the_problem->metric,lengthDefault,
	    			an_analysis.the_domain->isDurative(),an_analysis.the_problem->the_goal,current_analysis);

		if(LaTeX)
		{
			latex.LaTeXPlanReport(&(pr.getValidator()),the_plan);
		}
		else if(Verbose)
			pr.getValidator().displayPlan();



	    bool showGraphs = false;


	    try {

		    if(pr.getValidator().execute())
		    {
		    	if(LaTeX)
		    		*report << "Plan executed successfully - checking goal\\\\\n";
		    	else
		    		if(!Silent) cout << "Plan executed successfully - checking goal\n";

		    	if(pr.getValidator().checkGoal(an_analysis.the_problem->the_goal))

		    	{
		    		if(!(pr.getValidator().hasInvariantWarnings()))
		    		{
		    			vector<double> vs(pr.getValidator().finalValue());
		    			rnk[vs[0]].push_back(make_pair(name,vs));
		    			if(!Silent && !LaTeX) *report << "Plan valid\n";
		    			if(LaTeX) *report << "\\\\\n";
		    			if(!Silent && !LaTeX) *report << "Final value: ";
		    			if(Silent > 1 || (!Silent && !LaTeX))
		    			{
		    				vector<double> vs(pr.getValidator().finalValue());
		    				for(unsigned int i = 0;i < vs.size();++i)
		    					*report << vs[i] << " ";
		    				*report << "\n";
		    			}
		    		}
		    		else
		    		{
		    			vector<double> vs(pr.getValidator().finalValue());
						rnkInv[vs[0]].push_back(make_pair(name,vs));
		    			if(!Silent && !LaTeX) *report << "Plan valid (subject to further invariant checks)\n";
		    			if(LaTeX) *report << "\\\\\n";
		    			if(!Silent && !LaTeX)
		    			{
		    				*report << "Final value: ";
		    				vector<double> vs(pr.getValidator().finalValue());
		    				for(unsigned int i = 0;i < vs.size();++i)
		    					*report << vs[i] << " ";
		    				*report << "\n";
		    			};
						if(Silent > 1)
						{
							*report << "failed\n";
						}
		          };
		          	if(Verbose)
		          	{
		          		pr.getValidator().reportViolations();
		          	};
		    	}
		    	else
		    	{
		    		failed.push_back(name);
		    		if(Silent < 2) *report << "Goal not satisfied\n";
		    		if(Silent > 1) *report << "failed\n";

		    		if(LaTeX) *report << "\\\\\n";
		    		if(Silent < 2) *report << "Plan invalid\n";
				++errorCount;
			};

		    }
		    else
		    {
		    	failed.push_back(name);
			++errorCount;
         		    	if(ContinueAnyway)
                  {
                     if(LaTeX) *report << "\nPlan failed to execute - checking goal\\\\\n";
                     else
                     {
                     	if(Silent < 2) *report << "\nPlan failed to execute - checking goal\n";
						if(Silent > 1) *report << "failed\n";
					}
                     if(!pr.getValidator().checkGoal(an_analysis.the_problem->the_goal)) *report << "\nGoal not satisfied\n";

         		    }

                 else {
                 	if(Silent < 2) *report << "\nPlan failed to execute\n";
					if(Silent > 1) *report << "failed\n";
				}

        };

              if(pr.getValidator().hasInvariantWarnings())
              {
						if(LaTeX)
							*report << "\\\\\n\\\\\n";
						else
							if(Silent < 2) *report << "\n\n";


		    			*report << "This plan has the following further condition(s) to check:";

						if(LaTeX)
							*report << "\\\\\n\\\\\n";
						else
							if(Silent < 2) *report << "\n\n";

						pr.getValidator().displayInvariantWarnings();
		    		};

		    if(pr.getValidator().graphsToShow()) showGraphs = true;
		}
		catch(exception & e)
		{
			if(LaTeX)
			{
				*report << "\\error \\\\\n";
				*report << "\\end{tabbing}\n";
				*report << "Error occurred in validation attempt:\\\\\n  " << e.what() << "\n";
			}
			else
				if(Silent < 2) *report << "Error occurred in validation attempt:\n  " << e.what() << "\n";

			queries.push_back(name);

		};

    //display error report and plan repair advice
      if(giveAdvice && (Verbose || ErrorReport))
     {
            pr.firstPlanAdvice();
      };

      //display LaTeX graphs of PNEs
    		if(LaTeX && showGraphs)
		{
			latex.LaTeXGraphs(&(pr.getValidator()));
		};

    //display gantt chart of plan
		if(LaTeX)
		{
			latex.LaTeXGantt(&(pr.getValidator()));

		};

    planNoTimedLits->clear(); delete planNoTimedLits;
    delete the_plan;
	};

	if(!rnk.empty())
	{
		if(LaTeX)
		{
			*report << "\\section{Successful Plans}\n";


		}
		else
			if(!Silent) cout << "\nSuccessful plans:";


		if(an_analysis.the_problem->metric &&
				an_analysis.the_problem->metric->opt.front() == E_MINIMIZE)
		{
			if(LaTeX)
			{
				*report << "\\begin{tabbing}\n";
				*report << "{\\bf Value} \\qquad \\= {\\bf Plan}\\\\[0.8ex]\n";
			};


			if(!Silent) for_each(rnk.begin(),rnk.end(),showList());

			if(LaTeX) *report << "\\end{tabbing}\n";

		}
		else
		{
			if(LaTeX)
			{
				*report << "\\begin{tabbing}\n";
				*report << "{\\bf Value} \\qquad \\= {\\bf Plan}\\\\[0.8ex]\n";
			};


			if(!Silent) for_each(rnk.rbegin(),rnk.rend(),showList());



			if(LaTeX) *report << "\\end{tabbing}\n";
		};



		if(!Silent) *report << "\n";
	};

	if(!rnkInv.empty())
	{
		if(LaTeX)
		{
			*report << "\\section{Successful Plans Subject To Further Checks}\n";

		}
		else

			if(!Silent) cout << "\nSuccessful Plans Subject To Further Invariant Checks:";


		if(an_analysis.the_problem->metric &&
				an_analysis.the_problem->metric->opt.front() == E_MINIMIZE)
		{
			if(LaTeX)
			{
				*report << "\\begin{tabbing}\n";
				*report << "{\\bf Value} \\qquad \\= {\\bf Plan}\\\\[0.8ex]\n";
			};

			for_each(rnkInv.begin(),rnkInv.end(),showList());

			if(LaTeX) *report << "\\end{tabbing}\n";
		}
		else
		{
			if(LaTeX)
			{
				*report << "\\begin{tabbing}\n";
				*report << "{\\bf Value} \\qquad \\= {\\bf Plan}\\\\[0.8ex]\n";
			};

			for_each(rnkInv.rbegin(),rnkInv.rend(),showList());

			if(LaTeX) *report << "\\end{tabbing}\n";
		};



		if(!Silent) *report << "\n";
	};

	if(!failed.empty())
	{
		if(LaTeX)
		{
			*report << "\\section{Failed Plans}\n";

		}
		else
			if(Silent < 2) *report << "\n\nFailed plans:\n ";

		if(LaTeX)
			displayFailedLaTeXList(failed);
		else
			if(Silent < 2) copy(failed.begin(),failed.end(),ostream_iterator<string>(*report," "));



		if(Silent < 2) *report << "\n";
	};

	if(!queries.empty())
	{
		if(LaTeX)
		{
			*report << "\\section{Queries (validator failed)}\n";

		}
		else
			if(Silent < 2) *report << "\n\nQueries (validator failed):\n ";

		if(LaTeX)
			displayFailedLaTeXList(queries);
		else
			if(Silent < 2) copy(queries.begin(),queries.end(),ostream_iterator<string>(*report," "));



		if(Silent < 2) *report << "\n";
	};

};

void analysePlansForRobustness(int & argc,char * argv[],int & argcount,TypeChecker & tc,const DerivationRules * derivRules,
          double tolerance,bool lengthDefault,bool giveAdvice,double robustMeasure,int noTestPlans,bool car,bool cpr,RobustMetric robm,RobustDist robd)
{
  vector<string> failed;
  srand(time(0)); // Initialize random number generator.
  vector<plan_step *> timedIntitialLiteralActions = getTimedInitialLiteralActions();


	while(argcount < argc)
	{
      string name(argv[argcount]);
      plan * the_plan = getPlan(argc,argv,argcount,tc,failed,name);
      if(the_plan == 0) continue;



      RobustPlanAnalyser rpa(robustMeasure,noTestPlans,derivRules,tolerance,tc,an_analysis.the_domain->ops,
	    			an_analysis.the_problem->initial_state,
	    			the_plan,an_analysis.the_problem->metric,lengthDefault,
	    			an_analysis.the_domain->isDurative(),an_analysis.the_problem->the_goal,current_analysis,timedIntitialLiteralActions,car,cpr,robm,robd);

      rpa.analyseRobustness();

      delete the_plan;

  };

  deleteTimedIntitialLiteralActions(timedIntitialLiteralActions);

};

//main
int main(int argc,char * argv[])
{
	report->precision(10);
  try {
	if(argc < 2)
	{
		usage();
		return 0;
	};

  current_analysis= &an_analysis;
  //an_analysis.const_tab.symbol_put(""); //for events - undefined symbol
  Silent = 0;
  errorCount = 0;
	Verbose = false;
	ContinueAnyway = false;
	ErrorReport = false;
  Robust = false;
  JudderPNEs = false;
  EventPNEJuddering = false;
  TestingPNERobustness = false;
  RobustPNEJudder = 0;

	InvariantWarnings = false;
	LaTeX = false;
	ofstream possibleLatexReport;
	makespanDefault = false;
	stepLengthDefault = false;
   bool CheckDPs = true;
   bool giveAdvice = true;

	double tolerance = 0.01;
	bool lengthDefault = true;
	double robustMeasure = 0;
	int noTestPlans = 1000;
  bool calculateActionRobustness = false;
  bool calculatePNERobustness = false;
  RobustMetric robustMetric = MAX;
  RobustDist robustDist = UNIFORM;

	string s;
	bool ganttObjectsGot = false;

    int argcount = 1;
    while(argcount < argc && argv[argcount][0] == '-')
    {
		switch(argv[argcount][1])
		{
    		case 'v':

	    		Verbose = true;
	    		++argcount;
	    		break;

		case 'r':

			Robust = true; ++argcount;


          if(argv[argcount-1][2] == 'a')
          {
              calculateActionRobustness = true;
              if(argv[argcount][0] >= '0' && argv[argcount][0] <= '9')
              {
                RobustPNEJudder = atof(argv[argcount++]);
              }
              else RobustPNEJudder = 0;

          }
          else if(argv[argcount-1][2] == 'p')
          {

              calculatePNERobustness = true;
              if(argv[argcount][0] >= '0' && argv[argcount][0] <= '9')
              {
                robustMeasure = atof(argv[argcount++]);
              }
              else robustMeasure = 0;


          }
          else if(argv[argcount-1][2] == 'm')
          {

      	 			if(argv[argcount][0] == 'd') robustMetric = DELAY;
              else if(argv[argcount][0] == 'a') robustMetric = ACCUM;
              else if(argv[argcount][0] == 'm') robustMetric = MAX;

              ++argcount;

          }
          else if(argv[argcount-1][2] == 'd')
          {

      	 			if(argv[argcount][0] == 'u') robustDist = UNIFORM;
              else if(argv[argcount][0] == 'n') robustDist = NORMAL;
              else if(argv[argcount][0] == 'p') robustDist = PNORM;

              ++argcount;

          }
          else
          {

               if(argv[argcount][0] >= '0' && argv[argcount][0] <= '9')
               {
                 robustMeasure = atof(argv[argcount++]);
               }
               else calculateActionRobustness = true;

               if(argv[argcount][0] >= '0' && argv[argcount][0] <= '9')
               {
                 RobustPNEJudder = atof(argv[argcount++]);
               }
               else calculatePNERobustness = true;

               if(argv[argcount][0] >= '0' && argv[argcount][0] <= '9')
               {
                 noTestPlans = atoi(argv[argcount++]);
                 if(noTestPlans == 0) {noTestPlans = 1;};
               }
               else noTestPlans = 1000;
           };

	    		break;
		case 's':

		  Silent = 1;
		  ++argcount;
                  break;
        case 'S':
        	Silent = 2;
        	++argcount;
        	break;

	    	case 'j':


	 			EventPNEJuddering = true;
	 			++argcount;
	 			break;

    		case 't':

	    		tolerance = atof(argv[++argcount]);
	    		++argcount;
	    		break;

	    	case 'g':

	 			lengthDefault = false;
	 			++argcount;
	 			break;

	    	case 'h':

	    		usage();
	    	    exit(0);

			case 'c':


	    		ContinueAnyway = true;
	    		++argcount;
	    		break;



			case 'e':

	    		ErrorReport = true;
	    		ContinueAnyway = true;
	    		++argcount;
	    		break;

        case 'd':

	    		CheckDPs = false;
	    		++argcount;
	    		break;


	    	case 'i':

	    		InvariantWarnings = true;
	    		++argcount;
	    		break;

	    	case 'l':

	    		LaTeX = true;
	    		Verbose = true;
	    		++argcount;
	    		break;

	    	case 'p':

	    		latex.setnoGCPages(atoi(argv[++argcount]));

	    		latex.setnoGCPageRows(atoi(argv[++argcount]));
	    		++argcount;
	    		break;
		case 'q':
				latex.setnoPoints(atoi(argv[++argcount]));
	    		++argcount;
	    		break;
	    	case 'o':

				++argcount;
				if(ganttObjectsGot) break;

	    		while( !( (argv[argcount][0] == '-') && (argv[argcount][1] == 'o')) )
	    		{
	    			latex.addGanttObject(argv[argcount++]);
	    		};

	    		ganttObjectsGot = true;
	    		++argcount;
	    		break;
	    	case 'm':
	    		makespanDefault = true;

 	    		++argcount;
	    		break;
		case 'L':
		  stepLengthDefault = true;
		  ++argcount;
		  break;
	    	case 'a':
	    		giveAdvice = false;
 	    		++argcount;
	    		break;
	    	case 'f':
	    		{
	    			LaTeX = true;
	    			Verbose = true;
		    		++argcount;
		    		string s(argv[argcount]);
		    		s += ".tex";
		    		possibleLatexReport.open(s.c_str());
		    		report = &possibleLatexReport;
		    		++argcount;
	    		};
	    		break;
	    	default:
	    		cout << "Unrecognised command line switch: " << argv[argcount] << "\n";
	    		exit(-1);
		};
    };


	if(argcount>=argc)
	{
		usage();
		return 0;
	};


	if(LaTeX)
	{
		//LaTeX header
	   latex.LaTeXHeader();
	}

    ifstream domainFile(argv[argcount++]);
    if(!domainFile)
    {
    	cerr << "Bad domain file!\n";
    	if(LaTeX) *report << "\\section{Error!} Bad domain file! \n \\end{document}\n";
    	exit(-1);
    };

    yfl= new yyFlexLexer(&domainFile,&cout);

    yydebug=0;
    yyparse();
    delete yfl;

    if(!an_analysis.the_domain)
    {
    	cerr << "Problem in domain definition!\n";
    	if(LaTeX) *report << "\\section{Error!} Problem in domain definition! \n \\end{document}\n";
    	exit(-1);
    };

    TypeChecker tc(current_analysis);

	if(LaTeX) Verbose = false;
    bool typesOK = tc.typecheckDomain();

    if(LaTeX) Verbose = true;

    if(!typesOK)
    {
    	cerr << "Type problem in domain description!\n";

    	if(LaTeX)
    	{
    		*report << "\\section{Error!} Type problem in domain description! \n \\begin{verbatim}";
    		tc.typecheckDomain();
    		*report << "\\end{verbatim} \\end{document}\n";
    	};


    	exit(-1);
    };

	if(argcount>=argc)
	{
		return 0;
	};

    ifstream problemFile(argv[argcount++]);
    if(!problemFile)
    {
    	cerr << "Bad problem file!\n";
    	if(LaTeX) *report << "\\section{Error!} Bad problem file! \n \\end{document}\n";
    	exit(-1);
    };

    yfl = new yyFlexLexer(&problemFile,&cout);
    yyparse();
    delete yfl;

	if(!tc.typecheckProblem())
	{
		cerr << "Type problem in problem specification!\n";
		if(LaTeX) *report << "\\section{Error!} Type problem in problem specification!\n \\end{document}\n";
		exit(-1);
	};


	if(LaTeX)
	{
      latex.LaTeXDomainAndProblem();
	};



	const DerivationRules * derivRules = new DerivationRules (an_analysis.the_domain->drvs,an_analysis.the_domain->ops);

	if(CheckDPs && !derivRules->checkDerivedPredicates())
	{
		if(LaTeX) latex.LaTeXEnd();
		exit(-1);

	};

  if(Robust)
     analysePlansForRobustness(argc,argv,argcount,tc,derivRules,tolerance,lengthDefault,giveAdvice,robustMeasure,noTestPlans,calculateActionRobustness,calculatePNERobustness,robustMetric,robustDist);
  else
     executePlans(argc,argv,argcount,tc,derivRules,tolerance,lengthDefault,giveAdvice);


	delete derivRules;

	//LaTeX footer
	if(LaTeX) latex.LaTeXEnd();

  }
  catch(exception & e)
  {
  	cerr << "Error: " << e.what() << "\n";
  	an_analysis.error_list.report();
  	return -1;
  };

    return errorCount;
};


