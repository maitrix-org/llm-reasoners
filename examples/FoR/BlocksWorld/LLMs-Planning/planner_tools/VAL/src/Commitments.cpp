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

#include <string>
#include "State.h"
#include "Plan.h"
#include "Validator.h"
#include "typecheck.h"
#include "RobustAnalyse.h"

#include <cstdio>
#include <iostream>
#include <fstream>
#include "ptree.h"
#include "FlexLexer.h"
#include "Utils.h"

#include "LaTeXSupport.h"
#include "main.h"
#include "ptree.h"
#include "TIM.h"
#include "FuncAnalysis.h"
#include "AbstractGraph.h"
#include "CommitmentTracker.h"
#include "Agents.h"

using std::ifstream;
using std::ofstream;
using std::cerr;
using std::cout;
using std::for_each;
using std::copy;

extern int yyparse();
extern int yydebug;

using namespace TIM;
using namespace VAL;
//using namespace Inst;

namespace VAL {
  
extern analysis* current_analysis;

extern yyFlexLexer* yfl;
bool Silent;
int errorCount;


//bool Verbose;
bool ContinueAnyway;
bool ErrorReport;
bool InvariantWarnings;
bool LaTeX;

extern ostream * report;

};



typedef map<double,vector<pair<string,vector<double> > > > Ranking;

using namespace VAL;

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

	    	*report << "Bad plan description!\n";
	    	delete the_plan;
	    	the_plan = 0; return the_plan;       
	    };

		if(the_plan->getTime() >= 0) {name += " - Planner run time: "; name += toString(the_plan->getTime());};

    return the_plan;

};

vector<plan_step *> getTimedInitialLiteralActions()
{

  vector<plan_step *> timedIntitialLiteralActions;
  
    if(current_analysis->the_problem->initial_state->timed_effects.size() != 0)
      {
          int count = 1;
           for(pc_list<timed_effect*>::const_iterator e = current_analysis->the_problem->initial_state->timed_effects.begin(); e != current_analysis->the_problem->initial_state->timed_effects.end(); ++e)
           {                   
                  operator_symbol * timed_initial_lit = current_analysis->op_tab.symbol_put("Timed Initial Literal Action "+ toString(count++));

                  action  * timed_initial_lit_action = new action(timed_initial_lit,new var_symbol_list(),new conj_goal(new goal_list()),(*e)->effs,new var_symbol_table());

                  plan_step * a_plan_step =  new plan_step(timed_initial_lit,new const_symbol_list());
                  a_plan_step->start_time_given = true;
                  a_plan_step->start_time = dynamic_cast<const timed_initial_literal *>(*e)->time_stamp;

                  a_plan_step->duration_given = false;

                  timedIntitialLiteralActions.push_back(a_plan_step);
                  current_analysis->the_domain->ops->push_back(timed_initial_lit_action);
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
void executePlans(int & argc,char * argv[],int & argcount,Agents & as,TypeChecker & tc,const DerivationRules * derivRules,double tolerance,bool lengthDefault,bool giveAdvice)
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
       
       PlanRepair pr(timedInitialLiteralActions,deadLine,derivRules,tolerance,tc,current_analysis->the_domain->ops,
	    			current_analysis->the_problem->initial_state,
	    			the_plan,planNoTimedLits,current_analysis->the_problem->metric,lengthDefault,
	    			current_analysis->the_domain->isDurative(),current_analysis->the_problem->the_goal,current_analysis);

		CommitmentTracker commt(State(&(pr.getValidator()),current_analysis->the_problem->initial_state),&(pr.getValidator()),as);
		State::addObserver(&commt);
			
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

		    	if(pr.getValidator().checkGoal(current_analysis->the_problem->the_goal))

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
                     if(!pr.getValidator().checkGoal(current_analysis->the_problem->the_goal)) *report << "\nGoal not satisfied\n";

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
	//commt.write(std::cout);
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


		if(current_analysis->the_problem->metric &&
				current_analysis->the_problem->metric->opt.front() == E_MINIMIZE)
		{
			if(LaTeX)
			{
				*report << "\\begin{tabbing}\n";
				*report << "{\\bf Value} \\qquad \\= {\\bf Plan}\\\\[0.8ex]\n";
			};


			if(!Silent && !LaTeX) for_each(rnk.begin(),rnk.end(),showList());

			if(LaTeX) *report << "\\end{tabbing}\n";

		}
		else
		{
			if(LaTeX)
			{
				*report << "\\begin{tabbing}\n";
				*report << "{\\bf Value} \\qquad \\= {\\bf Plan}\\\\[0.8ex]\n";
			};


			if(!Silent && !LaTeX) for_each(rnk.rbegin(),rnk.rend(),showList());



			if(LaTeX) *report << "\\end{tabbing}\n";
		};



		*report << "\n";
	};

	if(!rnkInv.empty())
	{
		if(LaTeX)
		{
			*report << "\\section{Successful Plans Subject To Further Checks}\n";

		}
		else

			if(!Silent) cout << "\nSuccessful Plans Subject To Further Invariant Checks:";


		if(current_analysis->the_problem->metric &&
				current_analysis->the_problem->metric->opt.front() == E_MINIMIZE)
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



		*report << "\n";
	};

	if(!failed.empty())
	{
		if(LaTeX)
		{
			*report << "\\section{Failed Plans}\n";

		}
		else
			cout << "\n\nFailed plans:\n ";

		if(LaTeX)
			displayFailedLaTeXList(failed);
		else
			copy(failed.begin(),failed.end(),ostream_iterator<string>(cout," "));



		*report << "\n";
	};

	if(!queries.empty())
	{
		if(LaTeX)
		{
			*report << "\\section{Queries (validator failed)}\n";

		}
		else
			cout << "\n\nQueries (validator failed):\n ";

		if(LaTeX)
			displayFailedLaTeXList(queries);
		else
			copy(queries.begin(),queries.end(),ostream_iterator<string>(cout," "));



		*report << "\n";
	};

};



const_symbol * lookUpName(const string & s)
{
	return current_analysis->const_tab.symbol_get(s);
}

operator_ * lookUpAction(const string & s)
{
	operator_symbol * os = current_analysis->op_tab.symbol_probe(s);
	for(operator_list::iterator i = current_analysis->the_domain->ops->begin();
			i != current_analysis->the_domain->ops->end();++i)
	{
		if((*i)->name == os) 
		{
			return (*i);
		}
	}

	return NULL;
}

Agents extractAgents(char * fn)
{
	Agents a;
	std::ifstream agents(fn);

	
	while(!agents.eof())
	{
		int n,m;
		agents >> n;
		string s;
		agents >> s;
		if(!s.empty())
		{
			const_symbol * c = lookUpName(s);
			a.startNewGroup(c);
			for(int i = 0;i < n-1;++i)
			{
				agents >> s;
				c = lookUpName(s);
				a.addAgent(c);
			}
			agents >> n;
			for(int i = 0;i < n;++i)
			{
				agents >> s >> m;
				operator_ * o = lookUpAction(s);
				a.addAction(o,m);
			}
				
		}
	}

	return a;
}


int main(int argc,char * argv[])
{

   bool giveAdvice = true;

	double tolerance = 0.0001;
	bool lengthDefault = true;
	FAverbose = false;
	performTIMAnalysis(&argv[2]);

	Agents as(extractAgents(argv[1]));
 	//SAS::CausalGraph cg;
 	//cout << cg;
 	
	const DerivationRules * derivRules = new DerivationRules (current_analysis->the_domain->drvs,current_analysis->the_domain->ops);
   
	
	int argcount = 4;
	executePlans(argc,argv,argcount,as,*theTC,derivRules,tolerance,lengthDefault,giveAdvice);
   
};
