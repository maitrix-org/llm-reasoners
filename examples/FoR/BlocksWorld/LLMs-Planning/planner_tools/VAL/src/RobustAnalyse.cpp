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

  $Date: 2009-02-05 10:50:22 $
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

 #include "RobustAnalyse.h"
 #include "Utils.h"
 #include "LaTeXSupport.h"
 #include "tDistribution.h"
 #include "random.h"
 #include "Validator.h"

namespace VAL {

bool Robust;
double RobustPNEJudder;
bool EventPNEJuddering;
bool JudderPNEs;
bool TestingPNERobustness;
bool LaTeXRecord = false;

RobustPlanAnalyser::~RobustPlanAnalyser()
{

  
};

double RobustPlanAnalyser::getRandomNumber()
{          
   if(robustDist == UNIFORM) return getRandomNumberUni();
   else if(robustDist == NORMAL) return getRandomNumberNorm();
   else if(robustDist == PNORM) return getRandomNumberPsuedoNorm();
   
   return 0;
};

string RobustPlanAnalyser::getMetricName()
{
  if(robustMetric == MAX) return "Max";
  else if(robustMetric == ACCUM) return "Accumulative";
  else if(robustMetric == DELAY) return "Delay";

  return "?";
};

string RobustPlanAnalyser::getDistName()
{
   if(robustDist == UNIFORM) return "Uniform";
   else if(robustDist == NORMAL) return "Normal";
   else if(robustDist == PNORM) return "Psuedo-Normal";


   return "?";
};

double RobustPlanAnalyser::getRandomNumberUni()
{
   return getRandomNumberUniform();
};

double RobustPlanAnalyser::getRandomNumberNorm()
{   
   return (getRandomNumberNormal() + 1.0)/2.0;
};

double RobustPlanAnalyser::getRandomNumberPsuedoNorm()
{
  return getRandomNumberPsuedoNormal();      
};

map<const plan_step *,const plan_step *> RobustPlanAnalyser::varyPlanTimestamps(plan * aplan,const plan * p,double & variation)
{
    map<const plan_step *,const plan_step *> planStepMap;
    if(robustMetric == MAX) planStepMap = varyPlanTimestampsMax(aplan,p,variation);
    else if(robustMetric == ACCUM) planStepMap = varyPlanTimestampsAccum(aplan,p,variation);
    else if(robustMetric == DELAY) planStepMap = varyPlanTimestampsDelay(aplan,p,variation);

   return planStepMap;
};


//also make a map to the original plan step, vary timestamps with the max metric
map<const plan_step *,const plan_step *> RobustPlanAnalyser::varyPlanTimestampsMax(plan * aplan,const plan * p,double & variation)
{
  map<const plan_step *,const plan_step *> planStepMap;
  pc_list<plan_step *>::const_iterator origPlanStep = p->begin();

  for(pc_list<plan_step *>::iterator ps = aplan->begin(); ps != aplan->end(); ++ps, ++origPlanStep)
  {

      (*ps)->start_time = (*ps)->start_time + (1 - 2*getRandomNumber())*variation;
      if((*ps)->start_time < 0) (*ps)->start_time = 0;

      if((*ps)->duration_given)
      {
          (*ps)->originalDuration = (*ps)->duration;

         // (*ps)->duration = (*ps)->duration + (1 - 2*getRandomNumber())*variation;
         // if((*ps)->duration < 0) (*ps)->duration = 0;
      };
          //cout << (*ps)->start_time << " ";
      planStepMap[*ps] = *origPlanStep;
  };
     //cout << "\n";
  return planStepMap;
};

//vary timestamps with the accumulative metric
map<const plan_step *,const plan_step *> RobustPlanAnalyser::varyPlanTimestampsAccum(plan * aplan,const plan * p,double & variation)
{
  double accumTime = 0;
  double randomNumber;
  map<const plan_step *,const plan_step *> planStepMap;
  pc_list<plan_step *>::const_iterator origPlanStep = p->begin();

  for(pc_list<plan_step *>::iterator ps = aplan->begin(); ps != aplan->end(); ++ps, ++origPlanStep)
  {
      randomNumber = (1 - 2*getRandomNumber())*variation;
      accumTime += randomNumber;
      
      
      if(((*ps)->start_time + accumTime) > 0)
      {
        (*ps)->start_time = (*ps)->start_time + accumTime;
      }
      else
      {
        accumTime =  -(*ps)->start_time;
       (*ps)->start_time = 0;
      };

      if((*ps)->duration_given)
      {
          (*ps)->originalDuration = (*ps)->duration;
         
         // (*ps)->duration = (*ps)->duration;
         // if((*ps)->duration < 0) (*ps)->duration = 0;
      };
         // cout << (*ps)->start_time << " ";
      planStepMap[*ps] = *origPlanStep;
  };
     // cout << "\n";
  return planStepMap;
};

//vary timestamps with the delay metric
map<const plan_step *,const plan_step *> RobustPlanAnalyser::varyPlanTimestampsDelay(plan * aplan,const plan * p,double & variation)
{
  double accumTime = 0;
  double randomNumber;
  map<const plan_step *,const plan_step *> planStepMap;
  pc_list<plan_step *>::const_iterator origPlanStep = p->begin();

  for(pc_list<plan_step *>::iterator ps = aplan->begin(); ps != aplan->end(); ++ps, ++origPlanStep)
  {
      do{
        randomNumber = (1 - 2*getRandomNumber())*variation;
      }while(randomNumber < 0);
      
      accumTime += randomNumber;

      (*ps)->start_time = (*ps)->start_time + accumTime;
      //if((*ps)->start_time < 0) (*ps)->start_time = 0;

      if((*ps)->duration_given)
      {
          (*ps)->originalDuration = (*ps)->duration;

         // (*ps)->duration = (*ps)->duration;
        //  if((*ps)->duration < 0) (*ps)->duration = 0;
      };
          //cout << (*ps)->start_time << " ";
      planStepMap[*ps] = *origPlanStep;
  };
      //cout << "\n";
  return planStepMap;
};

map<const plan_step *,const plan_step *> RobustPlanAnalyser::varyPlanTimestampsBoundary(plan * aplan,const plan * p,double & variation,int runNo)
{
    map<const plan_step *,const plan_step *> planStepMap;
    if(robustMetric == MAX) planStepMap = varyPlanTimestampsBoundaryMax(aplan,p,variation,runNo);
    else if(robustMetric == ACCUM) planStepMap = varyPlanTimestampsBoundaryAccum(aplan,p,variation,runNo);
    else if(robustMetric == DELAY) planStepMap = varyPlanTimestampsBoundaryDelay(aplan,p,variation,runNo);

   return planStepMap;
};

//also make a map to the original plan step, vary timestamps with the max metric on boundary
map<const plan_step *,const plan_step *> RobustPlanAnalyser::varyPlanTimestampsBoundaryMax(plan * aplan,const plan * p,double & variation,int runNo)
{
  map<const plan_step *,const plan_step *> planStepMap;
  pc_list<plan_step *>::const_iterator origPlanStep = p->begin();
  
  int randomNo = 1;
  if(runNo == 2) randomNo = -1;
  
  for(pc_list<plan_step *>::iterator ps = aplan->begin(); ps != aplan->end(); ++ps, ++origPlanStep)
  {
       
       if(runNo < 3)
       {
         if(randomNo == 1) randomNo = -1; else randomNo = 1;
       }
       else
       {
         randomNo = rand()%2;
         if(randomNo == 0) randomNo = -1;         
       };
         
       
       
      (*ps)->start_time = (*ps)->start_time + randomNo*variation;
      if((*ps)->start_time < 0) (*ps)->start_time = 0;

      if((*ps)->duration_given)
      {
          (*ps)->originalDuration = (*ps)->duration;

         // (*ps)->duration = (*ps)->duration + (1 - 2*getRandomNumber())*variation;
         // if((*ps)->duration < 0) (*ps)->duration = 0;
      };
          //cout << (*ps)->start_time << " ";
      planStepMap[*ps] = *origPlanStep;
  };
     // cout << "\n";
  return planStepMap;
};

//vary timestamps with the accumulative metric on boundary
map<const plan_step *,const plan_step *> RobustPlanAnalyser::varyPlanTimestampsBoundaryAccum(plan * aplan,const plan * p,double & variation,int runNo)
{
  double accumTime = 0;
  map<const plan_step *,const plan_step *> planStepMap;
  pc_list<plan_step *>::const_iterator origPlanStep = p->begin();

  for(pc_list<plan_step *>::iterator ps = aplan->begin(); ps != aplan->end(); ++ps, ++origPlanStep)
  {
      int randomNo = rand()%2;
      if(randomNo == 0) randomNo = -1;
      
      if(runNo == 1) randomNo = 1;
      else if(runNo == 2) randomNo = -1;
      
      if(randomNo == -1) accumTime += variation;
      else accumTime -= variation;

      if(((*ps)->start_time + accumTime) > 0)
      {
        (*ps)->start_time = (*ps)->start_time + accumTime;
      }
      else
      {
        accumTime =  -(*ps)->start_time;
       (*ps)->start_time = 0;
      };

      if((*ps)->duration_given)
      {
          (*ps)->originalDuration = (*ps)->duration;

         // (*ps)->duration = (*ps)->duration;
         // if((*ps)->duration < 0) (*ps)->duration = 0;
      };
          //cout << (*ps)->start_time << " ";
      planStepMap[*ps] = *origPlanStep;
  };
     // cout << "\n";
  return planStepMap;
};

//vary timestamps with the delay metric on boundary
map<const plan_step *,const plan_step *> RobustPlanAnalyser::varyPlanTimestampsBoundaryDelay(plan * aplan,const plan * p,double & variation,int runNo)
{
  double accumTime = 0;
  map<const plan_step *,const plan_step *> planStepMap;
  pc_list<plan_step *>::const_iterator origPlanStep = p->begin();
//  int randomNo;
  
  for(pc_list<plan_step *>::iterator ps = aplan->begin(); ps != aplan->end(); ++ps, ++origPlanStep)
  {
 /*     randomNo = rand()%2;      
      
      if(randomNo == 0 || runNo == 1) accumTime += variation;
      else accumTime -= variation;      
*/
		accumTime += variation;
      (*ps)->start_time = (*ps)->start_time + accumTime;
      //if((*ps)->start_time < 0) (*ps)->start_time = 0;

      if((*ps)->duration_given)
      {
          (*ps)->originalDuration = (*ps)->duration;

         // (*ps)->duration = (*ps)->duration;
        //  if((*ps)->duration < 0) (*ps)->duration = 0;
      };
          //cout << (*ps)->start_time << " ";
      planStepMap[*ps] = *origPlanStep;
  };
     // cout << "\n";
  return planStepMap;
};

void RobustPlanAnalyser::displayPlan()
{
  Validator * v = new Validator(derivRules,tolerance,typeC,operators,initialState,p,
                          metric,stepLength,durative,current_analysis->the_domain->constraints,current_analysis->the_problem->constraints);

  if(LaTeX)
  		{
	*report << "\\subsection{Plan}\n";
	v->displayInitPlanLaTeX(p);
	*report << "\\subsection{Plan To Analyse for Robustness}\n";
	v->displayPlan();
	*report << "\\subsection{Plan Validation Details with No Variation}\n";
  		}
  else if(Verbose)
  			v->displayPlan();

  delete v;
};

double getStandardDev(double noTests,double numberOfInvalidPlans,double mean)
{
 double sum = numberOfInvalidPlans*mean*mean + (noTests - numberOfInvalidPlans)*(1.0 - mean)*(1.0 - mean);
 
 return sqrt(sum/noTests);  
};

//need to look up in a table
double getUpperCritialValueTDistribution(double alpha,int degreesFreedom)
{
 double ans = upperCritialValueTDistribution(alpha,degreesFreedom);
 
 return ans; 
};

void RobustPlanAnalyser::analyseRobustness()
{
                           
 if(Verbose || LaTeX) displayPlan();
                              
 bool latex = LaTeX;
 bool verbose = Verbose;
 LaTeX = false; Verbose = false;
 
 int numberOfInvalidPlans = 0;
 int numberOfErrorPlans = 0;
 double actionRobustnessOfPlan = -1;
 double actionRobustnessBound = 0;
 double pneRobustnessOfPlan = -1;
 double pneRobustnessBound = 0;
 if(RobustPNEJudder != 0) TestingPNERobustness = true;
  
 if(!calcActionRobustness && !calcPNERobustness) runAnalysis(robustMeasure,noTestPlans,true,numberOfInvalidPlans,numberOfErrorPlans,false,latex);

 if(calcActionRobustness || calcPNERobustness) robustDist = UNIFORM; //best to stick to uniform distribution for calculating robustnesses
 
 if(calcPNERobustness) calculatePNERobustness(pneRobustnessOfPlan,pneRobustnessBound);
 
 if(calcActionRobustness) calculateActionRobustness(actionRobustnessOfPlan,actionRobustnessBound);

 
 LaTeX = latex; Verbose = verbose;
           
 if(LaTeX) displayAnalysisLaTeX(noTestPlans,numberOfInvalidPlans,numberOfErrorPlans,actionRobustnessOfPlan,actionRobustnessBound,pneRobustnessOfPlan,pneRobustnessBound);
 else displayAnalysis(noTestPlans,numberOfInvalidPlans,numberOfErrorPlans,actionRobustnessOfPlan,actionRobustnessBound,pneRobustnessOfPlan,pneRobustnessBound);
 
};







void RobustPlanAnalyser::runAnalysis(double & variation,int & numberTestPlans,bool recordFailures,int & numberOfInvalidPlans,int & numberOfErrorPlans,bool allValid,bool latexAdvice)
{           
 ErrorReport = recordFailures;
 ContinueAnyway = false;
 int noBoundaryTests = 0; //299;
 Validator * testPlanValidator = 0;
 
 bool lxr = LaTeXRecord;
 LaTeXRecord = latexAdvice;
 
 for(int testNo = 1; testNo <= numberTestPlans; ++testNo)
 {        
    map<const plan_step *,const plan_step *> planStepMap; //we need to keep track of which actions fail
    plan * testPlan = newTestPlan(p);
    plan * testPlan2 = new plan(*testPlan); //we can delete the test plan now without deleting the timed initial literal actions
    bool planExecuted = false; bool goalSatisfied = false;
    bool executionError = false;

    if(testNo <= noBoundaryTests) planStepMap = varyPlanTimestampsBoundary(testPlan,p,variation,testNo);
    else planStepMap = varyPlanTimestamps(testPlan,p,variation);

    //add timed initial literals to the plan from the problem spec, these time are fixed
    for(vector<plan_step *>::iterator ps = timedIntitialLiteralActions.begin(); ps != timedIntitialLiteralActions.end(); ++ps)
    {
      testPlan->push_back(*ps);
    };

    testPlanValidator = new Validator(derivRules,tolerance,typeC,operators,initialState,testPlan,
                          metric,stepLength,durative,current_analysis->the_domain->constraints,current_analysis->the_problem->constraints);

    try{    
      planExecuted = testPlanValidator->execute();
    }
    catch(exception & e)
    {
        cout << e.what() << "\n";
        executionError = true;
    };

    if(planExecuted)
    {
      goalSatisfied = testPlanValidator->checkGoal(theGoal);
      if(!goalSatisfied) unsatisfiedGoal++;
    };

    if(!planExecuted || !goalSatisfied){ numberOfInvalidPlans++; if(allValid) return;
    if(recordFailures)
    {
        bool unsatGoal = false;
       //ErrorLog errorLog = testPlanValidator->getErrorLog();
       vector<const UnsatCondition *> unSatConds = testPlanValidator->getErrorLog().getConditions(); 
       if(!unSatConds.empty())
       {
         const UnsatCondition * firstError = *(unSatConds.begin());
         const Action * theAction = 0;
         if(const UnsatPrecondition * unsatpre = dynamic_cast<const UnsatPrecondition*>(firstError))
         {
            theAction = unsatpre->action;
         }
         else if(const UnsatInvariant * unsatinv = dynamic_cast<const UnsatInvariant*>(firstError))
         {       
            theAction = unsatinv->action;
         }
         else if(const UnsatDurationCondition * unsatdur = dynamic_cast<const UnsatDurationCondition*>(firstError))
         {
            theAction = unsatdur->action;
         }
         else if(const MutexViolation * muvi = dynamic_cast<const MutexViolation*>(firstError))
         {       
            theAction = muvi->action1; //just recond one action for now to get numbers
         }
         else if(dynamic_cast<const UnsatGoal*>(firstError))
         {
            unsatGoal = true;
         }
         else
            unknownErrors++;

         if(theAction != 0 || unsatGoal)
         {
            const plan_step * aPlanStep = 0;
            if(!unsatGoal) aPlanStep = theAction->getPlanStep();
            
            if(aPlanStep != 0 || unsatGoal)
            {
              map<const plan_step *, InvalidActionReport>::iterator ps;
              if(!unsatGoal) ps = record.find(planStepMap[aPlanStep]);
              else ps = record.find(0);
              
              if(ps != record.end())
              {
                (ps->second.number)++;
                 bool lx = LaTeX;
                 LaTeX = latexAdvice;                
                 string reason = firstError->getDisplayString();
                 LaTeX = lx;
                 map<string,pair<int,string> >::iterator r = ps->second.failReasons.find(reason);
                 if(r != ps->second.failReasons.end()) (r->second.first)++;
                 else
                 {
                   bool lx = LaTeX;
                   LaTeX = latexAdvice;                
                   ps->second.failReasons[reason] =  make_pair(1,firstError->getAdviceString());
                   LaTeX = lx;
                 };
                
              }
              else
              { bool lx = LaTeX;
                LaTeX = latexAdvice;
                record[planStepMap[aPlanStep]] = InvalidActionReport(1,firstError->getDisplayString(),firstError->getAdviceString());
                LaTeX = lx;   
              };
            }
            else
               unknownErrors++;
         };
       }
       else
        unknownErrors++;

    }
    else if(executionError)
    {
       numberOfErrorPlans++;
    };};

     if(LaTeX)
     {
       if(planExecuted)
       {
         *report << "Plan executed successfully - checking goal\\\\\n";
         if(goalSatisfied)
         {
          	*report << "Goal satisfied\\\\\n" << "Final value: ";
          	vector<double> vs(testPlanValidator->finalValue());
          	copy(vs.begin(),vs.end(),ostream_iterator<double>(*report," "));
          	*report <<"\n";
         }
         else *report << "Goal not satisfied\n";
       }
       else
         *report << "\nPlan failed to execute\n";
       maxTime = testPlanValidator->getMaxTime();  
       if(testPlanValidator->graphsToShow()) latex.LaTeXGraphs(testPlanValidator);
       latex.LaTeXGantt(testPlanValidator);
     };
    
    deleteTestPlan(testPlan2);
    testPlan->clear(); delete testPlan;
    delete testPlanValidator;
 };

  LaTeXRecord = lxr;
};

//how much can the time stamps vary, so that we can be 95% sure that the plan will always be valid (well at least 95%)
void RobustPlanAnalyser::calculateActionRobustness(double & robustnessOfPlan,double & robustnessBound)
{
 double lowerBound = 0.0;
 double upperBound = getMaxTime(p);
 double testValue = (upperBound + lowerBound)/2.0; 
 double lowerBoundLowerBound = 0.000001;
 bool robustnessFound = false;
 int numberOfTestPlans = 598;
 int noOfInvalidPlans = 0;
 int noOfErrorPlans = 0;

     runAnalysis(upperBound,numberOfTestPlans,false,noOfInvalidPlans,noOfErrorPlans,true,false);

     if(noOfInvalidPlans == 0)
     {
      robustnessOfPlan = -1;
      robustnessBound  = -1;
      return;
      };
  
 
 do{
      testValue = (upperBound + lowerBound)/2.0; noOfInvalidPlans = 0;
               //LaTeX = true;
      runAnalysis(testValue,numberOfTestPlans,false,noOfInvalidPlans,noOfErrorPlans,true,false);
           //cout << testValue << " testValue, "<< noOfInvalidPlans << " invalid plans \\\\\n";
      if(noOfInvalidPlans == 0)
      {
        lowerBound = testValue; 
      }
      else
      {
        upperBound = testValue; 
      };
  
    if(lowerBound == 0)
    {
      if(upperBound < lowerBoundLowerBound) robustnessFound = true;
    }
    else
    {
      if(upperBound - lowerBound < 0.01) robustnessFound = true;
    };

 }while(!robustnessFound);

 robustnessOfPlan = (lowerBound + upperBound)/2.0;
 robustnessBound  = (upperBound - lowerBound)/2.0; 
};

//how much can the time stamps vary, so that we can be 95% sure that the plan will always be valid (well at least 95%)
void RobustPlanAnalyser::calculatePNERobustness(double & robustnessOfPlan,double & robustnessBound)
{
 TestingPNERobustness = true;
 double oldRobustPNEJudder = RobustPNEJudder;
 double lowerBound = 0.0;
 double upperBound = 10.0;
 double upperBoundUpperBound = 150.0;
 double lowerBoundLowerBound = 0.000001;
 bool robustnessFound = false;
 int numberOfTestPlans = 299;
 int noOfInvalidPlans = 0;
 int noOfErrorPlans = 0;
 bool upperBoundFixed = false;

 do{
     RobustPNEJudder = upperBound;
     runAnalysis(robustMeasure,numberOfTestPlans,false,noOfInvalidPlans,noOfErrorPlans,true,false);

     if(noOfInvalidPlans == 0) upperBound += 10.0;
     else upperBoundFixed = true;

     if(upperBound > upperBoundUpperBound)
     {
        robustnessOfPlan = upperBound;
        robustnessBound  = 0.0;
        RobustPNEJudder = oldRobustPNEJudder;
        return;
     };

 }while(!upperBoundFixed);

 do{
      RobustPNEJudder = (upperBound + lowerBound)/2.0; noOfInvalidPlans = 0;
             
      runAnalysis(robustMeasure,numberOfTestPlans,false,noOfInvalidPlans,noOfErrorPlans,true,false);
          //cout << RobustPNEJudder << " RobustPNEJudder, "<< noOfInvalidPlans << " invalid plans \\\\\n";
      if(noOfInvalidPlans == 0)
      {
        lowerBound = RobustPNEJudder;
      }
      else
      {
        upperBound = RobustPNEJudder;
      };

    if(lowerBound == 0)
    {
      if(upperBound < lowerBoundLowerBound) robustnessFound = true;
    }
    else
    {
      if(upperBound - lowerBound < 0.01) robustnessFound = true;
    };

 }while(!robustnessFound);

 robustnessOfPlan = (lowerBound + upperBound)/2.0;
 robustnessBound  = (upperBound - lowerBound)/2.0;
 RobustPNEJudder = oldRobustPNEJudder;
};

string getPlanStepString(const plan_step * ps)
{
    if(ps == 0) return "";
		string act = "("+ps->op_sym->getName();
		for(typed_symbol_list<const_symbol>::const_iterator j = ps->params->begin();
			j != ps->params->end(); ++j)
		{
			act += " " + (*j)->getName();
		};
		act += ")";
		if(ps->duration_given) act += " [" + toString(ps->duration) + "]";
    if(LaTeX) latexString(act);
    return act;
};
    
void RobustPlanAnalyser::displayAnalysis(int noTestPlans,int numberOfInvalidPlans,int numberOfErrorPlans,double actionRobustnessOfPlan,double actionRobustnessBound,double pneRobustnessOfPlan,double pneRobustnessBound)
{
 int noTests = noTestPlans - numberOfErrorPlans;
 double validPlans = noTests - numberOfInvalidPlans;
 double estProbValidPlan = double(validPlans) / double(noTests);
 double standardDev = getStandardDev(noTests,numberOfInvalidPlans,estProbValidPlan);
 double upperCritialValTDistrib = getUpperCritialValueTDistribution(0.05,noTests - 1); //we need to look this up in a table, this is for alpher = 0.05 and N = noTests-1
 double errorLimit = (upperCritialValTDistrib*standardDev)/(sqrt(double(noTests)));
  cout << "Plan Robustness Report:\n";
  cout << "-----------------------\n\n";

  if(!calcActionRobustness && !calcPNERobustness)
  {
     if(noTestPlans - numberOfErrorPlans - numberOfInvalidPlans == 1) cout << "1 plan is valid from ";
     else cout << noTestPlans - numberOfErrorPlans - numberOfInvalidPlans << " plans are valid from ";
     cout << noTestPlans - numberOfErrorPlans << " plans for each action timestamp +-"<<robustMeasure<<" and each PNE +-"<<RobustPNEJudder<<".\n\n";

     if(estProbValidPlan != 1 && estProbValidPlan != 0)
       cout << "There is a 95% chance that the plan has a valid execution with probability in the range "<< estProbValidPlan*100.0 << " +-"<< errorLimit*100.0 <<".\n";
     else if(estProbValidPlan == 1)
          cout << "There is a 99% chance that the plan will be valid with probability of at least "<<100*pow(0.01,(1.0/(noTestPlans - numberOfErrorPlans)))<<"%.\n";
     else if(estProbValidPlan == 0)
          cout << "There is a 99% chance that the plan will fail with probability of at least "<<100*pow(0.01,(1.0/(noTestPlans - numberOfErrorPlans)))<<"%.\n";
     
  };

 if(calcActionRobustness)
 {
       if(actionRobustnessOfPlan == -1)
       {
         *report << "The plan has an action timestamp robustness greater than the length of the plan, "<<getMaxTime(p); 
       }
       else       
         *report << "The plan has an action timestamp robustness in the range "<< actionRobustnessOfPlan <<" +-"<<actionRobustnessBound;

       if(RobustPNEJudder > 0) *report << " when PNEs may vary up to "<< RobustPNEJudder <<".\n";
       else *report <<".\n";
 };
 
 if(calcPNERobustness)
 {
       *report << "The plan has a PNE robustness in the range "<< pneRobustnessOfPlan <<" +-"<<pneRobustnessBound;

       if(robustMeasure > 0) *report << " when action timestamps may vary up to "<< robustMeasure <<".\n";
       else *report <<".\n";
 };

 *report << "\nMetric: "<<getMetricName()<<"\n";
 if(!calcActionRobustness && !calcPNERobustness) *report << "Distribution: "<<getDistName()<<"\n";  
 *report << "\n";


 if(Verbose && numberOfInvalidPlans != 0)
 {
   vector<InvalidActionRecord> iar;

  	for(pc_list<plan_step *>::const_iterator i = p->begin() ; i != p->end() ; ++i)
    {
      map<const plan_step *, InvalidActionReport>::const_iterator k = record.find(*i);
      if(k != record.end()) iar.push_back(InvalidActionRecord(k->second.number,(*i)->start_time,*i));
    };
    
    
   cout << "Plan failures:\n\n";

    for(vector<InvalidActionRecord>::const_iterator d = iar.begin(); d != iar.end(); ++d)
   {
     if(d->number == 1) cout << "1 failure for "<<d->time << ": ";
     else cout << d->number << " failures for "<<d->time << ": ";
     cout << getPlanStepString(d->ps) << "\n";
   };

   if(unsatisfiedGoal == 1) cout << "1 plan failed because the goal is not satisfied.\n";
   else if(unsatisfiedGoal != 0) cout << unsatisfiedGoal << " plans failed because the goal is not satisfied.\n";
    
   if(unknownErrors == 1) cout << "1 other plan failure.\n";
   else if(unknownErrors != 0) cout << unknownErrors << " other plan failures.\n";
     
   if(numberOfErrorPlans == 1) cout << "1 plan did not execute due errors in the validation process.\n";
   else if(numberOfErrorPlans != 0) cout << numberOfErrorPlans << " plans did not execute due errors in the validation process.\n";
       
   cout << "\n";

    if(record.size() == 0) return;

   cout << "Reasons for Plan Failures:\n\n";
  
   for(map<const plan_step *,InvalidActionReport>::const_iterator d = record.begin(); d != record.end(); ++d)
   {      
     string theTime;
     if(d->first != 0) theTime = toString(d->first->start_time);
     else theTime = " the goal";
     if(d->second.number == 1) cout << "1 failure for "<<theTime<<" ";
     else cout << d->second.number << " failures for "<<theTime<<" ";
     cout << getPlanStepString(d->first) << "\n";
     cout <<" ";
     for(map<string,pair<int,string> >::const_iterator fr = d->second.failReasons.begin(); fr != d->second.failReasons.end(); ++fr)
     {
       if(fr->second.first == 1) *report << " 1 failure: ";
       else cout << " "<<fr->second.first <<" failures: ";
       cout <<fr->first<<". Sample plan repair advice: ";
       cout << "  "<<fr->second.second <<"  \n";       
     };
     cout << " \n";
   };
   
 };
 
};


void RobustPlanAnalyser::displayAnalysisLaTeX(int noTestPlans,int numberOfInvalidPlans,int numberOfErrorPlans,double actionRobustnessOfPlan,double actionRobustnessBound,double pneRobustnessOfPlan,double pneRobustnessBound)
{     
 double zeroVary = 0.0; int oneTest = 1; int noIP; int noEP;
 double rpnej = RobustPNEJudder;
 RobustPNEJudder = 0;
 runAnalysis(zeroVary,oneTest,false,noIP,noEP,false,false);//for display purposes
 RobustPNEJudder = rpnej;
 
    
 *report << "\\subsection{Plan Robustness Report}\n";      
 int noTests = noTestPlans - numberOfErrorPlans;
 double validPlans = noTests - numberOfInvalidPlans;
 double estProbValidPlan = double(validPlans) / double(noTests);
 double standardDev = getStandardDev(noTests,numberOfInvalidPlans,estProbValidPlan);
 double upperCritialValTDistrib = getUpperCritialValueTDistribution(0.05,noTests - 1); //we need to look this up in a table, this is for alpher = 0.05
 double errorLimit = (upperCritialValTDistrib*standardDev)/(sqrt(double(noTests)));

 *report << "\\begin{itemize}\n";
 if(!calcActionRobustness && !calcPNERobustness)
 {
       *report << "\\item ";
       if(noTestPlans - numberOfErrorPlans - numberOfInvalidPlans == 1) *report << "1 plan is valid from ";
       else *report <<noTestPlans - numberOfErrorPlans - numberOfInvalidPlans << " plans are valid from ";
       *report << noTestPlans - numberOfErrorPlans << " plans for each action timestamp $\\pm$"<<robustMeasure<<" and each PNE $\\pm$"<<RobustPNEJudder<<".\\\\\n";

       *report << "\\item ";
        if(estProbValidPlan != 1 && estProbValidPlan != 0)
       *report << "There is a 95$\\%$ chance that the plan has a valid execution with probability in the range "<< estProbValidPlan*100.0 << " $\\pm$"<< errorLimit*100.0 <<".\\\\\n";
        else if(estProbValidPlan == 1)
       *report << "There is a 99$\\%$ chance that the plan will be valid with probability of at least "<<100*pow(0.01,(1.0/(noTestPlans - numberOfErrorPlans)))<<"$\\%$.\\\\\n";
        else if(estProbValidPlan == 0)
       *report << "There is a 99$\\%$ chance that the plan will fail with probability of at least "<<100*pow(0.01,(1.0/(noTestPlans - numberOfErrorPlans)))<<"$\\%$.\\\\\n";
 
 };

 
 if(calcActionRobustness)
 {
       *report << "\\item ";
       if(actionRobustnessOfPlan == -1)
       {
         *report << "The plan has an action timestamp robustness greater than the length of the plan, "<<getMaxTime(p);
       }
       else 
         *report << "The plan has an action timestamp robustness in the range "<< actionRobustnessOfPlan <<"$\\pm$"<<actionRobustnessBound;
       
       if(RobustPNEJudder > 0) *report << " when PNEs may vary up to "<< RobustPNEJudder <<".\\\\\n";
       else *report <<".\\\\\n";
 };
 
 if(calcPNERobustness)
 {
       *report << "\\item ";
       *report << "The plan has a PNE robustness in the range "<< pneRobustnessOfPlan <<"$\\pm$"<<pneRobustnessBound;
       if(robustMeasure > 0) *report << " when action timestamps may vary up to "<< robustMeasure <<".\\\\\n";
       else *report <<".\\\\\n";
 };

 *report << "\\item Plan-tube: "<<getMetricName();
 if(!calcActionRobustness && !calcPNERobustness) *report <<"\\item Distribution: "<<getDistName();
 *report <<"\\\\\n";
 *report << "\\end{itemize}\n";
 
 if(numberOfInvalidPlans != 0)
 {
  map<double,int> noErrors;
  vector<InvalidActionRecord> iar;

	for(pc_list<plan_step *>::const_iterator i = p->begin() ; i != p->end() ; ++i)
  {
    map<const plan_step *, InvalidActionReport>::const_iterator k = record.find(*i);
    if(k != record.end())
    {
     map<double,int>::const_iterator j = noErrors.find((*i)->start_time);
     if(j != noErrors.end()) noErrors[(*i)->start_time] += k->second.number;
     else noErrors[(*i)->start_time] = k->second.number;

     iar.push_back(InvalidActionRecord(k->second.number,(*i)->start_time,*i));
    };
  };
  if(unsatisfiedGoal != 0)
  {
     map<double,int>::const_iterator j = noErrors.find(maxTime);
     if(j != noErrors.end()) noErrors[maxTime] += unsatisfiedGoal;
     else noErrors[maxTime] = unsatisfiedGoal; 
  };
  
   *report << "\\subsubsection{Plan Failures}\n";
   string act;
   *report << "\\begin{tabbing}\n {\\bf Failures} \\= {\\bf Time} \\qquad \\= {\\bf Action}\\\\[0.8ex]\n";
   //for(map<const plan_step *, int>::const_iterator ps = record.begin(); ps != record.end(); ++ps)
   for(vector<InvalidActionRecord>::const_iterator d = iar.begin(); d != iar.end(); ++d)
   {
     if(d->number == 1) *report << "1 \\> \\atime{"<<d->time << "} \\>";
     else *report << d->number << "\\> \\atime{"<<d->time << "} \\>";
     *report << "\\listrow{\\action{"<<getPlanStepString(d->ps) << "}}\\\\\n";
      
   };
   
   
   if(unsatisfiedGoal == 1) *report << "1 \\> \\> Plan failed because the goal is not satisfied.\\\\\n";
   else if(unsatisfiedGoal != 0) *report << unsatisfiedGoal << " \\> \\> Plans failed because the goal is not satisfied.\\\\\n";
    
   if(unknownErrors == 1) *report << "1 \\> \\> other plan failure.\\\\\n";
   else if(unknownErrors != 0) *report << unknownErrors << " \\> \\> other plan failures.\\\\\n";
       
   if(numberOfErrorPlans == 1) *report << "1  \\> \\> plan did not execute due errors in the validation process.\\\\\n";
   else if(numberOfErrorPlans != 0) *report << numberOfErrorPlans << "  \\> \\> plans did not execute due errors in the validation process.\\\\\n";

   *report << "\\end{tabbing}\n";    

   if(record.size() == 0) return;
  
   *report << "\\subsubsection{Reasons for Plan Failures}\n";   
   *report << "\\begin{enumerate}\n";
   for(map<const plan_step *,InvalidActionReport>::const_iterator d = record.begin(); d != record.end(); ++d)
   {
     *report << "\\item ";
     string theTime;
     if(d->first != 0) theTime = "\\atime{"+toString(d->first->start_time)+"}";
     else theTime = " the goal";
     if(d->second.number == 1) *report << "1 failure for "<<theTime<<" ";
     else *report << d->second.number << " failures for "<<theTime<<" "; 
     *report << "\\listrow{\\action{"<<getPlanStepString(d->first) << "}}\\\\\n";
     *report << "\\begin{enumerate}\n";
     for(map<string,pair<int,string> >::const_iterator fr = d->second.failReasons.begin(); fr != d->second.failReasons.end(); ++fr)
     {
       if(fr->second.first == 1) *report << "\\item 1 failure: ";
       else *report << "\\item "<<fr->second.first <<" failures: ";
       *report <<fr->first<<".\\\\ Sample plan repair advice:\\\\\n";
       *report << "\\begin{enumerate}\n";
       *report <<fr->second.second <<"\\\\\n";
       *report << "\\end{enumerate}\n";
     };
     *report << "\\end{enumerate}\n";
   };
   *report << "\\end{enumerate}\n";
  
   
   *report << "\\subsection{Graphs of Failed Plans}\n";
	 *report << "\\setcounter{figure}{0}\n";
   FEGraph errorGraph("Number of plans failing at these times",0,0);
   FEGraph accumErrorGraph("Accumulative number of plans failing by these times",0,0);
   FEGraph errorPerGraph("Percentage of plans failing at these times",0.0,100.0);
   FEGraph accumPerErrorGraph("Accumulative percentage of plans failing by these times",0.0,100.0);
   FEGraph errorGraphL("Number of plans failing at these times",0,0);
   FEGraph accumErrorGraphL("Accumulative number of plans failing by these times",0,0);
   FEGraph errorPerGraphL("Percentage of plans failing at these times",0.0,100.0);
   FEGraph accumPerErrorGraphL("Accumulative percentage of plans failing by these times",0.0,100.0);
   int prevValue = 0; int accPrevValue = 0;
   double prevPerValue = 0; double accPerPrevValue = 0; 

   //ensure these appear in the correct order!
    //for(map<const plan_step *, int>::const_iterator ps = record.begin(); ps != record.end(); ++ps)
   for(map<double,int>::const_iterator g = noErrors.begin(); g != noErrors.end(); ++g)
   {                   
      errorGraphL.points[g->first] =  g->second;
      errorGraphL.happenings.insert(g->first);

      accumErrorGraphL.points[g->first] =  accPrevValue + g->second;
      accumErrorGraphL.happenings.insert(g->first);

      errorPerGraphL.points[g->first] = ((g->second)/double(noTests))*100.0;
      errorPerGraphL.happenings.insert(g->first);

      accumPerErrorGraphL.points[g->first] =  accPerPrevValue + ((g->second)/double(noTests))*100.0;
      accumPerErrorGraphL.happenings.insert(g->first);
     
      errorGraph.discons[g->first] =  make_pair(prevValue,g->second);
      prevValue = g->second;
      errorGraph.happenings.insert(g->first);

      accumErrorGraph.discons[g->first] =  make_pair(accPrevValue,accPrevValue + g->second);
      accPrevValue = accPrevValue + g->second;
      accumErrorGraph.happenings.insert(g->first);

      errorPerGraph.discons[g->first] =  make_pair(prevPerValue,((g->second)/double(noTests))*100.0);
      prevPerValue = ((g->second)/double(noTests))*100.0;
      errorPerGraph.happenings.insert(g->first);

      accumPerErrorGraph.discons[g->first] =  make_pair(accPerPrevValue,accPerPrevValue + ((g->second)/double(noTests))*100.0);
      accPerPrevValue = accPerPrevValue + ((g->second)/double(noTests))*100.0;
      accumPerErrorGraph.happenings.insert(g->first);
    
   };

   errorGraph.displayLaTeXGraph(maxTime);
   accumErrorGraph.displayLaTeXGraph(maxTime);
   errorPerGraph.displayLaTeXGraph(maxTime);
   accumPerErrorGraph.displayLaTeXGraph(maxTime);
   errorGraphL.displayLaTeXGraph(maxTime);
   accumErrorGraphL.displayLaTeXGraph(maxTime);
   errorPerGraphL.displayLaTeXGraph(maxTime);
   accumPerErrorGraphL.displayLaTeXGraph(maxTime);
  
 };
 
};

};

