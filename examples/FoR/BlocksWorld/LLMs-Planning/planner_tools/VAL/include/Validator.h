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

  $Date: 2009-02-05 10:50:24 $
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
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include "Action.h"
#include "Plan.h"
#include "State.h"
#include "Polynomial.h"
#include "Events.h"

#ifndef __VALIDATOR
#define __VALIDATOR
#include "typecheck.h"
#include "RepairAdvice.h"
#include "TrajectoryConstraints.h"


namespace VAL {

  extern bool makespanDefault;
  extern bool stepLengthDefault;
  
class plan;
class TypeChecker;

struct FEGraph{

	const FuncExp * fe;
    string title; 
	map<double,double> points;
	set<double> happenings;
	map<double,pair<double,double> > discons; //discontinuities; time , left hand side value and right
	double initialTime; //first time FE is defined, -1 denotes time not yet defined
	double initialValue;
		
	double maxValue;
	double minValue;
	double timeAxisV;

	static const double graphMaxH;
	static const double graphMaxV;
	static const int pointSize;

	
	FEGraph(const FuncExp * f) : fe(f),title(""),points(),happenings(),discons(),initialTime(-1),initialValue(0),maxValue(0),minValue(0) {};
  FEGraph(string t,double min,double max) : fe(0),title(t),points(),happenings(),discons(),initialTime(0),initialValue(0),maxValue(max),minValue(min) {};
	~FEGraph();
	
	
	void setMinMax();
	void displayLaTeXGraph(double maxTime);
	void drawLaTeXAxis(double maxTime) const;
	void drawLaTeXLine(double t1,double y1,double t2,double y2,double maxTime) const;
   void amendPoints(double maxTime);
};

struct GanttElement{
	double start;
	double end;
	string label;

	vector<string> sigObjs; //significant objects for this Element

	
	GanttElement(double s,double e,string l,vector<string> so) : start(s),end(e),label(l),sigObjs(so) {};
	
};

struct Gantt{

	double maxTime;
	vector<string>	sigObjs;//significant objects, will tried to be grouped in gantt chart
	vector<string>	usedSigObjs;
	
	map<int, map<int, GanttElement *> > chartRows;

	static const double graphH;
	static const double graphV;
	static const int pointSize;
	
	
	Gantt() : maxTime(0), sigObjs(), usedSigObjs(), chartRows()  {};

  ~Gantt();
  
	void buildRows(const Plan & p);
	void shuffleRows();
	void swapRows(int r1,int r2);
	void insertRow(int r1,int r2);
	string getSigObj(int r);
	string getColour(int r);
	void setMaxTime(const Plan & p);
	void setSigObjs(vector<string> & objects);
	vector<string> getSigObjs(const Action * a);
	void drawLaTeXGantt(const Plan & p,int noPages,int noPageRows);
	void drawLaTeXGantt(double startTime,double endTime,int startRow,int endRow,int numRows);
	int getNoPages(int noPages);
	int getNoPageRows();
	
	void displayKey();
	void drawRowNums(int startRow,int endRow,int numRows);
	void drawLaTeXDAElement(const GanttElement * ge,int row,int pos,double startTime,double endTime,int numRows) const;
	void drawLaTeXElement(const GanttElement * ge,int row,int pos,double startTime,double endTime,int numRows) const;
	pair<double,double> transPoint(double x,double y) const;
	
};

class DerivationRules {
private:
	derivations_list * drvs;
	const operator_list * ops;
	map<string,pair<const goal *,const var_symbol_table *> > derivPreds;
	vector<const disj_goal *> repeatedDPDisjs;//used to keep list of disj to be deleted  
public:
	
	DerivationRules(const derivations_list * d,const operator_list * o);
	~DerivationRules();

	bool checkDerivedPredicates() const;
	bool stratification() const;
	unsigned int occurNNF(derivation_rule * drv1,derivation_rule * drv2) const;
	unsigned int occur(string s,const goal * g) const;
	const goal * NNF(const goal * gl) const;
	bool effects() const;
	bool effects(const effect_lists* efflist) const;
	bool isDerivedPred(string s) const;
	map<string,pair<const goal *,const var_symbol_table *> > getDerivPreds() const {return derivPreds;};
};

void changeVars(goal * g,map<parameter_symbol*,parameter_symbol*> varMap);
void changeVars(expression * e,map<parameter_symbol*,parameter_symbol*> varMap);

class GoalTracker {
private:
	const goal * finalGoal;
	goal_list trajGoals;

public:
	GoalTracker(const goal * goals,const con_goal * constraints);

};

class Validator {
public:
	FuncExpFactory fef;
	PropositionFactory pf;
    
private:
   ErrorLog  errorLog; 
	const DerivationRules * derivRules;
   Events events;

	double tolerance;
	TypeChecker & typeC;
	
	const metric_spec * metric;
	
	int stepcount;
	bool stepLength;
	bool durative;
	double maxTime;
	
	vector<string> invariantWarnings;
	vector<Action *> actionRegistry;
	map<const FuncExp *,FEGraph *> graphs;
	Gantt gantt;
	
	Plan theplan;
	State state;

	State * finalInterestingState;
	Plan::const_iterator followUp;

	Plan::const_iterator thisStep;

	map<string,int> violations;

	TrajectoryConstraintsMonitor tjm;

	bool step();

	void computeMetric(const State *,vector<double> &) const;
  
public:
	Validator(const DerivationRules * dr,double tol,TypeChecker & tc,const operator_list * ops,const effect_lists * is,const plan * p,const metric_spec * m,
					bool lengthDefault,bool isDur,con_goal * cg1,con_goal * cg2) :
		fef(), pf(this), errorLog(), derivRules(dr), events(ops), tolerance(tol), typeC(tc), metric(m), stepcount(p->size()),
		stepLength(lengthDefault), durative(isDur), invariantWarnings(), actionRegistry(),
		graphs(),gantt(),  theplan(this,ops,p), state(this,is), finalInterestingState(0), followUp(theplan.end()), thisStep(theplan.begin() ),
		tjm(this,cg1,cg2)
	{Polynomial::setAccuracy(tol);};
	~Validator();

	bool execute();
	bool checkGoal(const goal * g);
	vector<double> finalValue() const;
	int simpleLength() const;
	bool durativePlan() const;
	double getTolerance() const {return tolerance;};
	void registerAction(Action * a) {actionRegistry.push_back(a);};
	void addInvariantWarning(string s) {invariantWarnings.push_back(s);};
	bool hasInvariantWarnings() const {return (!(invariantWarnings.empty()));};
  
	void displayInvariantWarnings() const;
	void displayPlan() const;
   void displayInitPlanLaTeX(const plan * p) const;
	void displayInitPlan(const plan * p) const;
	void displayLaTeXGraphs() const;
	void setMaxTime();
	void setSigObjs(vector<string> & objects);
	double getMaxTime() {return maxTime;};
	bool graphsToShow() const;
 	FEGraph * getGraph(const FuncExp * fe);
 void drawLaTeXGantt(int noPages,int noPageRows);
   
   bool hasEvents() const {return events.hasEvents();};
   Events & getEvents() {return events;};
   double getNextHappeningTime() const;
   double getCurrentHappeningTime() const;
   bool isLastHappening() const;
   
   ErrorLog & getErrorLog() {return errorLog;};
   analysis * getAnalysis() const {return current_analysis;};
   void adjustActiveCtsEffects(ActiveCtsEffects * ace) {(*thisStep)->adjustActiveCtsEffects(*ace);};
   ActiveCtsEffects * getActiveCtsEffects() {return thisStep.getActiveCtsEffects();};
   ExecutionContext * getExecutionContext() {return thisStep.getExecutionContext();};
	
	const State & getState() const {return state;};
	void setState(const effect_lists * effs)
	{
		state.setNew(effs);
	};
   bool executeHappening(const Happening * h);
   bool executeHappeningCtsEvent(const Happening * h);
	const DerivationRules * getDerivRules() const {return derivRules;};
	
	vector<const_symbol*> range(const var_symbol * v);

	Plan::const_iterator begin() const {return theplan.begin();};

	Plan::const_iterator end() const {return theplan.end();};
	Plan::const_iterator recoverStep() {return thisStep;};
	
	double timeOf(const Action * a) const;
	void countViolation(const State * s,const string & nm,const AdviceProposition * a)
	{
// Do we want to count this violation? It will depend on whether it is being
// evaluated in the execution phase or repair phase.
		++violations[nm];
		if(Verbose)
		{
			if(LaTeX)
			{
				*report << "Preference " << nm << ": ";
				a->displayLaTeX();
				*report << " violated at " << s->getTime() << "\\\\\n";
			}
			else
			{
				cout << "Preference " << nm << ": ";
				a->display();
				cout << " violated at " << s->getTime() << "\n";
			};
		};
	};
	int violationsFor(const string & nm)
	{
		return violations[nm];
	};
	void reportViolations() const;
	void resetStep(const Plan::const_iterator & n) {thisStep = n;};
};

class PlanRepair {
private:

    const plan * p;
    vector<plan_step *> timedInitialLiteralActions;
    double deadLine;
    
    Validator v;   //contains ErrorLog with vector of unsatisfied conditions each owning an AdviceProposition

//the following are used for creating new Validator objects
	TypeChecker & typeC;
	const metric_spec * metric;
	bool stepLength;
	bool durative;
	const operator_list * operators;
	const effect_lists * initialState;
  analysis * current_analysis;
//for checking the goal
   const goal * theGoal;
public:

     PlanRepair(vector<plan_step *> initLits,double dl,const DerivationRules * dr,double tol,TypeChecker & tc,const operator_list * ops,const effect_lists * is,const plan * p1,const plan * p2,const metric_spec * m,
					bool lengthDefault,bool isDur,const goal * g,analysis * ca) :  p(p2), timedInitialLiteralActions(initLits), deadLine(dl), v(dr,tol,tc,ops,is,p1,m,lengthDefault,isDur,ca->the_domain->constraints,ca->the_problem->constraints),
          typeC(tc), metric(m), stepLength(lengthDefault), durative(isDur), operators(ops), initialState(is), theGoal(g)
          {};

  ~PlanRepair() {};
  void setDeadline(double d)
  {
  	deadLine = d;
  };
  void setPlanAndTimedInitLits(const plan * p, set<plan_step *> lockedActions);
  void advice(ErrorLog & el);
  void firstPlanAdvice();
  Validator & getValidator() {return v;};
  vector<const UnsatCondition *> getUnSatConditions() {return v.getErrorLog().getConditions();}; //conditions of only Validator object for now
  const plan * getPlan() const {return p;};
  bool isInTimeInitialLiteralList(const plan_step * ps);  
  set<const Action *> getUniqueFlawedActions(Validator * vld);
  bool repairPlanBeagle();
  void repairPlan();
  pair<const plan_step *,pair<bool,bool> > repairPlanOneAction(const plan * repairingPlan,const plan_step * firstAction);
  bool repairPlanOneActionAtATime(const plan * repairingPlan,const plan_step * firstAction);
  bool slideEndOfPlan(const plan * repairingPlan,const plan_step * firstAction);
  bool shakePlan(const plan * repairingPlan,const plan_step * firstAction,double variation);
  void setState(const effect_lists * effs)
  {
  	initialState = effs;
  	v.setState(effs);
  };
};

double getMaxTime(const plan * aPlan);

};

#endif
