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
#include "RepairAdvice.h"
#include "Validator.h"
#include "Utils.h"

namespace VAL {
  
auto_ptr<UnsatConditionFactory> ErrorLog::fac(new UnsatConditionFactory);

string UnsatCondition::getAdviceString() const
{
   
   string ans;
   ostringstream aStringStream;
   ostream * oldReport = report;
   report = &aStringStream;
   advice();
   
		ans = aStringStream.str();

   report = oldReport;
   return ans;
};

void UnsatPrecondition::display() const
{
    *report << *action << " has an unsatisfied precondition at time "<< time; 
     if(LaTeX) *report << "\\\\";
     *report <<"\n";
};

string UnsatPrecondition::getDisplayString() const
{
    string ans = "The precondition is unsatisfied";
    return ans;
};

void UnsatPrecondition::advice() const
{
    if(LaTeX) *report <<"\\item "; else *report <<"\n";
    
   display();

    action->displayDurationAdvice(&state);

    if(ap->isAdvice()) 
    {
    	if(LaTeX) ap->displayLaTeX(); else ap->display();  
    }
};

void UnsatDurationCondition::display() const
{
    *report << *action << " has an unsatisfied duration constraint at time "<< time; 
     if(LaTeX) *report << "\\\\";
     *report <<"\n";
};

string UnsatDurationCondition::getDisplayString() const
{
    string ans = "The duration constraint is unsatisfied";
    return ans;
};

void UnsatDurationCondition::advice() const
{
    if(LaTeX) *report <<"\\item "; else *report <<"\n";
    display();

    *report <<"Change the duration by at least "<<error<<"!";
    if(LaTeX) *report << "\\\\";
    *report <<"\n";
};

void MutexViolation::display() const
{
    //*report << *action1 << " and "<<action2<<" are mutex at time "<< time;
    *report << *action1 << " has a mutex violation at time "<< time;
    if(action2 != 0) *report << " with "+ action2->getName();
     if(LaTeX) *report << "\\\\";
     *report <<"\n";
};

string MutexViolation::getDisplayString() const
{
    string ans = action1->getName() + " has a mutex violation";
    if(action2 != 0) ans += " with "+ action2->getName();
    return ans;
};

void MutexViolation::advice() const
{
    if(LaTeX) *report <<"\\item "; else *report <<"\n";
    display();

    *report <<"Separate these actions!";
    if(LaTeX) *report << "\\\\";
    *report <<"\n";
};

void UnsatGoal::display() const
{
    *report << "The goal is not satisfied";
     if(LaTeX) *report << "\\\\";
     *report <<"\n";
};

string UnsatGoal::getDisplayString() const
{
    string ans = "The goal is not satisfied";
    return ans;
};

   
void UnsatGoal::advice() const
{
    if(LaTeX) *report <<"\\item "; else *report <<"\n";
    display();

    if(LaTeX) ap->displayLaTeX(); else ap->display();
};

void UnsatInvariant::display() const
{
     if(LaTeX) *report <<"\\item ";
    *report << *action << " has its condition unsatisfied between time "<< startTime <<" to "<< endTime << ", ";
    
    if(!rootError)
    {
      *report << "the condition is satisfied on ";
      satisfiedOn.writeOffset(startTime); *report << ". ";
      }
      else
      {
        *report << "Sorry there were problems computing the intervals!";
      };

     // if(LaTeX) *report << " For each $t$ in $( 0, "<<endTime-startTime<<")$ follow:\\\\";
     // else *report << " For each t in ( 0, "<<endTime-startTime<<") follow:\n"; 
     if(!LaTeX) *report << "\n"; 
};

string UnsatInvariant::getDisplayString() const
{
    string ans =  "The invariant condition is unsatisfied";
    
  return ans;
};

void UnsatInvariant::advice() const
{     
    if(!LaTeX) *report <<"\n";
    display();

    if(ap->isAdvice()) 
    {
    	if(LaTeX) ap->displayLaTeX(); else ap->display();
    }
};

void ErrorLog::displayReport() const
{
  if(conditions.size() == 0) return;
  if(LaTeX) *report << "\\subsection{Error Report}\n";
  else *report << "\nError Report:\n";
  for(vector<const UnsatCondition *>::const_iterator i = conditions.begin();i != conditions.end();++i)
  	{
  	     (*i)->display();
  	};
   
};

ErrorLog::~ErrorLog()
{       
	for(vector<const UnsatCondition *>::const_iterator i = conditions.begin();i != conditions.end();++i)
	{
			delete (*i);
	}; 
};

void ErrorLog::addPrecondition(double t, const Action * a, const State * s)
{
  const UnsatCondition * pre = fac->buildUnsatPrecondition(t,a,s);
  conditions.push_back(pre);  
};

void ErrorLog::addUnsatDurationCondition(double t, const Action * a, const State * s,double e)
{
  const UnsatCondition * pre = fac->buildUnsatDurationCondition(t,a,s,e);
  conditions.push_back(pre);
};

void ErrorLog::addMutexViolation(double t, const Action * a1, const Action * a2, const State * s)
{
  const UnsatCondition * pre = fac->buildMutexViolation(t,a1,a2,s);
  conditions.push_back(pre);
};

void ErrorLog::addGoal(const Proposition * p, const State * s)
{
  const UnsatCondition * pre = fac->buildUnsatGoal(p,s);
  conditions.push_back(pre);
};

void ErrorLog::addUnsatInvariant(double st, double e, Intervals ints, const Action * a, const State * s,bool rootError)
{
    const UnsatInvariant * inv = fac->buildUnsatInvariant(st,e,ints,a,s,rootError);
    conditions.push_back(inv);  
};

};

