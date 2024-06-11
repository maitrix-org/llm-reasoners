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

  $Date: 2009-02-05 10:50:23 $
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
#include "State.h"
#include "Plan.h"
#include "main.h"
#include "Validator.h"
#include "Proposition.h"
#include "Exceptions.h"
#include "LaTeXSupport.h"
#include "RobustAnalyse.h"

#include <sstream>

using std::stringstream;
using std::cerr;
//#define list std::list
//#define map std::map

namespace VAL {

vector<StateObserver *> State::sos;

void State::setNew(const effect_lists * is)
{
	logState.clear();
	feValue.clear();
	changedPNEs.clear();
	
	for(list<simple_effect*>::const_iterator i = is->add_effects.begin();
		i != is->add_effects.end();++i)
	{
		logState[vld->pf.buildLiteral(*i)] = true;
	};

	for(pc_list<assignment*>::const_iterator i1 = is->assign_effects.begin();
		i1 != is->assign_effects.end();++i1)
	{
		const FuncExp * fe = vld->fef.buildFuncExp((*i1)->getFTerm());
// DPL 2/2/06: Need to remember which PNEs changed in order to trigger
// processes properly in the initial state.
		changedPNEs.insert(fe);
		FEScalar feNewValue = dynamic_cast<const num_expression *>((*i1)->getExpr())->double_value();

		feValue[fe]	= feNewValue;

		//setup initial value for LaTeX graph
		if(LaTeX)
		{
			FEGraph * feg = getValidator()->getGraph(fe);

			//setup initial value if nec
			if(feg->initialTime == -1)
			{
					feg->initialTime = time;
					feg->initialValue = feNewValue;
			};
		};

	};
};
 
State::State(Validator * const v,const effect_lists* is) : 
		tolerance(v->getTolerance()), vld(v), time(0.0)
{
	setNew(is);
};

bool State::evaluate(const SimpleProposition * p) const
{
	LogicalState::const_iterator i = logState.find(p);
	if(i != logState.end())
	{
		return i->second;
	}
	else
	{
		return false;
	};
};

FEScalar State::evaluateFE(const FuncExp * fe) const
{        
	NumericalState::const_iterator i = feValue.find(fe);
	if(i!=feValue.end())
	{    
		return i->second;
	}
	else
	{
	/*	cerr << "Attempt to access undefined expression: " << *fe << "\n";
		if(LaTeX)
		{
			*report << "\\error{Attempt to access undefined expression: " << *fe << "}\n";
		};
		*/
		// Throw? Yes !
		//exit(0);


		BadAccessError bae;
		throw bae;
	};
};

FEScalar State::evaluate(const expression * e,const Environment & bs) const
{
	if(dynamic_cast<const div_expression *>(e))
	{
		return evaluate(dynamic_cast<const div_expression*>(e)->getLHS(),bs) /
				evaluate(dynamic_cast<const div_expression*>(e)->getRHS(),bs);
	};

	if(dynamic_cast<const minus_expression *>(e))
	{
		return evaluate(dynamic_cast<const minus_expression*>(e)->getLHS(),bs) -
				evaluate(dynamic_cast<const minus_expression*>(e)->getRHS(),bs);
	};



	if(dynamic_cast<const mul_expression *>(e))
	{
		return evaluate(dynamic_cast<const mul_expression*>(e)->getLHS(),bs) *
				evaluate(dynamic_cast<const mul_expression*>(e)->getRHS(),bs);
	};

	if(dynamic_cast<const plus_expression *>(e))
	{
		return evaluate(dynamic_cast<const plus_expression*>(e)->getLHS(),bs) +
				evaluate(dynamic_cast<const plus_expression*>(e)->getRHS(),bs);
	};

	if(dynamic_cast<const num_expression*>(e))
	{
		return dynamic_cast<const num_expression*>(e)->double_value();
	};
	
	if(dynamic_cast<const uminus_expression*>(e))
	{
		return -(evaluate(dynamic_cast<const uminus_expression*>(e)->getExpr(),bs));
	};

	if(dynamic_cast<const func_term*>(e))
	{
		const FuncExp * fexp = vld->fef.buildFuncExp(dynamic_cast<const func_term*>(e),bs);
    return fexp->evaluate(this);
		//if(feValue.find(fexp) != feValue.end())
			//return feValue.find(fexp)->second;
   /*
		if(LaTeX)
		{
			*report << "\\error{Attempt to access undefined expression: " << *fexp << "}\n";
		}
		else if(Verbose) cout << "Attempt to inspect undefined value: " << *fexp << "\n";
		 */
		//BadAccessError bae;
		//throw bae;
	};

	if(const special_val_expr * sp = dynamic_cast<const special_val_expr *>(e))
	{
		if(sp->getKind() == E_TOTAL_TIME)
		{
				if(vld->durativePlan()) return time;
				return vld->simpleLength();
		};

		if(sp->getKind() == E_DURATION_VAR)
		{
			return bs.duration;
		};

		if(sp->getKind() == E_HASHT)

		{
			if(LaTeX)
				*report << "The use of \\#t is not valid in this context!\n";
			else if(Verbose)
				cout << "The use of #t is not valid in this context!\n";
			SyntaxTooComplex stc;
			throw stc;
		}

	};

	if(const violation_term * vt = dynamic_cast<const violation_term *>(e))
	{
		return vld->violationsFor(vt->getName());
	};

	UnrecognisedCondition uc;
	throw uc;
};

bool
State::progress(const Happening * h)
{
  DerivedGoal::resetLists(this);
  resetChanged();
   
  if(TestingPNERobustness) JudderPNEs = true;
  bool canHappen = h->canHappen(this);
       
  if(canHappen || ContinueAnyway)
	{
		time = h->getTime();

    if(TestingPNERobustness) JudderPNEs = false;
		return h->applyTo(this) && canHappen;
	}
	else
	{
		return false;
	};
};

bool
State::progressCtsEvent(const Happening * h)
{
  DerivedGoal::resetLists(this);
  resetChanged();
   
  if(TestingPNERobustness) JudderPNEs = true;
  bool canHappen = h->canHappen(this);

  if(canHappen || ContinueAnyway)
	{
		time = h->getTime();
    if(TestingPNERobustness) JudderPNEs = false;
		return h->applyTo(this) && canHappen;
	}
	else
	{
		return false;
	};
};

void
State::add(const SimpleProposition * a)
{
	if(LaTeX)
		*report << " \\> \\adding{"<<*a<<"}\\\\\n";
	else if(Verbose)
		cout << "Adding " << *a << "\n";

		
	logState[a] = true;
};

void
State::del(const SimpleProposition * a)
{
	if(LaTeX)
		*report << " \\> \\deleting{"<<*a<<"}\\\\\n";
	else if(Verbose)
		cout << "Deleting " << *a << "\n";
	logState[a] = false;

};

void
State::addChange(const SimpleProposition * a)
{
	if(LaTeX)
		*report << " \\> \\adding{"<<*a<<"}\\\\\n";
	else if(Verbose)
		cout << "Adding " << *a << "\n";

   if(!(logState[a])) changedLiterals.insert(a);
	logState[a] = true;
};

void
State::delChange(const SimpleProposition * a)
{
	if(LaTeX)
		*report << " \\> \\deleting{"<<*a<<"}\\\\\n";
	else if(Verbose)
		cout << "Deleting " << *a << "\n";

   if(logState[a]) changedLiterals.insert(a);
	logState[a] = false;
};

void
State::updateChange(const FuncExp * fe,assign_op aop,FEScalar value)
{
   FEScalar initialValue = feValue[fe];
     
   update(fe,aop,value);
          
   if(feValue[fe] != initialValue) changedPNEs.insert(fe);

};

State & State::operator=(const State & s)
{
	logState = s.logState;
	feValue = s.feValue;
	time = s.time;
	changedLiterals = s.changedLiterals;
	changedPNEs = s.changedPNEs;

	return *this;
};

void 
State::update(const FuncExp * fe,assign_op aop,FEScalar value)
{
	bool setInitialValue = false;
	FEGraph * feg = 0;
	
	if(LaTeX)
	{
		feg = getValidator()->getGraph(fe);
		
		//setup initial value if nec
		if(feg->initialTime == -1)
		{
			map<const FuncExp*,FEScalar>::const_iterator i = feValue.find(fe);

			
			if(i != feValue.end())
			{
				feg->initialTime = 0;
				feg->initialValue = fe->evaluate(this);
			}
			else
			{
				feg->initialTime = time;
				setInitialValue = true;
					
			};
			
			
		};
		
		
	};


	
	if(Verbose && !LaTeX) *report << "Updating " << *fe << " (" << feValue[fe] << ") by " << value << " ";

	FEScalar feValueInt = feValue[fe];

	switch(aop)
	{
		case E_ASSIGN:
			if(LaTeX)
			{
				*report << " \\> \\assignment{"<<*fe<<"}{"<<feValue[fe]<<"}{"<<value<<"}\\\\\n";
			}
			else if(Verbose) cout << "assignment\n";
			feValue[fe] = value;
			break;
		case E_ASSIGN_CTS:
			if(LaTeX)
			{
				*report << " \\> \\assignmentcts{"<<*fe<<"}{"<<feValue[fe]<<"}{"<<value<<"}\\\\\n";
			}
			else if(Verbose) cout << "assignment\n";
			feValue[fe] = value;
			return;
		case E_INCREASE:
			if(LaTeX)
			{
				*report << " \\> \\increase{"<<*fe<<"}{"<<feValue[fe]<<"}{"<<value<<"}\\\\\n";
			}
			else if(Verbose) cout << "increase\n";
			feValue[fe] += value;
			break;
		case E_DECREASE:
			if(LaTeX)
			{
				*report << " \\> \\decrease{"<<*fe<<"}{"<<feValue[fe]<<"}{"<<value<<"}\\\\\n";
			}
			else if(Verbose) cout << "decrease\n";
			feValue[fe] -= value;
			break;
		case E_SCALE_UP:
			if(LaTeX)
			{
				*report << " \\> \\scaleup{"<<*fe<<"}{"<<feValue[fe]<<"}{"<<value<<"}\\\\\n";
			}
			else if(Verbose) cout << "scale up\n";
			feValue[fe] *= value;
			break;
		case E_SCALE_DOWN:
			if(LaTeX)
			{
				*report << " \\> \\scaledown{"<<*fe<<"}{"<<feValue[fe]<<"}{"<<value<<"}\\\\\n";
			}
			else if(Verbose) cout << "scale down\n";
			feValue[fe] /= value;
			break;
		default:
			return;
	};

	//handle discontinuities in graphs
	if(LaTeX)
	{
		if(setInitialValue)
		{
			feValueInt = feValue[fe];
			feg->initialValue = feValueInt;
		};
		

		if( (feValueInt != feValue[fe]) || setInitialValue )
		{          
			//check value is already defined, may be communitive updates at the same time
			map<double,pair<double,double> >::iterator j = feg->discons.find(time);

			if(j == feg->discons.end())
			{
				feg->discons[time] = make_pair(feValueInt,feValue[fe]);
				feg->happenings.insert(time);
			}
			else
			{
				j->second.second = feValue[fe];
			};
			
		};

	

		
		
	};

	return;
	
};

};



