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

  $Date: 2009-02-05 10:50:20 $
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
#include "Ownership.h"
#include "Validator.h"
#include "Action.h"
#include "FuncExp.h"
#include "Environment.h"
#include "Proposition.h"

using std::make_pair;
using std::cerr;

namespace VAL {
  
bool 
Ownership::ownsForAdd(const Action * a,const SimpleProposition * p)
{
  map<const SimpleProposition *,pair<const Action *,ownership> >::iterator po = propOwner.find(p);
   
	if(po != propOwner.end())
	{
		if(po->second.first != a && po->second.second != E_ADD)
		{
			// Different action wants to use the same proposition.
			if(Verbose) 
			{
				if(LaTeX)
				{
					*report << " \\> \\listrow{Mutex violation: \\action{" << *a << "} (adds \\exprn{" << *p << "})";
					if(po->second.first) *report << " and \\exprn{" << po->second.first << "}\n";
					*report << "}\\\\";
				}
				else
				{
					cout << "Mutex violation: " << *a << " (adds " << *p << ")";
					if(po->second.first) cout << " and " << po->second.first << "\n";
				};
				
				*report << "\n";
			};
			if(ErrorReport) vld->getErrorLog().addMutexViolation(vld->getCurrentHappeningTime(),a,po->second.first,&(vld->getState())); 
			return false;
		};

		switch(po->second.second)
		{
			case E_PPRE:
				if(Verbose)
				{
					if(LaTeX)
					{
						*report << " \\> \\listrow{WARNING: \\action{" << *a
							<< "} adds a precondition literal \\exprn{" 
							<< *p << "}}\\\\\n";
					}
					else
					{
						cout << "WARNING: " << *a << " adds a precondition literal " 
							<< *p << "\n";
					};
					
				};
			case E_PRE:
			case E_NPRE:
				po->second.second = E_ADD;
				return true;
			case E_DEL:
				// Action deletes and adds the same literal.
				if(Verbose)
				{
					if(LaTeX)
					{
						*report << " \\> \\listrow{WARNING: \\action{" << *a
							<< "} adds and deletes the literal \\exprn{" 
							<< *p << "}}\\\\\n";
					}
					else
					{
						cout << "WARNING: " << *a << " adds and deletes the literal " 
							<< *p << "\n";
					};
					
				};
				return true;
			case E_ADD:
				// Action adds the same literal twice.
				if(Verbose)
				{
					if(po->second.first == a)
					{
						if(LaTeX)
						{
							*report << " \\> \\listrow{WARNING: \\action{" << *a
						<< "} adds the literal \\exprn{" 
								<< *p << "} twice}\\\\\n";
						}
						else
						{
							cout << "WARNING: " << *a << " adds the literal " 
								<< *p << " twice\n";
						};
					}
					else
					{
						if(LaTeX)
						{
							*report << " \\> \\listrow{WARNING: \\action{" << *a
								<< "} and \\action{" << *(po->second.first)
								<< "} both add the literal \\exprn{" 
								<< *p << "}}\\\\\n";
						}
						else
						{
							cout << "WARNING: " << *a << " and " << *(po->second.first)
							    << " both add the literal " 
								<< *p << "\n";
						};					
					};
				};
				return true;
			default:
        if(ErrorReport) vld->getErrorLog().addMutexViolation(vld->getCurrentHappeningTime(),a,po->second.first,&(vld->getState())); 
				return false;
		};
	}
	else
	{
		propOwner[p] = make_pair(a,E_ADD);
		return true;
	};
};

bool 
Ownership::ownsForDel(const Action * a,const SimpleProposition * p)
{
  map<const SimpleProposition *,pair<const Action *,ownership> >::iterator po = propOwner.find(p);
  
	if(po != propOwner.end())
	{
		if(po->second.first != a && po->second.second != E_DEL)
		{
			// Different action wants to use the same proposition.
			if(Verbose) 
			{
				if(LaTeX)
				{
					*report << " \\> \\listrow{Mutex violation: \\action{" << a
						 << "} (deletes \\exprn{" << *p << "})";
					if(po->second.first) *report << " and \\exprn{" << po->second.first << "}}\\\\";
				}
				else
				{
					cout << "Mutex violation: " << a << " (deletes " << *p << ")";
					if(po->second.first) cout << " and " << po->second.first;
				};
				
				*report << "\n";
			};
      if(ErrorReport) vld->getErrorLog().addMutexViolation(vld->getCurrentHappeningTime(),a,po->second.first,&(vld->getState())); 
			return false;
		};

		switch(po->second.second)
		{
			case E_NPRE:
				if(Verbose)
				{
					if(LaTeX)
					{
						*report << " \\> \\listrow{WARNING: \\action{" << *a
							 << "} deletes a false precondition literal \\exprn{" 
							 << *p << "}}\\\\\n";
					}
					else
					{
						cout << "WARNING: " << *a << " deletes a false precondition literal " 
							<< *p << "\n";
					};
					
					
				};
			case E_PPRE:
			case E_PRE:
				po->second.second = E_DEL;
				return true;
			case E_DEL:
				// Action deletes the same literal twice.
				if(Verbose)
				{
					if(po->second.first == a)
					{
						if(LaTeX)
						{
							*report << " \\> \\listrow{WARNING: \\action{" << *a
								<< "} deletes the literal \\exprn{" 
								<< *p << "} twice}\\\\\n";
						}
						else
						{
							cout << "WARNING: " << *a << " deletes the literal " 
								<< *p << " twice\n";
						};
					}
					else
					{
						if(LaTeX)
						{
							*report << " \\> \\listrow{WARNING: \\action{" << *a
								<< "} and \\action{" << *(po->second.first)
								<< "} both delete the literal \\exprn{" 
								<< *p << "}}\\\\\n";
						}
						else
						{
							cout << "WARNING: " << *a << " and " << *(po->second.first)
							    << " both delete the literal " 
								<< *p << "\n";
						};					
					};
				};
				return true;
			case E_ADD:
				// Action adds and deletes the same literal.
				if(Verbose)
				{
					if(LaTeX)
					{
						*report << " \\> \\listrow{WARNING: \\action{" << *a
							 << "} adds and deletes the literal \\exprn{" 
								<< *p << "}}\\\\\n";
					}
					else
					{
						cout << "WARNING: " << *a << " adds and deletes the literal " 
								<< *p << "\n";
					};
					
					
				};
				return true;
			default:
        if(ErrorReport) vld->getErrorLog().addMutexViolation(vld->getCurrentHappeningTime(),a,po->second.first,&(vld->getState())); 
				return false;
		};
	}
	else
	{
		propOwner[p] = make_pair(a,E_DEL);
		return true;
	};
};

bool Ownership::markOwnedPrecondition(const Action * a,const SimpleProposition * p,ownership o)
{
  map<const SimpleProposition *,pair<const Action *,ownership> >::const_iterator po = propOwner.find(p);
  
	if(po != propOwner.end())
	{
		if(po->second.second == E_PRE || po->second.second == E_NPRE
				|| po->second.second == E_PPRE)
		{
			if(po->second.first != a) propOwner[p] = make_pair(static_cast<const Action *>(0),o);
			return true;
		}
		else
		{
			if(Verbose && po->second.first != a)
			{
				if(LaTeX)
				{
					*report << " \\> \\listrow{Mutex violation: \\action{" << a
						 << "} (requires \\exprn{" << p << "})}\\\\\n";
				}
				else
				{
					cout << "Mutex violation: " << a << " (requires " << p << ")\n";
				};
				
			};
      if((po->second.first != a) && ErrorReport) vld->getErrorLog().addMutexViolation(vld->getCurrentHappeningTime(),a,po->second.first,&(vld->getState())); 
			return po->second.first == a;
		};
	}
	else
	{
		propOwner[p] = make_pair(a,o);
		return true;
	};
};

bool Ownership::markOwnedPreconditionFEs(const Action * a,const expression * e,const Environment & bs)
{	
	if(dynamic_cast<const num_expression *>(e)) return true;
	if(const func_term * fe = dynamic_cast<const func_term *>(e))
	{
		const FuncExp * fexp = vld->fef.buildFuncExp(fe,bs);
		if(FEOwner.find(fexp) != FEOwner.end())
		{
			if(FEOwner[fexp].first == a)
				return true;
				
			if(FEOwner[fexp].second == E_PRE)
			{
				FEOwner[fexp] = make_pair(static_cast<const Action *>(0),E_PRE);
				return true;
			}
			else
			{
				if(LaTeX)
				{
					*report << " \\> \\listrow{Mutex violation: \\action{" << a
						 << "} (requires \\fexprn{" << *fexp << "})}\\\\\n";
			
				}
				else
				{
					if(Verbose) cout << "Mutex violation: " << a << " (requires " << *fexp << ")\n";
				};
        if(ErrorReport) vld->getErrorLog().addMutexViolation(vld->getCurrentHappeningTime(),a,0,&(vld->getState())); 
				return false;
			};
		}
		else
		{
			FEOwner[fexp] = make_pair(a,E_PRE);
			return true;
		};
	};
	if(const binary_expression * bexp = dynamic_cast<const binary_expression *>(e))
	{
		return markOwnedPreconditionFEs(a,bexp->getLHS(),bs) &&
				markOwnedPreconditionFEs(a,bexp->getRHS(),bs);
	};
	if(const uminus_expression * uexp = dynamic_cast<const uminus_expression *>(e))
	{
		return markOwnedPreconditionFEs(a,uexp->getExpr(),bs);
	};
	if(dynamic_cast<const special_val_expr *>(e))
	{
		return true;
	};

	if(Verbose)
	{
			if(LaTeX) *report << " \\> ";
			*report << "Unrecognised expression type\n";
			if(LaTeX) *report << " \\\\";
	};
	
	UnrecognisedCondition uc;
	throw uc;
	
};

bool Ownership::markOwnedEffectFE(const Action * a,const FuncExp * fe,assign_op aop,
								const expression * e,const Environment & bs)
{
	if(!markOwnedPreconditionFEs(a,e,bs))
	{
		if(Verbose)
		{
			if(LaTeX)
			{
				*report << " \\> \\listrow{Conflict over ownership of resource: \\fexprn{"
					 << *fe << "} wanted by \\action{" << a << "}}\\\\\n";
			
			}
			else
			{
				cout << "Conflict over ownership of resource: " << *fe << " wanted by " << a << "\n";
			};
			
			
		};
		if(ErrorReport) vld->getErrorLog().addMutexViolation(vld->getCurrentHappeningTime(),a,0,&(vld->getState())); 
		return false;
	};
	
	if(FEOwner.find(fe) != FEOwner.end())
	{
		if(FEOwner[fe].first != a)
		{
			if(FEOwner[fe].second == E_ADD && (aop == E_INCREASE || aop == E_DECREASE))
			{
				// Commuting additive assignments are OK.
				FEOwner[fe].first = static_cast<const Action *>(0);
				return true;
			};
			
			// Different action wants to use the same FE for conflicting uses.
			if(Verbose) 
			{
				if(LaTeX)
				{
					*report << " \\> \\listrow{Mutex violation: \\action{" << a
						 << "} (expression \\fexprn{" << *fe << "})";
					if(FEOwner[fe].first) *report << " and \\fexprn{" << FEOwner[fe].first << "}";
					*report << "}\\\\";
				}
				else
				{
					cout << "Mutex violation: " << a << " (expression " << *fe << ")";
					if(FEOwner[fe].first) cout << " and " << FEOwner[fe].first;
				};
						
				*report << "\n";
			};
      if(ErrorReport) vld->getErrorLog().addMutexViolation(vld->getCurrentHappeningTime(),a,0,&(vld->getState()));
			return false;
		};

		switch(FEOwner[fe].second)
		{
			case E_PRE:
				if(aop == E_INCREASE || aop == E_DECREASE)
				{
					FEOwner[fe].second = E_ADD;
				}
				else
				{
					FEOwner[fe].second = E_ASSIGNMENT;
				};
				return true;
				
			case E_ADD:
				// Action deletes the same literal twice.
				
				if(aop != E_INCREASE && aop != E_DECREASE)
				{
					if(Verbose)
					{
						if(LaTeX)
						{
							*report << " \\> \\listrow{\\action{"<< *a
							     << "} both assigns to and updates expression \\fexprn{" << *fe << "}}\\\\\n";		
						}
						else
						{
							cout << *a << " both assigns to and updates expression " << *fe << "\n";
						};
						
					};
					if(ErrorReport) vld->getErrorLog().addMutexViolation(vld->getCurrentHappeningTime(),a,0,&(vld->getState()));
					return false;
				};

				if(Verbose)
				{
					if(LaTeX)
					{
						*report << " \\> \\listrow{WARNING: \\action{" << *a
						     << "} assigns to expression \\fexprn{" 
							<< *fe << "} twice}\\\\\n";
					}
					else
					{
						cout << "WARNING: " << *a << " assigns to expression " 
							<< *fe << "twice\n";
					};
					
				};
				return true;
				
			case E_ASSIGNMENT:
				// Action assigns to expression twice.
				if(Verbose)
				{
					if(LaTeX)
					{
						*report << " \\> \\listrow{\\action{"<< *a
							 << "} assigns to expression \\fexprn{" << *fe << "} twice}\\\\\n";
					
					}
					else
					{
						cout << *a << " assigns to expression " << *fe << " twice\n";
					};
					
					
				};
				if(ErrorReport) vld->getErrorLog().addMutexViolation(vld->getCurrentHappeningTime(),a,0,&(vld->getState()));
				return false;
			default:
				if(Verbose)
				{
					if(LaTeX) *report << " \\> \\listrow{";
					*report << "Unknown expression type in " << a << "\n";
					if(LaTeX) *report << "}\\\\";
				};
				
				UnrecognisedCondition uc;
				throw uc;
		};
	}
	else
	{
		FEOwner[fe] = make_pair(a,(aop==E_INCREASE || aop==E_DECREASE)?E_ADD:E_ASSIGNMENT);
		return true;
	};	
};

};
