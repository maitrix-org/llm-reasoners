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

  $Date: 2009-02-05 10:50:14 $
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
#include "FuncExp.h"
#include "State.h"
#include "random.h"
#include "main.h"
#include "RobustAnalyse.h"

//#define map std::map
namespace VAL {
  
double
FuncExp::evaluate(const State * s) const 
{
  double ans = s->evaluateFE(this);
 
  if(JudderPNEs && hasChangedCtsly)
  {
        ans += RobustPNEJudder*(1-2*getRandomNumberUniform()); //if not robustness testing this change will not be activated    
  };
	return ans;
};

string FuncExp::getParameter(int paraNo) const
{          
      int parameterNo = 1;
		for(parameter_symbol_list::const_iterator i = fe->getArgs()->begin();
				i != fe->getArgs()->end();++i)
		{
         if(paraNo == parameterNo)
         {
         			if(dynamic_cast<const var_symbol *>(*i))
         			{
         				return bindings.find(dynamic_cast<const var_symbol *>(*i))->second->getName();
         			}
         			else

         			{
         				return (*i)->getName();

         			};
         };
         ++parameterNo;

		};
      
  return "";
};

bool FuncExp::checkConstantsMatch(const parameter_symbol_list* psl) const
{
  const_symbol * aConst;

  parameter_symbol_list::const_iterator ps = psl->begin();   //from event
  	for(parameter_symbol_list::const_iterator i = fe->getArgs()->begin(); //from func
  				i != fe->getArgs()->end();++i)
  {
     if(dynamic_cast<const const_symbol*>(*ps))
     {

       if(const var_symbol * aVariable = dynamic_cast<const var_symbol *>(*i))
       {
            aConst = const_cast<const_symbol*>(bindings.find(aVariable)->second);
       }
       else
       {
            aConst = const_cast<const_symbol*>(dynamic_cast<const const_symbol*>(*i));
       };

       if(*ps != aConst) return false;
     };

     ++ps;
  };

  return true;
};

ostream & operator <<(ostream & o,const FuncExp & fe) 
{
	fe.write(o);
	return o;
};

void FuncExp::setChangedCtsly()
{ 
 hasChangedCtsly = true;
};

Environment FuncExpFactory::nullEnv;

FuncExpFactory::~FuncExpFactory()
{
	for(map<string,const FuncExp*>::const_iterator i = funcexps.begin();i != funcexps.end();++i)
		delete const_cast<FuncExp*>(i->second);
};

};
