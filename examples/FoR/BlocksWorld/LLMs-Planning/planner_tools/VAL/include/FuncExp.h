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
#include <iostream>
#include <stdlib.h>

#include "ptree.h"
#include "Environment.h"
#include "Utils.h"

#ifndef __FUNCEXP
#define __FUNCEXP

//#define map std::map
using std::map;

namespace VAL {
  
class State;
extern bool LaTeX;


class FuncExp {
private:
	const Environment & bindings;
	const func_term * fe;

  bool hasChangedCtsly; //for testing robustness w.r.t. numerical accuracy
public:
	FuncExp(const func_term * f,const Environment &bs) :
		bindings(bs), fe(f), hasChangedCtsly(false)
	{};

	 double evaluate(const State * s) const;
   string getName() const {return fe->getFunction()->getName();};
   string getParameter(int paraNo) const;
   bool checkConstantsMatch(const parameter_symbol_list* psl) const;
   void setChangedCtsly();
  
   
  	void write(ostream & o) const
	{
      string st = "(" + fe->getFunction()->getName();
		for(parameter_symbol_list::const_iterator i = fe->getArgs()->begin();
				i != fe->getArgs()->end();++i)
		{
			if(dynamic_cast<const var_symbol *>(*i))
			{
				st += " " + bindings.find(dynamic_cast<const var_symbol *>(*i))->second->getName();
			}
			else
			{
				st += " " + (*i)->getName();
			};
		};
		st += ")";

      if(LaTeX)  latexString(st);
      o << st;
	};	
};


ostream & operator <<(ostream & o,const FuncExp & fe);

class FuncExpFactory {
private:
	static Environment nullEnv;
	map<string,const FuncExp *> funcexps;
public:
	const FuncExp * buildFuncExp(const func_term * f)
	{
		string s(f->getFunction()->getName());
    
		for(parameter_symbol_list::const_iterator i = f->getArgs()->begin();
					i != f->getArgs()->end();++i)
		{
			s += (*i)->getName();
		};
		map<string,const FuncExp*>::const_iterator i1 = funcexps.find(s);
		if(i1 != funcexps.end())
			return i1->second;
		const FuncExp * p = funcexps[s] = new FuncExp(f,nullEnv);
		return p;
	};
	const FuncExp * buildFuncExp(const func_term * f,const Environment & bs)
	{
		string s(f->getFunction()->getName());
		for(parameter_symbol_list::const_iterator i = f->getArgs()->begin();
					i != f->getArgs()->end();++i)
		{
			if(dynamic_cast<const var_symbol*>(*i))
			{
				map<const var_symbol*,const const_symbol*>::const_iterator j = bs.find(dynamic_cast<const var_symbol*>(*i));
				if(j != bs.end())
				{
					s += j->second->getName();
				}
				else
				{
					cout << "Error: could not find parameter "<<dynamic_cast<const var_symbol*>(*i)->getName()<<"\n";
					exit(-1);
				};
			}
			else
			{
				s += (*i)->getName();
			};
		};
		map<string,const FuncExp*>::const_iterator i1 = funcexps.find(s);
		if(i1 != funcexps.end())
			return i1->second;
		const FuncExp * p = funcexps[s] = new FuncExp(f,bs);
		return p;
	};
 
	~FuncExpFactory();
};	

};



#endif
