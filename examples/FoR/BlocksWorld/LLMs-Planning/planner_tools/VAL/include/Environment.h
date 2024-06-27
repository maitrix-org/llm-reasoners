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

  $Date: 2009-02-05 10:50:12 $
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
#include <map>
#include <vector>

namespace VAL {
  
class var_symbol;
class const_symbol;
class Validator;

};

//#undef vector
//#undef map

using std::map;
using std::vector;

#ifndef __MYENVIRONMENT
#define __MYENVIRONMENT

  
//#define vector std::vector

namespace VAL {

template<class T> bool operator != (T & t1,T & t2) {return ! (t1==t2);};

struct Environment : public map<const var_symbol*,const const_symbol*> {
	static map<Validator*,vector<Environment *> > copies;

	double duration;
	
	Environment * copy(Validator * v) const
	{
		Environment * e = new Environment(*this);
		copies[v].push_back(e);
		//cout << "Copy of "<<this<<" to "<<e<<"\\\\\n";
		return e;
	};

	static void collect(Validator * v)

	{
		for(vector<Environment *>::iterator i = copies[v].begin();i != copies[v].end();++i)
			delete *i;
		copies[v].clear();
		
	  //cout << "Deleting the copies of enviroments here!\\\\\n";
	};
};

template<class TI>
struct EnvironmentParameterIterator {
	Environment * env;
	TI pi;

	EnvironmentParameterIterator(Environment * f,TI p) :
		env(f), pi(p) {};

// Having to cast the const is not good...currently we are forced to do it in order
// to interact with Cascader, but should look at fixing it.
	const_symbol * operator*()
	{
		if(const_symbol * s = const_cast<const_symbol *>(dynamic_cast<const const_symbol *>(*pi)))
		{
			return s;
		};
		return const_cast<const_symbol*>((*env)[dynamic_cast<const var_symbol *>(*pi)]);
	};

	EnvironmentParameterIterator & operator++()
	{
		++pi;
		return *this;
	};

	bool operator==(const EnvironmentParameterIterator<TI> & li) const
	{
		return pi==li.pi;
	};

	bool operator!=(const EnvironmentParameterIterator<TI> & li) const
	{
		return pi!=li.pi;
	};
};

template<class TI>
EnvironmentParameterIterator<TI> makeIterator(Environment * f,TI p)
{
	return EnvironmentParameterIterator<TI>(f,p);
};


};

#endif
