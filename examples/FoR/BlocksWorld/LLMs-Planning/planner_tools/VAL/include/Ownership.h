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
#ifndef __OWNERSHIP
#define __OWNERSHIP

#include <map>
#include "ptree.h"

namespace VAL {

class Validator;
class Action;
class FuncExp;
struct Environment;
class SimpleProposition;
class expression;

using std::map;
using std::pair;
  
enum ownership {E_PRE,E_PPRE,E_NPRE,E_ADD,E_DEL,E_ASSIGNMENT};

class Ownership {
private:
	map<const SimpleProposition *,pair<const Action *,ownership> > propOwner;

	Validator * vld;
	map<const FuncExp *,pair<const Action *,ownership> > FEOwner;

public:
	Ownership(Validator * v) : vld(v) {};

	bool markOwnedPrecondition(const Action * a,const SimpleProposition * p,ownership o);
	bool markOwnedPreconditionFEs(const Action * a,const expression * e,const Environment & bs);
	bool ownsForAdd(const Action * a,const SimpleProposition * p);
	bool ownsForDel(const Action * a,const SimpleProposition * p);
	bool markOwnedEffectFE(const Action * a,const FuncExp * fe,assign_op aop,
								const expression * e,const Environment & bs);
	
};

};

#endif
