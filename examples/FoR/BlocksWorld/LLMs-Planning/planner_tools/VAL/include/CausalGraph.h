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

#ifndef __CGA
#define __CGA

#include <map>
#include <set>

#include "ToFunction.h"

namespace SAS {

class CausalGraph {
public:
	typedef pair<const VAL::pddl_type *,int> Var;
	typedef std::set<Var> Vars;
	typedef std::map<Var,Vars> Graph;

private:
	FunctionStructure fs;

	Graph dependencies;
	Graph dependents;
	
public:
	CausalGraph();
	const Vars & getDependencies(Var p)
	{
		return dependencies[p];
	};
	const Vars & getDependents(Var p)
	{
		return dependents[p];
	};
	void add(Var,Var);
	void write(std::ostream & o) const;
};

inline std::ostream & operator<<(std::ostream & o,const CausalGraph & cg)
{
	cg.write(o);
	return o;
};

}

#endif
