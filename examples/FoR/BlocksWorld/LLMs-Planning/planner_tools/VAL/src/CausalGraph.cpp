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

#include "CausalGraph.h"
#include "ptree.h"
#include "VisitController.h"
#include "TimSupport.h"
#include "ToFunction.h"
#include "SASActions.h"
#include <set>

using namespace std;
using namespace VAL;

namespace SAS {



CausalGraph::CausalGraph()
{
	fs.normalise();
	fs.initialise();
	fs.processActions();

	for(FunctionStructure::iterator i = fs.begin();i != fs.end();++i)
	{
		cout << *(i->second);
		set<const ValueRep *> pres;
		set<const ValueRep *> posts;
		for(SASActionTemplate::iterator j = i->second->precondsBegin();j != i->second->precondsEnd();++j)
		{
			pres.insert(j->second.begin(),j->second.end());
		};
		for(SASActionTemplate::iterator j = i->second->postcondsBegin();j != i->second->postcondsEnd();++j)
		{
			posts.insert(j->second.begin(),j->second.end());
		};
		for(set<const ValueRep *>::iterator j = posts.begin();j != posts.end();++j)
		{
			for(set<const ValueRep *>::iterator k = pres.begin();k != pres.end();++k)
			{
				add(Var((*j)->getType(),(*j)->getSegment()),Var((*k)->getType(),(*k)->getSegment()));
			};
		};
	};
};

void CausalGraph::add(Var e,Var p)
{
	if(e != p)
	{
		dependencies[e].insert(p);
		dependents[p].insert(e);
	};
};

void CausalGraph::write(ostream & o) const
{
	for(Graph::const_iterator i = dependencies.begin();i != dependencies.end();++i)
	{
		o << i->first.first->getName() << "_" << i->first.second << ":\n";
		for(Vars::iterator j = i->second.begin(); j != i->second.end();++j)
		{
			o << "\t" << j->first->getName() << "_" << j->second << "\n";
		};
	};
};


}
