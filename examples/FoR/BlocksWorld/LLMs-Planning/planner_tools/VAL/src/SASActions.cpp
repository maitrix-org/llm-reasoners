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

#include "SASActions.h"
#include "ToFunction.h"
#include "instantiation.h"

using namespace TIM;
using namespace Inst;

namespace SAS {

PreMap SASActionTemplate::preMap;
map<const pred_symbol *,vector<SASActionTemplate *> > SASActionTemplate::otherprecs;


bool SegmentRep::growOneLevel(const pddl_type * pt,const TIMobjectSymbol * tob,
										FunctionStructure * fs)
{
// Need to be careful here: the valuereps.size will change as new actions are added 
// and we don't want to include them at the same level they are activated by an action.

	int last = levelcounts[levelcounts.size()-2];
	int sz = levelcounts[levelcounts.size()-1];
	bool activated = false;
//	levelcounts.push_back(sz);

	for(int i = last;i < sz;++i)
	{
//		cout << "Handling a value: " << pt->getName() << " " << tob->getName() << " " <<
//				valuereps[i]->getSegment() << " " << *(valuereps[i]->getValue()) << "\n";
		int seg = valuereps[i]->getSegment();
		for(vector<pair<const var_symbol *,SASActionTemplate *> >::iterator j =
					SASActionTemplate::findOps(pt,seg).begin();
					j != SASActionTemplate::findOps(pt,seg).end();++j)
		{
//		cout << j->second->getOp()->name->getName() << "\n";
			int k = fs->startFor(j->second->getOp());
			int ke = fs->endFor(j->second->getOp());
			for(OpStore::iterator iop = instantiatedOp::from(k);
					k != ke;++k,++iop)
			{
//				cout << "Considering " << *instantiatedOp::getInstOp(k) << "\n";
				if((*(*iop)->getEnv())[j->first] != tob)
				{
//					cout << "Irrelevant\n";
					continue;
				};
				activated |= fs->tryMatchedPre(k,*iop,
												j->first,j->second,valuereps[i]);
			};
		};
	};
	return activated;
};

bool ValueRep::matches(ValueRep * vrep,FastEnvironment * fenv)
{
	if(segment != vrep->segment) return false;
	return vel->matches(vrep->vel,fenv);

};

void SASActionTemplate::enact(FastEnvironment * fenv,Reachables & reachables,
								vector<proposition*> & others)
{
	for(VMap::iterator i = postconditions.begin();i != postconditions.end();++i)
	{
		RangeRep * rr = reachables[i->first->type][TOB((*fenv)[i->first])];
		for(vector<ValueRep *>::iterator j = i->second.begin();j != i->second.end();++j)
		{
			rr->add(*j,fenv);
		};
	};
	for(vector<proposition *>::iterator i = otherposts.begin();i != otherposts.end();++i)
	{
		vector<proposition *>::iterator j = others.begin();
		for(;j != others.end();++j)
		{
			if((*i)->head == (*j)->head)
			{
				break;
			};
		};
		if(j == others.end())
		{
			others.push_back(*i);
		};
	};
};

bool SASActionTemplate::checkPre(FunctionStructure * fs,FastEnvironment * fenv,
										const var_symbol * v,ValueRep * vrep)
{
//	cout << "Checking " << *this << "\n";
	for(vector<ValueRep*>::iterator i = preconditions[v].begin();
						i != preconditions[v].end();++i)
	{
		if(vrep->matches(*i,fenv)) return true;
	};
	return false;
};

ValueRep::ValueRep(ValueRep * vr,FastEnvironment * fenv) : 
	segment(vr->segment), vel(new ValueElement(vr->vel,fenv))
{};

};

