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

#include "HowAnalyser.h"
#include "AbstractGraph.h"

namespace VAL {

HowAnalyser::HowAnalyser() : VisitController(), ag(new AbstractGraph())
{};

void HowAnalyser::visit_simple_effect(simple_effect * se)
{
	extended_pred_symbol * e = EPS(se->prop->head);
	if(epss.find(e) == epss.end())
	{
		ag->addInitialFact(e);
		epss.insert(e);
	};
};

void HowAnalyser::visit_action(action * a)
{
	cout << "Action: " << *(a->name) << "(" << *acts[a] << ")\n";
	acts[a]->analyse(a);
	ag->addAction(acts[a]);
};

void HowAnalyser::completeGraph()
{
	ag->develop();
};

};
