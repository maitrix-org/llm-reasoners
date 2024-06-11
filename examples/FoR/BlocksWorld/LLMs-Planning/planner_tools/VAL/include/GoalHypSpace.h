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

#ifndef __HYPSP
#define __HYPSP

#include <ostream>

namespace VAL {

class GoalHypothesisSpace {
public:
	virtual ~GoalHypothesisSpace() {};
	virtual void write(std::ostream & o) const
	{
		std::cout << "Goal Hypothesis Space:\n<< >>\n";
	};
};

std::ostream & operator<<(std::ostream & o,const GoalHypothesisSpace & g)
{
	g.write(o);
	return o;
};

}

#endif
