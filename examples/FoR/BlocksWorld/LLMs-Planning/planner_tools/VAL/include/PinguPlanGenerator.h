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

#ifndef __PINGUPLANGEN
#define __PINGUPLANGEN

#include "ptree.h"
#include "VisitController.h"
#include <iostream>
#include "TypedAnalyser.h"
#include "TimSupport.h"
#include <map>
#include <string>
#include <set>

using std::string;
using std::pair;
using std::map;
using std::cout;
using std::set;
using namespace TIM;

namespace VAL {

struct PinguPosition {
	float first;
	float second;
	int di;

	PinguPosition(float f1,float f2,int d) : first(f1), second(f2), di(d) {};
};

 struct PinguAction {
   string name;
   int x;
   int y;
   
   PinguAction(string n,int x1,int y1) : name(n), x(x1), y(y1) {};
 };

class PinguPlanGen : public VisitController {
private:
  map<string,pair<float,float> > position;
  set<string> midairLocs;
  map<string,string> bounceLocs;
  map<string,int> recordDirection;
  set<string> blocked;
  string lastAt;
  int count;

  map<int,int> lastActionTime;
  map<string,int> whoIsAt;
  map<string,int> lastActAt;
  map<string,string> lastActWas;
  map<string,string> path;
  int pingu;
  int lastMoved;
  bool mustDelay;

  void doAction(string,plan_step *);
  void doBomb(plan_step *,string,string);
  void doBridge(string,string);
  void doMine(string,string);
  void doBash(string,string);
  int findDirection(string);
  PinguPosition getPosition(string);
  void doThis(string,string);
  
public:
	PinguPlanGen(char * name);

	virtual void visit_effect_lists(effect_lists * el)
	{
		for(pc_list<simple_effect *>::iterator i = el->add_effects.begin();
				i != el->add_effects.end();++i)
		{
			(*i)->visit(this);
		};

		for(pc_list<assignment *>::iterator i = el->assign_effects.begin();
				i != el->assign_effects.end();++i)
		{
			(*i)->visit(this);
		};

		for(pc_list<timed_effect *>::iterator i = el->timed_effects.begin();
				i != el->timed_effects.end();++i)
		{
			(*i)->visit(this);
		};
	};

	virtual void visit_simple_effect(simple_effect * se);
	
	virtual void visit_plan_step(plan_step * p);
};

};

#endif
