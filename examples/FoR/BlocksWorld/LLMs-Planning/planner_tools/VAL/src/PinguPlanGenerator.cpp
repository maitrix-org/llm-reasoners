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

#include "PinguPlanGenerator.h"
#include <fstream>
#include <algorithm>

using namespace std;

namespace VAL {


  PinguPlanGen::PinguPlanGen(char * name) : VisitController(), lastAt("theStart"), count(0), pingu(0), lastMoved(0), mustDelay(true)
{
  ifstream mapper(name);
  string s;
  float x,y;
  while(true)
    {
      s = "";
      mapper >> s >> x >> y;
      if(s == "")
	break;
      position[s] = make_pair(x,y);
    }
};

void PinguPlanGen::visit_simple_effect(simple_effect * se)
{
  string s = se->prop->head->getName();
	if(s =="canbridge")
	  {
//	    cout << "Got a bridge " << "\n";
	  }
	else if(s=="canjump")
	  {
//	    cout << "Got a jump\n";
	  }
	else if(s=="bounceloc")
	  {
//	    cout << "Got a bounceloc\n";
	    string s1 = (*(se->prop->args->begin()))->getName();
//	    cout << s1 << "\n";
	    string s2 = (*(++(se->prop->args->begin())))->getName();
	    bounceLocs[s1] = s2;
	    bounceLocs[s2] = s1;
	  }
	 else if(s == "connected" || s == "canfall" || s == "canjump")
	 {
	 	string s1 = (*(se->prop->args->begin()))->getName();
//	    cout << s1 << "\n";
	    string s2 = (*(++(se->prop->args->begin())))->getName();
	    path[s1] = s2;
	 }
};

void PinguPlanGen::doThis(string a,string p)
{
  PinguPosition pp = getPosition(p);
    ++count;
    cout << "(action (name " << a << "er) (position " << pp.first << " " <<
    pp.second << " 0) (state blocker) (time " << count << "))\n";
    mustDelay = true;
};

void PinguPlanGen::visit_plan_step(plan_step * ps)
{
  string s = ps->op_sym->getName();
  if(s == "fallfromstart")
    {
      whoIsAt[(*(++(ps->params->begin())))->getName()] = ++pingu;
      //     cout << "Got a starter " << pingu << " at " << (*(++(ps->params->begin())))->getName() << "\n";
      lastMoved = pingu;
      lastActionTime[pingu] = 0;
	return;
    }
  if(s == "unblock_by_bombing")
    {
      string pos = (*(ps->params->begin()))->getName();
      doThis("bomb",pos);
      blocked.erase(pos);
    }
  else
    if(s == "unblock_by_jumping")
      {
      string pos = (*(ps->params->begin()))->getName();
      doThis("jump",pos);
      blocked.erase(pos);
      }
  else
  if(s == "jump" || s == "block" || s == "float" || s == "bomb" || s == "climb")
    {
      doAction(s,ps);
    }
  else if(s == "bridgepotentiallink" || s == "bridge")
    {
      doAction("bridg",ps);
    }
  else if(s == "mine-1way" || s == "mine-2way")
    {
      doAction("min",ps);
    }
   else if(s == "bash-1way" || s == "bash-2way")
    {
      doAction("bash",ps);
    }
   else if(s == "walk" || s == "fall")
   	{
   	  	string pos = (*(ps->params->begin()))->getName();
		  string sec = (*(++(ps->params->begin())))->getName();
		  whoIsAt[sec] = whoIsAt[pos];
		  if(whoIsAt[pos] == lastMoved)
		    {
		      //	      cout << lastMoved << " moved again ";
		    }
		  else
		    {
		      lastMoved = whoIsAt[pos];
		      //cout << lastMoved << " moved this time ";
		    };
		  if(blocked.find(pos) != blocked.end())
		    {
		      cout << "NEED TO UNBLOCK " << pos << "\n";
		    }
		  if(position.find(pos) != position.end() && position.find(sec) != position.end())
		  {
		  	
		  	int di = (position[sec].first > position[pos].first);
		  	if(recordDirection.find(pos) == recordDirection.end())
		  	{
		  		recordDirection[pos] = di;
		  	}
		  	if(recordDirection.find(sec) == recordDirection.end())
		  	{
		  		recordDirection[sec] = di;
		  	}
		  }
		  lastAt = sec;
	}
  else if(s == "bounce")
    {
   	  	string pos = (*(ps->params->begin()))->getName();
		  string sec = (*(++(ps->params->begin())))->getName();
		   whoIsAt[sec] = whoIsAt[pos];
		  if(whoIsAt[pos] == lastMoved)
		    {
		      //	      cout << lastMoved << " moved again ";
		    }
		  else
		    {
		      lastMoved = whoIsAt[pos];
		      //cout << lastMoved << " moved this time ";
		    };
		  if(position.find(pos) != position.end() && position.find(sec) != position.end())
		  {
		  	
		  	if(recordDirection.find(sec) == recordDirection.end())
		  	{
		  		recordDirection[sec] = 1-recordDirection[pos];
		  	}
		  }
		  lastAt = sec;
    }
};

PinguPosition PinguPlanGen::getPosition(string pos)
{
	if(position.find(pos) != position.end())
	{
		pair<float,float> p = position[pos];
		return PinguPosition(p.first,p.second,findDirection(pos));
	}
	else
	{
		if(bounceLocs.find(pos) != bounceLocs.end())
		{
			string pos1 = bounceLocs[pos];
			pair<float,float> p = position[pos1];
			position[pos] = p;
			int di = findDirection(pos1);
			di = 1-di;
			recordDirection[pos] = di;
			return PinguPosition(p.first,p.second,di);
		}
		else
		{
			cout << "Unsure about location " << pos << "\n";
		}
	}
	return PinguPosition(0,0,0);
}

void PinguPlanGen::doAction(string s, plan_step * ps)
{
  bool haveProblem = false;
  string pos = (*(ps->params->begin()))->getName();
  string sec = (*(++(ps->params->begin())))->getName();
  if(lastActWas[pos] == s && lastActAt[pos] == count)
    {
      //      cout << "Errr - we've got a problem\n";
      haveProblem = true;
    }
  cout << "(action (name " << s << "er) (position ";

  if(s != "block")
    {
      whoIsAt[sec] = whoIsAt[pos];
    }
      if(whoIsAt[pos] == lastMoved)
		    {
		      // cout << lastMoved << " moved again (last moved at " <<
		      //lastActionTime[lastMoved] << ") ";
		    }
		  else
		    {
		      lastMoved = whoIsAt[pos];
		      //	      cout << lastMoved << " moved this time (last moved at "
		      // << lastActionTime[lastMoved] << ") ";
		    };
   

  if(position.find(pos) != position.end() && position.find(sec) != position.end())
  {
  	
  	int di = (position[sec].first > position[pos].first);
  	if(recordDirection.find(pos) == recordDirection.end())
  	{
  		recordDirection[pos] = di;
  	}
  	if(recordDirection.find(sec) == recordDirection.end())
  	{
  		recordDirection[sec] = di;
  	}
  }
  if(s == "bridg")
    {
       if(!mustDelay && lastActionTime[lastMoved] < count) 
	{
	  if(lastActAt[pos] == count) 
	    {
	      // cout << "Ooops - better delay...";
	      }
	  else
	    --count;
	}
      doBridge(pos,sec);
      mustDelay = true;
    }
  else if(s == "bomb")
    { 
      if(!mustDelay && lastActionTime[lastMoved] < count) 
	{
	  if(lastActAt[pos] == count) 
	    {
	      // cout << "Ooops - better delay...";
	      }
	  else
	    --count;
	}
      doBomb(ps,pos,sec);
      mustDelay = true;
    }
  else if(s == "min")
    {
       if(!mustDelay && lastActionTime[lastMoved] < count) 
	{
	  if(lastActAt[pos] == count) 
	    {
	      // cout << "Ooops - better delay...";
	      }
	  else
	    --count;
	}
      doMine(pos,sec);
      mustDelay = true;
    }
    else if(s == "bash")
    {
       if(!mustDelay && lastActionTime[lastMoved] < count) 
	{
	  if(lastActAt[pos] == count) 
	    {
	      // cout << "Ooops - better delay...";
	      }
	  else
	    --count;
	}
      doBash(pos,sec);
      mustDelay = false;
    }
  else 
    {
      PinguPosition p = getPosition(pos);
	if(blocked.find(pos) != blocked.end())
	  {
	    p.first += (p.di?-17:17);
	  }
      if(haveProblem)
      {
	  p.first += (p.di?-7:7);
	}

      cout << ((int) p.first) << " " << ((int) p.second) << " 0) ";
      if(bounceLocs.find(pos) != bounceLocs.end())
      {
      	cout << "(direction " << (p.di?"right":"left") << ") ";
      	if(blocked.find(pos) != blocked.end())
      	{
      		cout << "(state walker) ";
      	}
      }
      if(s == "block")
      {
      	blocked.insert(pos);
      }
      if(!mustDelay && lastActionTime[lastMoved] < count) 
	{
	  if(lastActAt[pos] == count) 
	    {
	      // cout << "Ooops - better delay...";
	      }
	  else
	    --count;
	}
      mustDelay = false;
      if(s == "climb") mustDelay = true;
    }
  ++count;
  cout << " (time " << count << "))\n";
  lastActionTime[lastMoved] = count;
  lastActAt[pos] = count;
  lastActWas[pos] = s;
 lastAt = sec;
};

int PinguPlanGen::findDirection(string s)
{
	if(recordDirection.find(s) != recordDirection.end())
	{
		return recordDirection[s];
	}
	else
	{
		return recordDirection[lastAt];
	}
}

void PinguPlanGen::doMine(string pos,string sec)
{
  //	cout << "Hmm - what do we do now\n";
  //	cout << "Trust starting location and depth\n";
	PinguPosition p = getPosition(pos);
  //	cout << "At " << pos << " facing " << (p.di?"right":"left") << "\n";
	recordDirection[sec] = p.di;
	  if(bounceLocs.find(pos) != bounceLocs.end())
	  {
	    string s = bounceLocs[pos];
	    if(blocked.find(s) != blocked.end())
	      {
		p.first += (p.di?17:-17);
	      }
	  }
	PinguPosition q = getPosition(sec);
	float depth = q.second - p.second;
	depth *= 7.0/8.0; // Fiddle factor - not sure if this works generally!
	//	cout << "Propose placing " << sec << " at " << p.first + (p.di?depth:-depth) <<
	//		"," << q.second << " instead of " << q.first << "," << q.second << "\n";
	position[sec] = make_pair(p.first+(p.di?depth:-depth),q.second);
	cout << p.first << " " << p.second << " 0) (direction " << (p.di?"right":"left") << ") ";
	
}

void PinguPlanGen::doBash(string pos,string sec)
{
  //	cout << "Hmm - what do we do now\n";
	PinguPosition p = getPosition(pos);
  //	cout << "At " << pos << " facing " << (p.di?"right":"left") << "\n";
	recordDirection[sec] = p.di;
	  if(bounceLocs.find(pos) != bounceLocs.end())
	  {
	    string s = bounceLocs[pos];
	    if(blocked.find(s) != blocked.end())
	      {
		p.first += (p.di?17:-17);
	      }
	  }
	  cout << p.first << " " << p.second << " 0) ";
}

void PinguPlanGen::doBridge(string pos,string sec) 
{
	PinguPosition p = getPosition(pos);
	pair<float,float> p1 = position[pos];
	if(bounceLocs.find(pos) != bounceLocs.end())
	  {
	    string s = bounceLocs[pos];
	    if(blocked.find(s) != blocked.end())
	      {
		p.first += (p.di?17:-17);
		p1.first = p.first;
	      }
	  }
	if(p.di == 0)
	{
		p.first -= 60;
	}
	else
	{
		p.first += 60;
	}
	p.second -= 30;
	position[sec] = make_pair(p.first,p.second);
	recordDirection[sec] = p.di;
	
	cout << ((int) p1.first) << " " << ((int) p1.second) << " 0) ";
	if(bounceLocs.find(pos) != bounceLocs.end())
	  {
	  	cout << "(direction " << (p.di?"right":"left") << ") ";
	  }
	midairLocs.insert(sec);
	if(midairLocs.find(pos) != midairLocs.end())
	{
		cout << "(state waiter) ";
	}
	lastAt = sec;
};

void PinguPlanGen::doBomb(plan_step * ps,string pos,string sec) 
{
  if(position.find(sec) != position.end())
    {
      PinguPosition p = getPosition(sec);
      p.first += (p.di?-140:140);
      cout << ((int) p.first) << " " << ((int) p.second) << " 0) ";      
    }
  else 
    {
      cout << "NOT SURE WHERE TO DO THIS! ";
    }
};

}
