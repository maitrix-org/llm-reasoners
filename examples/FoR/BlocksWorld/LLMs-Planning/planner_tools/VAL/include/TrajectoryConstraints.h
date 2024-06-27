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

#ifndef __TRAJECTORYMONITOR
#define __TRAJECTORYMONITOR

#include "ptree.h"
#include "Proposition.h"
#include <set>

namespace VAL {

class State;
class Validator;

class Monitor {
protected:
	Monitor(const Monitor & m) {};
public:
	Monitor() {};

	virtual ~Monitor() {};
	virtual bool violationSerious(const State * s) const {return true;};
	virtual Monitor * copy() const {return new Monitor(*this);};
};

class PreferenceMonitor : public Monitor {
private:
	static int id;
	static std::set<int> done;
	
	int myId;
	mutable Validator * vld;
	string name;
public:
	PreferenceMonitor(Validator * v,const string & nm) : myId(id++), vld(v), name(nm) {};
	PreferenceMonitor * copy() const {return new PreferenceMonitor(*this);};
	bool violationSerious(const State * s) const;
};
	
class MonitorOwner {
private:
	Monitor * mon;
public:
	virtual ~MonitorOwner() 
	{
		delete mon;
	};
	MonitorOwner(Validator * v) : 
		mon(new PreferenceMonitor(v,"anonymous")) {};
	MonitorOwner(Validator * v,const string & n) : 
		mon(new PreferenceMonitor(v,n)) {};
	MonitorOwner() : mon(new Monitor()) {};
	MonitorOwner & operator=(const MonitorOwner & m)
	{
		delete mon;
		mon = m.mon->copy();
		return *this;
	};
	MonitorOwner(const MonitorOwner & m) : mon(m.mon->copy()) {};
	virtual bool violationSerious(const State * s) const {return mon->violationSerious(s);};
	void setPreference(Validator * v,const string & nm)
	{
		delete mon;
		mon = new PreferenceMonitor(v,nm);
	};
	
		
};

template<typename T>
T & passOn(T & t,const MonitorOwner & mo)
{
	static_cast<MonitorOwner &>(t) = mo;
	return t;
};


struct PropMonitor : public MonitorOwner {
	const Proposition * prop;
	PropMonitor(const Proposition * p) : prop(p) {};
	const Proposition & operator*() const {return *prop;};
	const Proposition * operator->() const {return prop;};
};

struct PropositionPair : public MonitorOwner {
	const Proposition * first;
	const Proposition * second;
	PropositionPair(const Proposition * f,const Proposition * s) : first(f), second(s) {};
};

struct Deadlined : public MonitorOwner {
	double first;
	const Proposition * second;
	Deadlined(double d,const Proposition * p) : first(d), second(p) {};
};

struct TriggeredDeadlined : public MonitorOwner {
	const Proposition * first;
	pair<double,const Proposition *> second;
	TriggeredDeadlined(const Proposition * p,const Deadlined & d) :
		first(p), second(make_pair(d.first,d.second))
	{};
};

struct Window : public MonitorOwner {
	pair<double,double> first;
	const Proposition * second;
	Window(const pair<double,double> & pds,const Proposition * p) :
		first(pds), second(p) {};
};
		
typedef vector<PropMonitor> Propositions;
typedef vector<PropositionPair> PropositionPairs;
typedef vector<Deadlined> Deadlines;
typedef vector<TriggeredDeadlined> TriggeredDeadlines;
typedef vector<Window> Windows;

class TrajectoryConstraintsMonitor {
private:
	bool isActive;


// Interactions with continuous effects are much harder - ignore that for the 
// moment!
// What about derived predicates?

// These are the easy ones - just check in final state
	Propositions atEnd;
// Not too bad - must check in every state
	Propositions always;
// Check in every state - once satisfied, remove. Must be empty in final state.
	Propositions sometime;
// Check in every state. If satisfied then move to currently.
	Propositions atMostOnce;
// Check every state - if they become false then move to never.
	Propositions currently;
// Check in every state - if true then problem.
	Propositions never;
// Check trigger in every state. If it becomes true then check requirement. If it is
// false then add it to sometime.
	PropositionPairs sometimeAfter;
// This is the hard one. Perhaps the best way to tackle it is to check 
// in every state and remove the pair if the requirement becomes true, but
// return false if the trigger becomes true first.
	PropositionPairs sometimeBefore;
// This one is not too bad: check each state - if the timestamp is too late then 
// return false, but if the trigger becomes true then remove the requirement.
	Deadlines within;
// If the state has a time after the timestamp then the requirement can be shifted 
// into the always (and must hold in the current state).
	Deadlines holdAfter;
// Check the trigger in every state. If true then set up a within goal.
	TriggeredDeadlines alwaysWithin;
// From start time to end time, the requirement must hold.
	Windows holdDuring;

	Propositions allProps;

	class CollectProps;
	friend class CollectProps;

	Validator * vld;
public: 
	TrajectoryConstraintsMonitor(Validator * v,con_goal * cg1,con_goal * cg2);

	
	~TrajectoryConstraintsMonitor();

	bool checkAtState(const State & s);
	bool checkFinalState(const State & s);

};

};

#endif
