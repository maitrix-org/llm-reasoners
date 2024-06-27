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

#ifndef __GRAPHCONSTRUCT
#define __GRAPHCONSTRUCT

#include "instantiation.h"

#include <vector>
#include <iostream>

using std::vector;
using std::ostream;

namespace VAL {
class State;
class Validator;
};

namespace Inst {

class SpikeEntry {
private:
	int layerEntered;
	
public: 
	void addAt(int l) {layerEntered = l;};
	virtual ~SpikeEntry(){};
	virtual void write(ostream & o) const {};
	int getWhen() const {return layerEntered;};
	virtual void lateEntry() {};
};

inline ostream & operator<<(ostream & o,const SpikeEntry & s)
{
	s.write(o);
	return o;
};

template<typename T>
class Spike {
private:
	vector<T *> spk;
	vector<int> levelheads;
public:
	typedef typename vector<T *>::iterator SpikeIterator;
	int size() const {return spk.size();};
	
	Spike(){};
	T * addEntry(T * spe){
		spk.push_back(spe);
		spe->addAt(levelheads.size());
		return spe;
	}
	void finishedLevel(){
		levelheads.push_back(spk.size());
	}
	int numLevels() const {
		return levelheads.size();
	};
	void write(ostream & o) const
	{
		for(int i = 0;i < numLevels();++i)
		{
			o << "Level " << i << " (";
			for(int j = 0;j < levelheads[i];++j)
			{
				spk[j]->write(o);
				o << ",";
			};
			o << ")\n\n";
		};
	};
	int & lastLevelHead()
	{
		return levelheads[numLevels()-1];
	};
	const int & lastLevelHead() const
	{
		return levelheads[numLevels()-1];
	};
	void insertAbsentee(T * t)
	{
// This inserts t at the previous level, where it was meant to have been added earlier.
		cout << "Adding absentee to level " << lastLevelHead() << "\n";
		int idx = lastLevelHead();
		spk.push_back(spk[idx]);
		spk[idx] = t;
		t->lateEntry();
		++lastLevelHead();
	};
	template<typename U>
	T * find(const U * u) const
	{
		int j = 0;
		for(typename vector<T *>::const_iterator i = spk.begin();j != lastLevelHead();++i,++j)
		{
			if((*i)->represents(u))
			{
				return *i;
			};
		};
		return 0;
	};
	template<typename U>
	T * findInAll(const U * u) const
	{
		for(typename vector<T *>::const_iterator i = spk.begin();i != spk.end();++i)
		{
			if((*i)->represents(u))
			{
				return *i;
			};
		};
		return 0;
	};
	SpikeIterator begin() {return spk.begin();};
	SpikeIterator end() {return spk.end();};
	// This is the address of the first element *not* in level i
	SpikeIterator toLevel(int i) {return spk.begin()+levelheads[i];};
};




class ActEntry;
class FluentEntry;
class PlanGraph;

class PropEntry : public SpikeEntry {
private:
	static int counter;
	const int myID;
	
	Literal * theprop;

	bool initiallyTrue;

	vector<ActEntry *> achievers;
	vector<ActEntry *> deleters;
	
public: 
	PropEntry(Literal * p) : myID(counter++), theprop(p), initiallyTrue(true) {};
	int getID() const {return myID;};
	
	void write(ostream & o) const
	{
		if(!initiallyTrue) o << "*";
		theprop->write(o);
	};
	bool represents(const Literal * lit) const {return theprop==lit;};
	void setInitiallyFalse() {initiallyTrue = false;};
	
	void addAchievedBy(ActEntry * ae)
	{
		achievers.push_back(ae);
	};
	void addDeletedBy(ActEntry * ae)
	{
		deleters.push_back(ae);
	};
	bool gotAchievers() const
	{
		return !achievers.empty() || initiallyTrue;
	};
	bool gotDeleters() const
	{
		return !deleters.empty() || !initiallyTrue;
	};
	void lateEntry()
	{
		initiallyTrue = false;
	};
};

enum ActType {START,END,INV,ATOMIC};

class DurationConstraint;

class DurationHolder {
private:
	map<string,vector<int> > relevantArgs;
	map<string,DurationConstraint *> dursFor;
public:
	void readDurations(const string & nm);
	DurationConstraint * lookUp(const string & nm,instantiatedOp * op);
};

class ActEntry : public SpikeEntry {
private:
	instantiatedOp * theact;

	vector<PropEntry *> achieves;
	vector<PropEntry *> deletes;
	vector<FluentEntry *> updates;

	vector<PropEntry *> supports;
	vector<PropEntry *> negSupports;

// This flag is used for identifying those actions that must be repeatedly applied
// because they have relative metric effects.
	bool iterating;

// If this is part of a durative action structure we want to know about it
	ActType atype;
	DurationConstraint * dur;

	static DurationHolder dursFor;

public:
	static void readDurations(const string & nm)
	{
		dursFor.readDurations(nm);
	};
	bool isActivated(const vector<bool> & actives) const;
	bool isActivated(VAL::Validator * v,const VAL::State *) const;
	bool isRelevant(VAL::Validator * v,const VAL::State *) const;

	ActEntry(instantiatedOp * io);
	instantiatedOp * getIO() {return theact;};
	bool isEvent() const
	{
		const VAL::event * e = dynamic_cast<const VAL::event *>(theact->forOp());
		return (e != 0);
	};
	bool represents(const instantiatedOp * op) const {return op==theact;};

	void addUpdates(FluentEntry * fe)
	{
		updates.push_back(fe);
	};
	void addAchieves(PropEntry * pe)
	{
		achieves.push_back(pe);
	};
	void addDeletes(PropEntry * pe)
	{
		deletes.push_back(pe);
	};
	void addSupportedBy(PropEntry * pe) 
	{
		supports.push_back(pe);
	};
	void addSupportedByNeg(PropEntry * pe)
	{
		negSupports.push_back(pe);
	};
	void write(ostream & o) const;

	bool isIterating() const {return iterating;};
};

class BoundedValue;

class Constraint {
protected:
	BoundedValue * bval;
public:
	Constraint(BoundedValue * b) : bval(b) {};
	virtual ~Constraint();
	virtual void write(ostream & o) const;
};

class DurationConstraint : public Constraint {
private:
	ActEntry * start;
	ActEntry * inv;
	ActEntry * end;
	
public:
	DurationConstraint(BoundedValue * b) : 
		Constraint(b), start(0), inv(0), end(0) {};
	void setStart(ActEntry * s) {start = s;};
	void setEnd(ActEntry * e) {end = e;};
	void setInv(ActEntry * i) {inv = i;};
	virtual void write(ostream & o) const;
};


class InitialValue : public Constraint {
public:
	InitialValue(BoundedValue * b) : Constraint(b) {};
	void write(ostream & o) const;
};

class UpdateValue : public Constraint {
private:
	ActEntry * updater;
	const VAL::expression * exp;
	const VAL::assign_op op;
public:
	UpdateValue(ActEntry * ae,const VAL::expression * e,const VAL::assign_op o,BoundedValue * b) :
		Constraint(b), updater(ae), exp(e), op(o)
	{};
	void write(ostream & o) const;
};



class BoundedValue {
public:
	virtual ~BoundedValue() {};
	virtual void write(ostream & ) const {};
	virtual BoundedValue * operator+=(const BoundedValue *) = 0;
	virtual BoundedValue * operator-=(const BoundedValue *) = 0;
	virtual BoundedValue * operator*=(const BoundedValue *) = 0;
	virtual BoundedValue * operator/=(const BoundedValue *) = 0;
	virtual void negate() {};
	virtual bool gotLB() const {return true;};
	virtual bool gotUB() const {return true;};
	virtual double getLB() const = 0;
	virtual double getUB() const = 0;
	virtual BoundedValue * accum(const BoundedValue * bv) = 0;
	virtual bool contains(double d) const {return false;};

	virtual BoundedValue * infUpper() =0;
	virtual BoundedValue * infLower() =0;
	virtual BoundedValue * copy() const = 0;
};

class BoundedInterval : public BoundedValue {
private:
	bool finitelbnd;
	double lbnd;
	bool finiteubnd;
	double ubnd;
public:
	BoundedInterval(double l,double u) : finitelbnd(true), lbnd(l), 
		finiteubnd(true), ubnd(u) {};
	BoundedValue * infUpper() {finiteubnd = false; return this;};
	BoundedValue * infLower() {finitelbnd = false; return this;};
	void negate() 
	{
		bool x = finitelbnd;
		finitelbnd = finiteubnd;
		finiteubnd = x;
		double y = lbnd;
		lbnd = -ubnd;
		ubnd = -y;
	};

	BoundedValue * accum(const BoundedValue * bv)
	{
		if(!finitelbnd || !bv->gotLB())
		{
			finitelbnd = false;
		}
		else
		{
			lbnd = min(lbnd,bv->getLB());
		};
		if(!finiteubnd || !bv->gotUB())
		{
			finiteubnd = false;
		}
		else
		{
			ubnd = max(ubnd,bv->getUB());
		};
		return this;
	};
	
	void write(ostream & o) const
	{
		if(finitelbnd)
		{
			o << "[" << lbnd << ",";
		}
		else
		{
			o << "(-INF,";
		};
		if(finiteubnd)
		{
			o << ubnd << "]";
		}
		else
		{
			o << "INF)";
		};
	};
	virtual BoundedValue * operator+=(const BoundedValue *);
	virtual BoundedValue * operator-=(const BoundedValue *);
	virtual BoundedValue * operator*=(const BoundedValue *);
	virtual BoundedValue * operator/=(const BoundedValue *);
	bool gotLB() const {return finitelbnd;};
	bool gotUB() const {return finiteubnd;};
	double getLB() const {return lbnd;};
	double getUB() const {return ubnd;};
	bool contains(double d) const
	{
		return (!finitelbnd || lbnd <= d) && (!finiteubnd || ubnd >= d);
	};
	BoundedInterval * copy() const 
	{
		return new BoundedInterval(*this);
	};
};

class PointValue : public BoundedValue {
private:
	double val;
public:
	PointValue(double v) : val(v) {};
	void write(ostream & o) const 
	{
		o << "[" <<  val << "]";
	};
	virtual BoundedValue * operator+=(const BoundedValue *);
	virtual BoundedValue * operator-=(const BoundedValue *);
	virtual BoundedValue * operator*=(const BoundedValue *);
	virtual BoundedValue * operator/=(const BoundedValue *);

	BoundedValue * infUpper() 
	{
		BoundedInterval * bi = new BoundedInterval(val,val);
		bi->infUpper();
		return bi;
	};
	BoundedValue * infLower() 
	{
		BoundedInterval * bi = new BoundedInterval(val,val);
		bi->infLower();
		return bi;
	};
	void negate() {val = -val;};
	double getLB() const {return val;};
	double getUB() const {return val;};
	BoundedValue * accum(const BoundedValue * bv);
	bool contains(double d) const
	{
		return val == d;
	};
	PointValue * copy() const
	{
		return new PointValue(*this);
	};
};


class Undefined : public BoundedValue {
public:
	void write(ostream & o) const 
	{
		o << "UNDEF";
	};
	virtual BoundedValue * operator+=(const BoundedValue *) {return this;};
	virtual BoundedValue * operator-=(const BoundedValue *) {return this;};
	virtual BoundedValue * operator*=(const BoundedValue *) {return this;};
	virtual BoundedValue * operator/=(const BoundedValue *) {return this;};

	BoundedValue * infUpper() {return this;};
	BoundedValue * infLower() {return this;};
	double getLB() const {return 0;};
	double getUB() const {return 0;};
	BoundedValue * accum(const BoundedValue * bv) {return bv->copy();};
	Undefined * copy() const {return new Undefined(*this);};
};

inline ostream & operator<<(ostream & o,const BoundedValue & b)
{
	b.write(o);
	return o;
};

inline ostream & operator<<(ostream & o,const Constraint & b)
{
	b.write(o);
	return o;
};

class FluentEntry : public SpikeEntry {
private:
	 vector<Constraint *> constrs;
	 PNE * thefluent;

	 BoundedValue * bval;
	 BoundedValue * tmpaccum;
	 
public:
	FluentEntry(PNE * pne) : thefluent(pne), bval(new Undefined()), tmpaccum(0) {};
	void addInitial(double d)
	{
		delete bval;
		bval = new PointValue(d);
		constrs.push_back(new InitialValue(bval->copy()));
	};
	void addUpdatedBy(ActEntry * ae,const VAL::expression * expr,const VAL::assign_op op,PlanGraph * pg);
	void write(ostream & o) const;
	BoundedValue * getBV() const {return bval;};
	bool represents(const PNE * pne) const {return pne==thefluent;};
	~FluentEntry() {delete bval;};
	void transferValue();
};




class GraphFactory {
public:
	virtual PropEntry * makePropEntry(Literal * l){return new PropEntry(l);};
	virtual ActEntry * makeActEntry(instantiatedOp * io){return new ActEntry(io);};
	virtual FluentEntry * makeFluentEntry(PNE * pne){return new FluentEntry(pne);};
	virtual ~GraphFactory() {};
};



class PlanGraph {
private: 
	GraphFactory * myFac;

	Spike<PropEntry> props;
	Spike<ActEntry> acts;
	Spike<FluentEntry> fluents;

// Use a list of candidates and filter them
	list<instantiatedOp *> inactive;
	typedef list<instantiatedOp *> InstOps;

	vector<ActEntry *> iteratingActs;

public:
	class BVEvaluator;
	friend class BVEvaluator;

	PlanGraph(GraphFactory * gf); // Constructor can set up initial state.
	~PlanGraph() {delete myFac;};
	bool extendPlanGraph(); // Extends the graph by one level.
	void extendToGoals(); 
	void write(ostream & o) const;

	bool activated(instantiatedOp *);
	void activateEntry(ActEntry *);
	void iterateEntry(ActEntry *);

	BoundedValue * update(BoundedValue * bv,const VAL::expression * exp,const VAL::assign_op op,VAL::FastEnvironment * fenv);

	vector<ActEntry *> applicableActions(VAL::Validator * v,const VAL::State * s);
	vector<ActEntry *> relevantActions(VAL::Validator * v,const VAL::State * s);
};

inline ostream & operator<<(ostream & o,const PlanGraph & pg)
{
	pg.write(o);
	return o;
};















};




























#endif
