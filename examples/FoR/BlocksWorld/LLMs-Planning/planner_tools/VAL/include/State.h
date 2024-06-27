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

  $Date: 2009-02-05 10:50:23 $
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
#include "Proposition.h"
#include "FuncExp.h"
#include "StateObserver.h"
#include<set>
using std::set;

namespace VAL {
class Validator;
class Happening;
};

#ifndef __STATE
#define __STATE

namespace VAL {
  
typedef long double FEScalar;

typedef map<const SimpleProposition*,bool> LogicalState;
typedef map<const FuncExp*,FEScalar> NumericalState;


class State {
private:
	const double tolerance;

	Validator * const vld;

	LogicalState logState;
	NumericalState feValue;
	
	double time;

   	//record which literals and PNEs have changed by appliaction of an happening (for triggering events)
   	set<const SimpleProposition *> changedLiterals;
   	set<const FuncExp *> changedPNEs;

	// Record which actions changed things.
   	map<const SimpleProposition *,set<const Action*> > responsibleForProps;
   	map<const FuncExp *,set<const Action*> > responsibleForPNEs;
   	
   	FEScalar evaluateFE(const FuncExp * fe) const;

   	static vector<StateObserver *> sos;
   
public:
	State(Validator * const v,const effect_lists* is);
	State & operator=(const State & s);

  friend class FuncExp;
	const double getTolerance() const {return tolerance;};
	Validator * getValidator() const {return vld;};
	double getTime() const {return time;};
	
	bool progress(const Happening * h);
  bool progressCtsEvent(const Happening * h);
	bool evaluate(const SimpleProposition * p) const;
	FEScalar evaluate(const FuncExp * fe) const;
	FEScalar evaluate(const expression * e,const Environment & bs) const;

	void add(const SimpleProposition *);
	void del(const SimpleProposition *);
	void update(const FuncExp * fe,assign_op aop,FEScalar value);

	const LogicalState & getLogicalState() const {return logState;};
   //to also record what has changed
  	void addChange(const SimpleProposition *);
	void delChange(const SimpleProposition *);
	void updateChange(const FuncExp * fe,assign_op aop,FEScalar value);
   set<const SimpleProposition *> getChangedLiterals() const {return changedLiterals;};
   set<const FuncExp *> getChangedPNEs() const {return changedPNEs;};
   void resetChanged() {changedLiterals.clear(); changedPNEs.clear();};

	void setNew(const effect_lists * effs);

	void write(ostream & o) const
	{
		for(LogicalState::const_iterator i = logState.begin();i != logState.end();++i)
		{
			if(i->second) o << *(i->first) << "\n";
		};
		for(NumericalState::const_iterator i = feValue.begin();i != feValue.end();++i)
		{
			o << *(i->first) << " = " << i->second << "\n";
		};
	};

	//	friend class const_iterator;

	class const_iterator {
	private:
		const State & st;
		LogicalState::const_iterator it;
	public:
		const_iterator(const State & s) : st(s), it(st.logState.begin())
		{
			while(it != st.logState.end() && !it->second) ++it;
		};
		
		bool operator==(const const_iterator & itr) const
		{
			return it == itr.it;
		};

		bool operator!=(const const_iterator & itr) const
		{
			return it != itr.it;
		};

		const_iterator & operator++()
		{
			++it;
			while(it != st.logState.end() && !it->second) ++it;
			return *this;
		};

		const SimpleProposition * operator*() const
		{
			return it->first;
		};

		void toEnd()
		{
			it = st.logState.end();
		};
	};

	const_iterator begin() const {return const_iterator(*this);};
	const_iterator end() const {const_iterator ci(*this); ci.toEnd(); return ci;};

	void nowUpdated(const Happening * h)
	{
		for(vector<StateObserver *>::iterator i = sos.begin();i != sos.end();++i)
		{
			(*i)->notifyChanged(this,h);
		};
	};
	
	static void addObserver(StateObserver * s) {sos.push_back(s);}
	bool hasObservers() const {return !sos.empty();}
	void recordResponsibles(const map<const SimpleProposition *,set<const Action *> >& m1,
				const map<const FuncExp *,set<const Action *> > & m2)
	{
		responsibleForProps = m1;
		responsibleForPNEs = m2;
	}
	const set<const Action *> & whatDidThis(const SimpleProposition * sp) const
	{
		return responsibleForProps.find(sp)->second;
	}
	const set<const Action *> & whatDidThis(const FuncExp * fe) const
	{
		return responsibleForPNEs.find(fe)->second;
	}
};

inline ostream & operator<<(ostream & o,const State & s)
{
	s.write(o);
	return o;
};


};

#endif
