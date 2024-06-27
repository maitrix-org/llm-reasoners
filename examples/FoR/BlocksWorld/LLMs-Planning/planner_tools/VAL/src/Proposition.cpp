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

  $Date: 2009-02-05 10:50:21 $
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
#include <string>
#include <utility>
#include "Proposition.h"
#include "State.h"
#include "Action.h"
#include "Plan.h"
#include "Validator.h"
#include "Exceptions.h"
#include "Utils.h"
#include "Polynomial.h"
#include "Events.h"
#include "PrettyPrinter.h"

using std::make_pair;

namespace VAL {

template<class T> T max(T & t1,T & t2) {return t1>t2?t1:t2;};

//using std::max;
//#define vector std::vector

Environment SimpleProposition::nullEnvironment;
Environment DerivedGoal::nullEnvironment;
//vector<string> DerivedGoal::calledDPs;

map<string,bool> DerivedGoal::DPliterals;
vector<string> calledDPsCreate;
vector<string> DerivedGoal::calledDPsEval;
map<string, pair<int,int> > DerivedGoal::ranks;
const pair<int,int> DerivedGoal::noRank = make_pair(-1,-1);
map<string,bool> DerivedGoal::evals;
const int DerivedGoal::noEval = -1;
map<string,Intervals> DerivedGoal::intervals;
const Intervals DerivedGoal::noIntervals = Intervals(true);
map<string,string> DerivedGoal::propStrings;
const string DerivedGoal::noPropString ="";
map<string,const Action *> DerivedGoal::preCons;
const ActiveCtsEffects * DerivedGoal::ace;
bool DerivedGoal::rhsOpen;

ostream & operator <<(ostream & o,const Proposition & p)
{
	p.write(o);
	return o;

};





bool isPointInInterval(CoScalar p, const vector< pair<intervalEnd,intervalEnd> > & ints)
{

	for(vector< pair<intervalEnd,intervalEnd> >::const_iterator i = ints.begin(); i != ints.end();++i)
	{
		if( (p >= i->first.first) && (p <= i->second.first) )
		{
			//check point is not at end of interval where end is open
			if( !(( (p == i->first.first) && (!(i->first.second)) ) || ( (p == i->second.first) && (!(i->second.second)) )) )
			{
				return true;
			};
		};

	};

	return false;
};

bool isPointInInterval(CoScalar p, const pair<intervalEnd,intervalEnd> & ints)
{
	vector< pair<intervalEnd,intervalEnd> > someInterval;
	someInterval.push_back(ints);
	return isPointInInterval(p,someInterval);
};

bool isPointInInterval(CoScalar p, const vector< pair<intervalEnd,intervalEnd> > & ints, const pair<intervalEnd,intervalEnd> & int1)
{
	vector< pair<intervalEnd,intervalEnd> > someIntervals(ints);
	someIntervals.erase(std::remove(someIntervals.begin(),someIntervals.end(),int1),someIntervals.end());


	return isPointInInterval(p,someIntervals);
};

pair<intervalEnd,intervalEnd> getIntervalFromPt(intervalEnd p, const vector< pair<intervalEnd,intervalEnd> > & ints,const pair<intervalEnd,intervalEnd> & int1)
{
	vector< pair<intervalEnd,intervalEnd> > someIntervals(ints);

	unsigned int size = someIntervals.size();
	someIntervals.erase(std::remove(someIntervals.begin(),someIntervals.end(),int1),someIntervals.end());
	if(someIntervals.size() != size - 1) someIntervals.push_back(int1); //if repeated make sure not all occurances are deleted

	for(vector< pair<intervalEnd,intervalEnd> >::const_iterator i = someIntervals.begin(); i != someIntervals.end();++i)
	{
		if( (p.first >= i->first.first) && (p.first <= i->second.first) )
		{

			//check point is not at end of interval where end is open(assuming point is at start of interval)
			if( !(( (p.first == i->first.first) && (!(i->first.second)) && (p.second) ) ||
				  ( (p.first == i->second.first) && (!(i->second.second)) && (p.second)) ||

				 (!(p.second) && (p.first == i->second.first) )  ))
			{
					return *i;
			};
		};

	};


	intervalEnd st, ed;
	st.first = 0; st.second = false;
	ed.first = 0; ed.second = false;
	return make_pair(st,ed);

};

Intervals setUnion(const Intervals & ints1,const Intervals & ints2)
{
	Intervals theUnion;
	vector<pair<intervalEnd,intervalEnd> > theIntervals;
	CoScalar progress = 0, upperLimit =0;
	intervalEnd startPt,endPt;
	pair<intervalEnd,intervalEnd> aInterval;


	for(vector< pair<intervalEnd,intervalEnd> >::const_iterator i = ints1.intervals.begin(); i != ints1.intervals.end();++i)
		theIntervals.push_back(*i);

	for(vector< pair<intervalEnd,intervalEnd> >::const_iterator i1 = ints2.intervals.begin(); i1 != ints2.intervals.end();++i1)
		theIntervals.push_back(*i1);

	//find upper limit
	for(vector< pair<intervalEnd,intervalEnd> >::iterator i2 = theIntervals.begin(); i2 != theIntervals.end();++i2)
	{
		if(upperLimit < i2->second.first) upperLimit = i2->second.first + 1;
	};






	//build union by adding each interval in turn based on the collection of given intervals
	for(unsigned int count=0; count != theIntervals.size(); ++count)
	{

		//find smallest start point after 'progress'
		startPt = make_pair(upperLimit,false);

		for(vector< pair<intervalEnd,intervalEnd> >::iterator i = theIntervals.begin(); i != theIntervals.end();++i)
		{
			if((i->first.first >= progress) && (i->first.first <= startPt.first ))
			{
				if(!((i->second.first == aInterval.second.first) && (aInterval.second.second) )) //check new interval is adding anything (ie not a point at a closed end)
				{
					startPt.first = i->first.first;
					endPt = i->second;
				};


			};

		};

		if(startPt.first == upperLimit) break;

		//set closure of start point
		for(vector< pair<intervalEnd,intervalEnd> >::iterator i1 = theIntervals.begin(); i1 != theIntervals.end();++i1)
		{
			if( (startPt.first == i1->first.first) && (i1->first.second)) startPt.second = true; //make end pt closed


		};





		aInterval = make_pair(startPt,endPt);
		//find largest end pt where the corresponding start pt is in the given interval
		//(or given interval has open end and other interval is closed with end points touching etc)
		for(unsigned int count2=0; count2 != theIntervals.size() + 1; ++count2)
		{
			for(vector< pair<intervalEnd,intervalEnd> >::iterator i = theIntervals.begin(); i != theIntervals.end();++i)
			{
				if( ( (isPointInInterval(i->first.first,aInterval) && !( (i->first.first == aInterval.second.first) &&  !(i->first.second) && !(aInterval.second.second) )  )||
						( (i->first.first == aInterval.second.first) && ( i->first.second || aInterval.second.second ) ) )
					&& (i->second.first >= endPt.first ) )
				{
					endPt.first = i->second.first;
					endPt.second = i->second.second;

				};

			};

			if(aInterval.second == endPt) break;

			aInterval = make_pair(startPt,endPt);

		};

		//set closure of end point, if nec
		for(vector< pair<intervalEnd,intervalEnd> >::iterator i2 = theIntervals.begin(); i2 != theIntervals.end();++i2)
		{
			if( (endPt.first == i2->second.first) && (i2->second.second)) {endPt.second = true; aInterval = make_pair(startPt,endPt);}; //make end pt closed
		};

		progress = endPt.first;

		theUnion.intervals.push_back(aInterval);
	};

	return Intervals(theUnion);
};

Intervals setIntersect(const Intervals & ints1,const Intervals & ints2)
{
	Intervals theIntersect;
	vector<pair<intervalEnd,intervalEnd> > theIntervals;

	CoScalar progress = -1, upperLimit =0;//set progress to some number less than 0
	intervalEnd startPt,endPt;
	pair<intervalEnd,intervalEnd> aInterval,tempInterval;

	for(vector< pair<intervalEnd,intervalEnd> >::const_iterator i = ints1.intervals.begin(); i != ints1.intervals.end();++i)
		theIntervals.push_back(*i);

	for(vector< pair<intervalEnd,intervalEnd> >::const_iterator i1 = ints2.intervals.begin(); i1 != ints2.intervals.end();++i1)
		theIntervals.push_back(*i1);





	//find upper limit
	for(vector< pair<intervalEnd,intervalEnd> >::iterator i2 = theIntervals.begin(); i2 != theIntervals.end();++i2)
	{
		if(upperLimit < i2->second.first) upperLimit = i2->second.first + 1;
	};

	//build intersection by adding each interval in turn based on the collection of given intervals
	for(unsigned int count=0; count != theIntervals.size(); ++count)
	{

		//find smallest start point after 'progress' that is in another interval except its own interval
		startPt = make_pair(upperLimit,true);


		for(vector< pair<intervalEnd,intervalEnd> >::iterator i = theIntervals.begin(); i != theIntervals.end();++i)
		{
			if((i->first.first >= progress) && (i->first.first <= startPt.first )  )
			{
				if(!((i->first.first == aInterval.first.first) && (i->second.first == aInterval.second.first))) //check not adding a single point again
				{
					tempInterval = getIntervalFromPt(i->first,theIntervals,*i);
						if(!((tempInterval.first.first == 0) && (tempInterval.second.first == 0) &&
						(!(tempInterval.first.second)) && (!(tempInterval.second.second)) ))
					{
						startPt.first = i->first.first;
						if(!(i->first.second)) startPt.second = false; //make end pt open

						//make end point minimum of the two intervals
						if(i->second.first < tempInterval.second.first)
							endPt = i->second;

						else if(i->second.first > tempInterval.second.first)
							endPt = tempInterval.second;
						else if( !(i->second.second) || !(tempInterval.second.second) )

							{

								endPt.first = tempInterval.second.first;
								endPt.second = false;
							}
						else
							{
								endPt.first = tempInterval.second.first;
								endPt.second = true;
							};
					};
				};

			};

		};

		if(startPt.first == upperLimit) break;

		aInterval = make_pair(startPt,endPt);


		progress = endPt.first;

		theIntersect.intervals.push_back(aInterval);


	};

		    //cout << "intersect of "<<ints1 << " and "<< ints2 << " is "<<theIntersect<<"\n";
	return theIntersect;
};

Intervals setComplement(const Intervals & ints,double endPoint)
{

	Intervals theComplement;
	intervalEnd startPt,endPt;

   if( ints.intervals.size() == 0 )
   {
    theComplement.intervals.push_back(make_pair( make_pair(0,true) , make_pair(endPoint, true)));
    return theComplement;
   };

	vector< pair<intervalEnd,intervalEnd> >::const_iterator i = ints.intervals.begin();

	if(i->first.first != 0 || ( i->first.first == 0 && !(i->first.second)))
	{
		startPt = make_pair(0,true);
		endPt = make_pair(i->first.first,!(i->first.second));
		theComplement.intervals.push_back(make_pair(startPt,endPt));
	};

	for(vector< pair<intervalEnd,intervalEnd> >::const_iterator i1 = ints.intervals.begin(); i1 != ints.intervals.end();++i1)

	{

		if(!(i1->second.first == endPoint && i->second.second))
		{



			startPt = make_pair(i1->second.first,!(i1->second.second));

			if((i1+1) != ints.intervals.end())
			{
				endPt = make_pair((i1+1)->first.first,!((i1+1)->first.second));
			}
			else
			{
				endPt = make_pair(endPoint,true);
			};


			theComplement.intervals.push_back(make_pair(startPt,endPt));
		};
	};

	return theComplement;
};

void Comparison::write(ostream & o) const
{
	string op;
	switch(comp->getOp())
	{
		case E_GREATER:
			op = ">";
			break;
		case E_GREATEQ:
			if(LaTeX)
				op = "\\geq";
			else
				op = ">=";
			break;
		case E_LESS:
			op = ">";
			break;
		case E_LESSEQ:
			if(LaTeX)
				op = "\\geq";
			else
				op = ">=";
			break;
		case E_EQUALS:
			op = "=";

	};
	o << "(" << getExprnString(comp->getLHS(),bindings) << " " << op << "  " <<
		getExprnString(comp->getRHS(),bindings) << ")";
};

//check goal for dps and comps also, first = has dp, second = has comparison
pair<bool,bool> hasDP(const goal * g,Environment * env,const Validator * vld,bool dp,bool comp)
{

	if(dynamic_cast<const comparison*>(g))
	{

	  return make_pair(dp,true);
	};



	if(dynamic_cast<const conj_goal *>(g))
	{

		const conj_goal * cg = dynamic_cast<const conj_goal *>(g);

		bool existsDP = false;
		bool existsComp = false;
		for(goal_list::const_iterator i = cg->getGoals()->begin(); i != cg->getGoals()->end(); ++i)
		  {
		    if(hasDP(*i,env,vld,dp,comp).first)

		    {
		      existsDP = true;
		    };

		     if(hasDP(*i,env,vld,dp,comp).second)
		    {
		      existsComp = true;
		      if(existsDP) break;

		    };
		  };
		return make_pair(existsDP,existsComp);
	};




	if(dynamic_cast<const disj_goal*>(g))
	{
	 	const disj_goal * dg = dynamic_cast<const disj_goal *>(g);

		bool existsDP = false;
		bool existsComp = false;
		for(goal_list::const_iterator i = dg->getGoals()->begin(); i != dg->getGoals()->end(); ++i)
		  {
		    if(hasDP(*i,env,vld,dp,comp).first)
		    {
		      existsDP = true;
		    };

		     if(hasDP(*i,env,vld,dp,comp).second)
		    {
		      existsComp = true;
		      if(existsDP) break;


		    };
		  };
		return make_pair(existsDP,existsComp);
	};

	if(dynamic_cast<const neg_goal *>(g))
	{
		const neg_goal * ng = dynamic_cast<const neg_goal *>(g);
		return hasDP(ng->getGoal(),env,vld,dp,comp);
	};

	if(dynamic_cast<const imply_goal*>(g))
	{
		const imply_goal * ig = dynamic_cast<const imply_goal*>(g);

		pair<bool,bool> ant = hasDP(ig->getAntecedent(),env,vld,dp,comp);

		pair<bool,bool> cons = hasDP(ig->getConsequent(),env,vld,dp,comp);
		return make_pair(ant.first || cons.first,ant.second || cons.second);
	};

	if(dynamic_cast<const simple_goal*>(g))
	{
		const simple_goal * sg = dynamic_cast<const simple_goal*>(g);

		map<string,pair<const goal *,const var_symbol_table *> > derivPreds = vld->getDerivRules()->getDerivPreds();


		for(map<string,pair<const goal *,const var_symbol_table *> >::const_iterator i = derivPreds.begin(); i != derivPreds.end(); ++i)

		{
			if(sg->getProp()->head->getName() == i->first)
			{

			  return make_pair(true,comp);
			};


		};

	};


	if(dynamic_cast<const qfied_goal*>(g))
	{
		const qfied_goal * qg = dynamic_cast<const qfied_goal*>(g);
		return hasDP(qg->getGoal(),env,vld,dp,comp);

	};

	return make_pair(dp,comp);
};



struct compareCond {

	compareCond() {};

	bool operator()(const Proposition * p1,const Proposition * p2) const
	{
		pair<int,int> rnk1 = p1->rank();
		pair<int,int> rnk2 = p2->rank();

		if(rnk1.second == -1 && rnk2.second != -1)
		  return false;


		else if(rnk2.second == -1 && rnk1.second != -1)
		  return true;
		else if(rnk1.second < rnk2.second)
			return true;
		else if(rnk1.second == rnk2.second)
		{
			if(rnk1.first < rnk2.first)
				return true;
			else
				return false;
		}
		else
			return false;

		return true;
	};
};

struct compareCond2 {

	compareCond2() {};



	bool operator()(const Proposition * p1,const Proposition * p2) const
	{
		pair<int,int> rnk1 = p1->rank();
		pair<int,int> rnk2 = p2->rank();


		if(rnk1.second > rnk2.second)
			return true;
		else if(rnk1.second == rnk2.second)
		{
			if(rnk1.first < rnk2.first)
				return true;
			else

				return false;
		}
		else
			return false;

		return true;
	};
};

bool
ConjGoal::evaluate(const State * s,vector<const DerivedGoal*> DPs) const
{
	//reorder conditions in order of complexity, simplest first
	vector<const Proposition *> conditions;

	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
		conditions.push_back(*i);

	pair<int,int> rnk = rank();
	//in the case of derived predicates and simple props only
	if(rnk.second == -1)
  {
	  std::sort( conditions.begin() ,  conditions.end(), compareCond2()); //too much much work
  }
	else
	  std::sort( conditions.begin() ,  conditions.end(), compareCond());

	for(vector<const Proposition *>::const_iterator i1 = conditions.begin();i1 != conditions.end();++i1)
	{          //cout << **i1 <<" conj\n";
		if(!(*i1)->evaluate(s,DPs)) return false;
	};


	return true;

};

bool

ConjGoal::markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const
{
	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
		if(!(*i)->markOwnedPreconditions(a,o,w)) return false;
	return true;
};

bool ConstraintGoal::evaluate(const State *,vector<const DerivedGoal*> dgs) const
{
// This is the key function, maybe?
	return true;
};

Intervals ConstraintGoal::getIntervals(const State* s) const
{
// Not yet sure what this should return - this is a placeholder...
	Intervals ans;
	return ans;
};

string ConstraintGoal::getPropString(const State* s) const
{
// Placeholder.
	string ss = "A constraint";
	return ss;
};

set<const SimpleProposition *> ConstraintGoal::getLiterals() const
{
	set<const SimpleProposition *> literals = requirement->getLiterals();
	if(trigger)
	{
		set<const SimpleProposition *> literals1 = trigger->getLiterals();
		for(set<const SimpleProposition *>::const_iterator i = literals1.begin();
				i != literals.end();++i)
		{
			literals.insert(*i);
		};
	};
	return literals;
};

pair<int,int> ConstraintGoal::rank() const
{
	int count = 0;
	int maxDegree = 0;

	pair<int,int> rnk = requirement->rank();
	count += rnk.first;
	if(rnk.second > maxDegree || (maxDegree == 0 && rnk.second == -1)) maxDegree = rnk.second;

	if(trigger)
	{
		rnk = trigger->rank();
		count += rnk.first;
		if(rnk.second > maxDegree || (maxDegree == 0 && rnk.second == -1)) maxDegree = rnk.second;
	};

	return make_pair( count, maxDegree );
};

bool ConstraintGoal::markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const
{
// This is irrelevant since constraints can never appear as preconditions of an action
	return true;
};

void ConstraintGoal::setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen)
{
	const_cast<Proposition *>(requirement)->setUpComparisons(ace,rhsOpen);
	if(trigger)
		const_cast<Proposition *>(trigger)->setUpComparisons(ace,rhsOpen);
};

const AdviceProposition * ConstraintGoal::getAdviceProp(const State * s) const
{
// Placeholder...
    const AdvicePropositionConj * apc = new AdvicePropositionConj();

    return apc;
};

void ConstraintGoal::write(ostream & o) const
{
	o << "(";
	switch(constraint->getCons())
	{
		case E_ATEND:
			o << "at end " << *(getRequirement());
			break;
		case E_ALWAYS:
			o << "always " << *(getRequirement());
			break;
		case E_SOMETIME:
			o << "sometime " << *(getRequirement());
			break;
		case E_WITHIN:
			o << "within " << getDeadline() << " "
					<< *(getRequirement());
			break;
		case E_ATMOSTONCE:
			o << "at-most-once " << *(getRequirement());
			break;
		case E_SOMETIMEAFTER:
			o << "sometime-after " << *(getTrigger()) << " "
									<< *(getRequirement());
			break;
		case E_SOMETIMEBEFORE:
			o << "sometime-before " << *(getTrigger()) << " "
									<< *(getRequirement());
			break;
		case E_ALWAYSWITHIN:
			o << "always-within " << (getDeadline()) << " "
					<< *(getTrigger()) << " "
					<< *(getRequirement());
			break;
		case E_HOLDDURING:
			o << "hold-during " << getFrom() << " "
				<< getDeadline() << " " << getRequirement();
			break;
		case E_HOLDAFTER:
			o << "hold-after " << getFrom() << " "
				<< *(getRequirement());
			break;
		default:
			break;
	};
	o << ")";


};


bool PreferenceGoal::evaluate(const State * s,vector<const DerivedGoal *> DPs) const
{
	bool b = thePref->evaluate(s,DPs);
	if(!b)
	{
// Have to be careful, here. We want to ensure we count a violation for each instance
// of a preference, but only one for each instance (in particular, an over all condition
// is only violated once, even if it oscillates in truth value over the duration of
// an action). We also only want to count violations that represent conditions that
// actually need to be true - evaluate might be called in contexts where a violation
// should not be assessed. We will manage this at the validator end, in the countViolation
// method.
		vld->countViolation(s,pref->getName(),thePref->getAdviceProp(s));
	};
	return true;
};


bool
DisjGoal::evaluate(const State * s,vector<const DerivedGoal*> DPs) const
{             // cout << gs.size() << " disj size\n";
  pair<int,int> rnk = rank();
	//if rank = 0 ,0 then do as below else do getinterval check it is covered etc...
	if(rnk.second == 0)
	{
		for(vector<const Proposition *>::const_iterator i =  gs.begin();i !=  gs.end();++i)
			{          // cout << **i <<" disj\n";
        if((*i)->evaluate(s,DPs))  return true;
      };


		return false;
	};



	//reorder conditions in order of complexity, simplest first
	vector<const Proposition *> conditions;

	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
		conditions.push_back(*i);


	//in the case of derived predicates and simple props only
	if(rnk.second == -1)
	{

	std::sort( conditions.begin() ,  conditions.end(), compareCond2()); //too much much work

		for(vector<const Proposition *>::const_iterator i =  conditions.begin();i !=  conditions.end();++i)
		   if((*i)->evaluate(s,DPs))  return true;

		return false;
	};

	std::sort( conditions.begin() ,  conditions.end(), compareCond());

	Intervals theAns;

	try
	{
		for(vector<const Proposition *>::const_iterator i =  conditions.begin();i !=  conditions.end();++i)
		{
			theAns = setUnion(theAns,(*i)->getIntervals(s));

			if( (theAns.intervals.size() == 1) && (theAns.intervals.begin()->first.first == 0) && (theAns.intervals.begin()->second.first == endOfInterval))
				return true;
		};
	}
	catch(InvariantDisjError ide)
	{
		if(InvariantWarnings)
		{
			if(LaTeX)
				s->getValidator()->addInvariantWarning(getPropString(s)+",   for values of $t$ in $( 0 , "+toString(endOfInterval)+" )$\\\\\n");
			else
				s->getValidator()->addInvariantWarning(getPropString(s)+",   for values of t in ( 0 , "+toString(endOfInterval)+" )\n");

			return true;
		}
		else
		{

			if(LaTeX)


				*report << "\\\\\n \\> \\listrow{Invariant: "+getPropString(s)+", $t \\in ( 0 , "+toString(endOfInterval)+" )$}\\\\\n";
			else
				cout << "\nInvariant: "+getPropString(s)+", t in ( 0 , "+toString(endOfInterval)+" )\n";

			InvariantError ie;
			throw ie;
		};

	};

	return false;
};

bool
DisjGoal::markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const
{
	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
		if(!(*i)->markOwnedPreconditions(a,o,E_PRE)) return false;
	return true;

};

bool evaluateEquality(const proposition * prop,const Environment & bindings)
{
    string s1,s2;

/*
    for(Environment::const_iterator i = bindings.begin();i != bindings.end();++i)
    {
    	cout << (i->first)->getName() << "[" << (i->first) << "] ->"
    		 << (i->second)->getName() << "[" << (i->second) << "]\n";
    };
*/
    parameter_symbol_list::const_iterator i = prop->args->begin();
//cout << (*i) << " and ";
     if(dynamic_cast<const var_symbol *>(*i))
 		{
 			s1 = bindings.find(dynamic_cast<const var_symbol*>(*i))->second->getName();
        }
    else
             s1 = (*i)->getName();

   ++i;
//cout << (*i) << "\n";
        if(dynamic_cast<const var_symbol *>(*i))
 		{
 			s2 = bindings.find(dynamic_cast<const var_symbol*>(*i))->second->getName();
        }

    else
             s2 = (*i)->getName();
//cout << "Equality test " << s1 << " " << s2 << "\n";
 if(s1 ==  s2) return true;
 else return false;

};

bool
SimpleProposition::evaluate(const State * s,vector<const DerivedGoal*> DPs) const
{

    if(prop->head->getName() == "=")// && prop->args.size() == 2 )
	  {
       return evaluateEquality(prop,bindings);
    };

    bool ans = s->evaluate(this);
    //cout << *this << " is "<< ans << "\n";
	return ans;
};


bool
SimpleProposition::markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const
{
	return o.markOwnedPrecondition(a,this,w);
};

bool
DerivedGoal::markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const
{
 string dpName = getDPName();




 //cout << " marking precons "<<*this<<"\n";
 map<string,const Action *>::iterator i = preCons.find(dpName);

 if(i != preCons.end() && i->second == a) return true;

 preCons[dpName] = a;

 bool ans = deriveFormula->markOwnedPreconditions(a,o,w);

 return ans;
};

//evaluate comparison at a single point
bool
Comparison::evaluateAtPoint(const State * s) const
{
      		double lhs = s->evaluate(comp->getLHS(),bindings);
      		double rhs = s->evaluate(comp->getRHS(),bindings);

      		switch(comp->getOp())
      		{
      			case E_GREATER:
      				return lhs > rhs;

      			case E_GREATEQ:
      				return lhs >= rhs;

      			case E_LESS:
      				return lhs < rhs;

      			case E_LESSEQ:
      				return lhs <= rhs;

      			case E_EQUALS:
      				return lhs == rhs;

      			default:
      				return false;
      		};

};

//evaluate comparison at a single point, but if it is within a certain error then that is OK
bool
Comparison::evaluateAtPointError(const State * s) const
{
      double eval = s->evaluate(comp->getLHS(),bindings)  - s->evaluate(comp->getRHS(),bindings);
      double tooSmall = 0.0001;

      		switch(comp->getOp())
      		{
      			case E_GREATER:
      				return eval > -tooSmall;

      			case E_GREATEQ:
      				return eval >= -tooSmall;

      			case E_LESS:
      				return eval < tooSmall;

      			case E_LESSEQ:
      				return eval <= tooSmall;


      			case E_EQUALS:
      				return eval < tooSmall && eval > -tooSmall;

      			default:
      				return false;
      		};
};

bool Proposition::evaluateAtPointWithinError(const State* s,vector<const DerivedGoal*> DPs) const
{
   return evaluate(s,DPs);
};

bool Comparison::evaluateAtPointWithinError(const State* s,vector<const DerivedGoal*> DPs) const
{
   return evaluateAtPointError(s);
};

bool
Comparison::evaluate(const State * s,vector<const DerivedGoal*> DPs) const
{
    if(ctsFtn == 0)  return evaluateAtPoint(s);
    // RH proposed Error version here, but seems
    // wrong to me!
//cout << "CHECK WITH " << rhsIntervalOpen << " " << endOfInterval << "\n";
    return ctsFtn->checkInvariant(this,s,endOfInterval,rhsIntervalOpen);

     /*
   Intervals someIntervals = ctsFtn->getIntervals(this,s,endOfInterval);

   if(someIntervals.intervals.size() != 1 || (*someIntervals.intervals.begin()).first.first != 0 || (*someIntervals.intervals.begin()).second.first != endOfInterval)
   return false;
   else return true;
       */

};




bool
Comparison::markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const
{
	return o.markOwnedPreconditionFEs(a,comp->getLHS(),bindings) &&
		   o.markOwnedPreconditionFEs(a,comp->getRHS(),bindings);
};


bool
ImplyGoal::evaluate(const State * s,vector<const DerivedGoal*> DPs) const
{
	bool b = !(ant->evaluate(s,DPs));
//	cout << "Implication from " << b << "\n";
//	if(b)
//	{
//		cout << "Rest is: " << cons->evaluate(s,DPs) << "\n";
//	};
	return (b || cons->evaluate(s,DPs));

};


bool
ImplyGoal::markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const
{
	return (ant->markOwnedPreconditions(a,o,E_PRE) &&
			cons->markOwnedPreconditions(a,o,E_PRE));
};

bool
NegGoal::evaluate(const State * s,vector<const DerivedGoal*> DPs) const
{

	return !(p->evaluate(s,DPs));
};

bool
NegGoal::markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const

{
	ownership w1 = w==E_PRE?E_PRE:(w==E_PPRE?E_NPRE:E_PPRE);
	return (p->markOwnedPreconditions(a,o,w1));

};


string ConjGoal::getPropString(const State* s) const

{
	string ans;
	bool lots = gs.size() > 1;

	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();)
	{

		if(lots)

			ans += "("+(*i)->getPropString(s)+")";
		else
			ans += (*i)->getPropString(s);
		++i;

		if(i!= gs.end())
		{
			if(LaTeX)
			{
				ans += " $\\land$ ";
			}
			else
			{
				ans += " AND ";
			};
		};
	};


	return ans;

};

string PreferenceGoal::getPropString(const State * s) const
{
	return "true";
};


// Maybe this should just return an empty set - depends on why getLiterals is used.
set<const SimpleProposition *> PreferenceGoal::getLiterals() const
{
   return thePref->getLiterals();
};

pair<int,int> PreferenceGoal::rank() const
{
	return make_pair(0,0);
};

bool PreferenceGoal::markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const
{
	return thePref->markOwnedPreconditions(a,o,w);
};

void PreferenceGoal::setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen)
{
	const_cast<Proposition *>(thePref)->setUpComparisons(ace,rhsOpen);
};
string DisjGoal::getPropString(const State* s) const
{
	string ans;
	bool lots = gs.size() > 1;


	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();)


	{
		if(lots)
			ans += "("+(*i)->getPropString(s)+")";
		else
			ans += (*i)->getPropString(s);


		++i;

		if(i!= gs.end())
		{
			if(LaTeX)
			{
				ans += " $\\lor$ ";
			}
			else
			{
				ans += " OR ";
			};
		};
	};


	return ans;


};

string SimpleProposition::getPropString(const State* s) const

{


	if(evaluate(s))

		return "true";

	else
		return "false";

};

string DerivedGoal::getPropString(const State* s) const
{

 string dpName = getDPName();

 map<string,string>::iterator i = propStrings.find(dpName);


 if(i != propStrings.end())
   {
     if(i->second == noPropString)
       return "("+dpName+")";
     else
       return i->second;
   };

 propStrings[dpName] = noPropString;

 string ans = deriveFormula->getPropString(s);


 propStrings[dpName] = ans;

 return ans;
};

string Comparison::getPropString(const State* s) const
{
	string ans;
	string op;
	switch(comp->getOp())
	{
		case E_GREATER:
			op = ">";
			break;
		case E_GREATEQ:
			if(LaTeX)
				op = "\\geq";
			else
				op = ">=";
			break;
		case E_LESS:
			op = ">";
			break;
		case E_LESSEQ:
			if(LaTeX)
				op = "\\geq";
			else
				op = ">=";
			break;
		case E_EQUALS:
			op = "=";

	};
      double offSet = 0;
      if(const BatteryCharge * bc = dynamic_cast<const BatteryCharge *>(ctsFtn))
      {
          offSet = bc->getOffSet();
      };


    	if(LaTeX)
    		{ans = "$"+toString(ctsFtn)+" "+op+" "+toString(offSet)+"$ for $t\\in$ (0,"+toString(endOfInterval);
            if(rhsIntervalOpen) ans += ")"; else ans += "]";
        }
    	else
    		{ans = toString(ctsFtn) + " "+op+" "+toString(offSet)+" for t in (0,"+toString(endOfInterval);
            if(rhsIntervalOpen) ans += ")"; else ans += "]";
          };
	return ans;
};

string ImplyGoal::getPropString(const State* s) const
{

	string ans;
	if(LaTeX)

	{
		ans = "("+cons->getPropString(s) +" $\\rightarrow$ "+ant->getPropString(s)+")";
	}

	else
	{
		ans = "("+cons->getPropString(s) +" implies "+ant->getPropString(s)+")";
	};

	return ans;
};


string NegGoal::getPropString(const State* s) const
{
	if(LaTeX)
		return "$\\neg$("+p->getPropString(s)+")";
	else
		return "NOT ("+p->getPropString(s)+")";

};

string QfiedGoal::getPropString(const State* s) const
{
	if(!pp) create();

	return pp->getPropString(s);

};



set<const SimpleProposition *> SimpleProposition::getLiterals() const
{

    set<const SimpleProposition *> literals;
    literals.insert(this);
    return literals;
};

set<const SimpleProposition *> Comparison::getLiterals() const
{
    set<const SimpleProposition *> comps;
    return comps;
};

set<const SimpleProposition *> ConjGoal::getLiterals() const
{
    set<const SimpleProposition *> literalspnes;

    for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end(); ++i)
	{
    set<const SimpleProposition *> someLiteralsPNEs = (*i)->getLiterals();


    for(set<const SimpleProposition *>::const_iterator j =  someLiteralsPNEs.begin(); j != someLiteralsPNEs.end(); ++j)
        literalspnes.insert(*j);
    };

    return literalspnes;
};

set<const SimpleProposition *> DisjGoal::getLiterals() const
{
   set<const SimpleProposition *> literalspnes;

  for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end(); ++i)
	{
    set<const SimpleProposition *> someLiteralsPNEs = (*i)->getLiterals();


    for(set<const SimpleProposition *>::const_iterator j =  someLiteralsPNEs.begin(); j != someLiteralsPNEs.end(); ++j)
        literalspnes.insert(*j);
   };

    return literalspnes;
};

set<const SimpleProposition *> NegGoal::getLiterals() const
{
    set<const SimpleProposition *> literalspnes = p->getLiterals();

    return literalspnes;
};

set<const SimpleProposition *> DerivedGoal::getLiterals() const
{
    set<const SimpleProposition *> literalspnes;

    string dpName = getDPName();

    map<string,bool>::iterator i = DPliterals.find(dpName);

    if(i != DPliterals.end()) return literalspnes;

    DPliterals[dpName] = true;

    literalspnes = deriveFormula->getLiterals();

    return literalspnes;
};


set<const SimpleProposition *> ImplyGoal::getLiterals() const
{
    set<const SimpleProposition *> literalspnes = ant->getLiterals();

    set<const SimpleProposition *> someLiterals = cons->getLiterals();

    for(set<const SimpleProposition *>::const_iterator j =  someLiterals.begin(); j != someLiterals.end(); ++j)
        literalspnes.insert(*j);

    return literalspnes;
};

set<const SimpleProposition *> QfiedGoal::getLiterals() const
{
    if(!pp) create();
    set<const SimpleProposition *> literalspnes = pp->getLiterals();

    return literalspnes;
};

Intervals ConjGoal::getIntervals(const State* s) const
{
	Intervals theAns;

	//reorder conditions in order of complexity, simplest first
	vector<const Proposition *> conditions;


	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
		conditions.push_back(*i);


	if(conditions.size() == 0)
  {
    theAns.intervals.push_back(make_pair( make_pair(0,true) , make_pair(endOfInterval, true)));
    return theAns;
  };

	vector<const Proposition *>::const_iterator i1 =  conditions.begin();

	if(conditions.size() == 1)
	{
		return (*i1)->getIntervals(s);
	};

	std::sort( conditions.begin() ,  conditions.end(), compareCond());

	theAns = (*i1)->getIntervals(s);
	++i1;

	for(;i1 !=  conditions.end();++i1)
	{
		theAns = setIntersect(theAns,(*i1)->getIntervals(s));
		if(theAns.intervals.size() == 0) return theAns;
	};

	return theAns;
};

Intervals DisjGoal::getIntervals(const State* s) const
{
	Intervals theAns;

	//reorder conditions in order of complexity, simplest first
	vector<const Proposition *> conditions;

	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
		conditions.push_back(*i);

	std::sort( conditions.begin() ,  conditions.end(), compareCond());

	for(vector<const Proposition *>::const_iterator i1 =  conditions.begin();i1 !=  conditions.end();++i1)
	{
		theAns = setUnion(theAns,(*i1)->getIntervals(s));
		if( (theAns.intervals.size() == 1) && (theAns.intervals.begin()->first.first == 0) && (theAns.intervals.begin()->second.first == endOfInterval))
			return theAns;
	};

	return theAns;
};

Intervals SimpleProposition::getIntervals(const State* s) const
{
	Intervals theAns;

	if(evaluate(s))

	{
		theAns.intervals.push_back(make_pair( make_pair(0,true) , make_pair(endOfInterval, true)));
	};


	return theAns;
};

Intervals DerivedGoal::getIntervals(const State* s) const
{


 string dpName = getDPName();

 map<string, Intervals >::iterator i = intervals.find(dpName);

 if(i != intervals.end())
   {
     if(i->second == noIntervals)
	 {
	   Intervals noInts;
	   return noInts;
	 }
     else
       return i->second;

   };


 intervals[dpName] = noIntervals;

 Intervals ans = deriveFormula->getIntervals(s);


 //can only cache if true
 vector< pair<intervalEnd,intervalEnd> >::const_iterator j = ans.intervals.begin();
 if((j->first.first == 0) && (j->first.second == true) && (j->second.first == endOfInterval) && (j->second.second == true)) intervals[dpName] = ans;

 return ans;
};

Intervals Comparison::getIntervals(const State* s) const
{

	Intervals theAns;

  if(ctsFtn == 0)
  {

		if(evaluate(s))
		{
			theAns.intervals.push_back(make_pair( make_pair(0,true) , make_pair(endOfInterval, true)));

		};
		return theAns;
	};



  theAns = ctsFtn->getIntervals(this,s,endOfInterval);
  /*
	if(poly == 0 || (poly->getDegree() == 0) )
	{
		if(evaluate(s))
		{
			theAns.intervals.push_back(make_pair( make_pair(0,true) , make_pair(endOfInterval, true)));
		};

		return theAns;
	}
	else if(comp->getOp() == E_EQUALS)
	{
		if( (poly->getDegree() == 0) && (poly->getCoeff(0) == 0))
		{
			theAns.intervals.push_back(make_pair( make_pair(0,true) , make_pair(endOfInterval, true)));
		};

		return theAns;
	};

	bool strict = (comp->getOp() == E_GREATER) || (comp->getOp() == E_LESS);

	vector<CoScalar> roots = getRootsForIntervals(s,endOfInterval);
	std::sort(roots.begin(),roots.end());

	pair<CoScalar,bool> startPt,endPt;

	if(roots.size() == 0)
	{

		if( poly->evaluate(endOfInterval/2) > 0 )
		{
			theAns.intervals.push_back(make_pair( make_pair(0,true) , make_pair(endOfInterval, true)));
		};

		return theAns;
	};

	//determine if starting satisfied or not
	Intervals aInt;
	vector< CoScalar >::const_iterator i = roots.begin();
	if(poly->evaluate(0) > 0)
	{
		startPt = make_pair(0,true);
		endPt = make_pair(*i,!(strict));
		aInt.intervals.push_back(make_pair(startPt,endPt));
		theAns = setUnion(theAns,aInt);
	}
	else if(poly->evaluate(0) == 0)
	{

		bool greaterZero;

		if( (i+1) != roots.end())
		{
			greaterZero = (poly->evaluate( *(i+1) / 2 ) > 0);
		}
		else
		{
			greaterZero = (poly->evaluate( endOfInterval / 2)  > 0);
		};



	};


	for(; i != roots.end();++i)
	{


		if(*i != endOfInterval)
		{
			aInt.intervals.clear();
			startPt = make_pair(*i,!(strict));

			if((i+1) != roots.end())
			{
				endPt = make_pair(*(i+1),!(strict));
			}
			else
			{
				endPt = make_pair(endOfInterval,true);
			};

			//in case of a repeated root
			if(poly->evaluate((startPt.first + endPt.first)/2) > 0)

			{
				aInt.intervals.push_back(make_pair(startPt,endPt));
				theAns = setUnion(theAns,aInt);
			}
			else if(!(strict))

			{
				aInt.intervals.push_back(make_pair(startPt,startPt));
				theAns = setUnion(theAns,aInt);
			};

		};

	};	*/
	//cout << " \\\\ \\> The ctsFtn $"<<*ctsFtn<<"$ is satisfied on "<<theAns<<" until "<<endOfInterval<<"\\\\\n";
	return theAns;
};



vector<CoScalar> Comparison::getRoots(const State* s,CoScalar t) const
{
	vector<CoScalar> roots;

	//check for polys we cannot find the roots of
	try
	{
		roots = ctsFtn->getRoots(t);
	}
	catch(PolyRootError & pre)

	{


		if(InvariantWarnings)
		{
			if(LaTeX)
				s->getValidator()->addInvariantWarning(getPropString(s)+",   for values of $t$ in $( 0 , "+toString(endOfInterval)+" )$\\\\\n");
			else
				s->getValidator()->addInvariantWarning(getPropString(s)+",   for values of t in ( 0 , "+toString(endOfInterval)+" )\n");


			return roots;
		}
		else
		{
			if(LaTeX)
				*report << "\\\\\n \\> Invariant: "+getPropString(s)+", $t \\in ( 0 , "+toString(endOfInterval)+" )$\\\\\n";
			else
				cout << "\nInvariant: "+getPropString(s)+", t in ( 0 , "+toString(endOfInterval)+" )\n";

			InvariantError ie;
			throw ie;
		};


	};



	return roots;
};

vector<CoScalar> Comparison::getRootsForIntervals(const State* s,CoScalar t) const
{

	vector<CoScalar> roots;

	try
	{
		roots = ctsFtn->getRoots(t);
	}
	catch(PolyRootError & pre)
	{
		// Let's see what we can do here.
		if(LaTeX)
				*report << "\\\\\nProblem with polynomial: cannot find roots. Assume it has none.\\\\\n";
		else
				cout << "\nProblem with polynomial: cannot find roots. Assume it has none.\n";
		//throw pre;
	};

	return roots;
};

Intervals ImplyGoal::getIntervals(const State* s) const
{
	return setUnion(setComplement(ant->getIntervals(s),endOfInterval),cons->getIntervals(s));
};

Intervals NegGoal::getIntervals(const State* s) const
{
	return setComplement(p->getIntervals(s),endOfInterval);
};

Intervals QfiedGoal::getIntervals(const State* s) const
{
        if(!pp) create();
	return pp->getIntervals(s);
};

Intervals PreferenceGoal::getIntervals(const State * s) const
{
	Intervals theAns;

	theAns.intervals.push_back(make_pair( make_pair(0,true) , make_pair(endOfInterval, true)));

	return theAns;
};

void ConjGoal::setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen)
{
	endOfInterval = ace->localUpdateTime;
	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)

		const_cast<Proposition*>(*i)->setUpComparisons(ace,rhsOpen);

};

void DisjGoal::setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen)
{
	endOfInterval = ace->localUpdateTime;
	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
		const_cast<Proposition*>(*i)->setUpComparisons(ace,rhsOpen);
};

void SimpleProposition::setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen)
{
	endOfInterval = ace->localUpdateTime;
};


void DerivedGoal::setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen)
{
};

bool isExpressionConstant(const expression * e,const ActiveCtsEffects * ace,const Environment & bs,CoScalar endInt)
{
	if(const div_expression * fexpression = dynamic_cast<const div_expression *>(e))
	{
		return isExpressionConstant(fexpression->getLHS(),ace,bs,endInt) && isExpressionConstant(fexpression->getRHS(),ace,bs,endInt);
	};

	if(dynamic_cast<const minus_expression *>(e))
	{
		return isExpressionConstant(dynamic_cast<const minus_expression*>(e)->getLHS(),ace,bs,endInt) &&
				isExpressionConstant(dynamic_cast<const minus_expression*>(e)->getRHS(),ace,bs,endInt);
	};

	if(dynamic_cast<const mul_expression *>(e))
	{
		return isExpressionConstant(dynamic_cast<const mul_expression*>(e)->getLHS(),ace,bs,endInt) &&
				isExpressionConstant(dynamic_cast<const mul_expression*>(e)->getRHS(),ace,bs,endInt);
	};

	if(dynamic_cast<const plus_expression *>(e))
	{
		return isExpressionConstant(dynamic_cast<const plus_expression*>(e)->getLHS(),ace,bs,endInt) &&
				isExpressionConstant(dynamic_cast<const plus_expression*>(e)->getRHS(),ace,bs,endInt);
	};

	if(dynamic_cast<const num_expression*>(e))
	{
		return true;
	};

	if(dynamic_cast<const uminus_expression*>(e))
	{
		return isExpressionConstant(dynamic_cast<const uminus_expression*>(e)->getExpr(),ace,bs,endInt);
	};

	if(const func_term * fexpression = dynamic_cast<const func_term *>(e))
	{
		const FuncExp * fexp;


		fexp = ace->vld->fef.buildFuncExp(fexpression,bs);

		map<const FuncExp *,ActiveFE*>::const_iterator i = ace->activeFEs.find(fexp);

		if(i !=  ace->activeFEs.end())
		{
			if(i->second->ctsFtn != 0)
			{
				if(const Polynomial * poly = dynamic_cast<const Polynomial *>(i->second->ctsFtn))
             {
                 return (poly->getDegree() == 0);
             }
				else if(const Exponential * exp = dynamic_cast<const Exponential *>(i->second->ctsFtn))
				{
                 return (exp->getK() == 0) || (exp->getPolynomial()->getDegree() == 0);
				}


            else if(dynamic_cast<const NumericalSolution *>(i->second->ctsFtn))
            {
                 return false;
            };

			}
			else
			{
          return true;
			};
		}
		else
		{

       return true;

		};
	};


	if(const special_val_expr * sp = dynamic_cast<const special_val_expr *>(e))
	{

		if(sp->getKind() == E_TOTAL_TIME)
		{

			return true;
		};

		if(sp->getKind() == E_DURATION_VAR)
		{
			return true;
		};

		if(sp->getKind() == E_HASHT)
		{


			return false;
		}
	};


 return true;
};

void Comparison::setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen)
{
	if(ctsFtn != 0) delete ctsFtn;
	endOfInterval = ace->localUpdateTime; //cout << endOfInterval << " setupcomps\n";

	rhsIntervalOpen = rhsOpen;

  //see if comparison is of the nature f(t) > k, for numerical solution f and constant k

  const BatteryCharge * numSoln = 0;

  const Exponential * expon = 0;
  double constant;
  bool defined =false, fndefined = false;
  const FuncExp * fexp;
  if(const func_term * fexpression = dynamic_cast<const func_term *>(comp->getLHS()))
	{
		fexp = ace->vld->fef.buildFuncExp(fexpression,bindings);
		map<const FuncExp *,ActiveFE*>::const_iterator i = ace->activeFEs.find(fexp);

		if(i !=  ace->activeFEs.end())
      {                 //cout << " cts ftn = "<<  *(i->second->ctsFtn) <<"\n";
        if(const BatteryCharge * bc = dynamic_cast<const BatteryCharge *>(i->second->ctsFtn))
        {
            numSoln = bc; fndefined  = true;
        }
        else if(const Exponential * exp = dynamic_cast<const Exponential *>(i->second->ctsFtn))
        {
             expon = exp; fndefined  = true;
        };

        if(fndefined)
        {
             if(isExpressionConstant(comp->getRHS(),ace,bindings,endOfInterval))  //check is constant!
             {
              constant = ace->vld->getState().evaluate(comp->getRHS(),bindings);
               defined  = true;
              };

         };
      };

    }
    else if(const func_term * fexpression = dynamic_cast<const func_term *>(comp->getRHS()))
	{
		fexp = ace->vld->fef.buildFuncExp(fexpression,bindings);
		map<const FuncExp *,ActiveFE*>::const_iterator i = ace->activeFEs.find(fexp);

		if(i !=  ace->activeFEs.end())
      {
        if(const BatteryCharge * bc = dynamic_cast<const BatteryCharge *>(i->second->ctsFtn))
        {
            numSoln = bc; fndefined =true;
        }
        else if(const Exponential * exp = dynamic_cast<const Exponential *>(i->second->ctsFtn))
        {
             expon = exp; fndefined  = true;
        };

        if(fndefined)
        {
            if(isExpressionConstant(comp->getLHS(),ace,bindings,endOfInterval))  //check is constant!
             {
              constant = ace->vld->getState().evaluate(comp->getLHS(),bindings);
               defined  = true;
              };
        };

      };



    };


    if(defined)
    {
     if(numSoln != 0)
     {
        BatteryCharge * numericalSolution = new BatteryCharge(*numSoln);
        numericalSolution->setOffSet(constant);
        ctsFtn = numericalSolution;
        return;
     }
     else if(expon != 0)
     {
        // cout << "exponential for inv = "<<*expon<<" with offset "<<constant<<"\n";
        Exponential * exponential;
        if((comp->getOp() ==  E_GREATER) || (comp->getOp() ==  E_GREATEQ) || (comp->getOp() ==  E_EQUALS))
	     {
          exponential = new Exponential(expon->getK(),new Polynomial(*expon->getPolynomial()),expon->getc());
          exponential->setOffSet(constant);
        }
        else
        {
          exponential = new Exponential(-expon->getK(),new Polynomial(*expon->getPolynomial()),-expon->getc());
          exponential->setOffSet(-constant);
        };
        ctsFtn = exponential;
        return;
      };
    };


   Polynomial thePoly;
	if((comp->getOp() ==  E_GREATER) || (comp->getOp() ==  E_GREATEQ) || (comp->getOp() ==  E_EQUALS))
	{
		thePoly = getPoly(comp->getLHS(),ace,bindings,endOfInterval) - getPoly(comp->getRHS(),ace,bindings,endOfInterval);
	}
	else
	{
		thePoly = getPoly(comp->getRHS(),ace,bindings,endOfInterval) - getPoly(comp->getLHS(),ace,bindings,endOfInterval);
	};

	ctsFtn = new Polynomial(thePoly);
	//cout << TestingPNERobustness << " "<<JudderPNEs<<" \\\\\\>Invariant Poly = $"<< *ctsFtn <<"$ \\\\\n";


};


void ImplyGoal::setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen)
{
	endOfInterval = ace->localUpdateTime;

	const_cast<Proposition*>(ant)->setUpComparisons(ace,rhsOpen);
	const_cast<Proposition*>(cons)->setUpComparisons(ace,rhsOpen);
};

void NegGoal::setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen)
{
	endOfInterval = ace->localUpdateTime;
	const_cast<Proposition*>(p)->setUpComparisons(ace,rhsOpen);
};

void QfiedGoal::setUpComparisons(const ActiveCtsEffects * ace,bool rhsOpen)
{
        if(!pp) create();

	endOfInterval = ace->localUpdateTime;
	const_cast<Proposition*>(pp)->setUpComparisons(ace,rhsOpen);
};

void PreferenceGoal::resetCtsFunctions()
{
	const_cast<Proposition *>(thePref)->resetCtsFunctions();
};

void ConjGoal::resetCtsFunctions()
{
	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
		const_cast<Proposition*>(*i)->resetCtsFunctions();

};

void DisjGoal::resetCtsFunctions()
{
	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
		const_cast<Proposition*>(*i)->resetCtsFunctions();

};

void SimpleProposition::resetCtsFunctions()
{
};


void DerivedGoal::resetCtsFunctions()
{
};

void ImplyGoal::resetCtsFunctions()
{
	const_cast<Proposition*>(ant)->resetCtsFunctions();
	const_cast<Proposition*>(cons)->resetCtsFunctions();
};

void NegGoal::resetCtsFunctions()
{
	const_cast<Proposition*>(p)->resetCtsFunctions();
};

void QfiedGoal::resetCtsFunctions()
{
        if(!pp) create();

	const_cast<Proposition*>(pp)->resetCtsFunctions();
};

void Comparison::resetCtsFunctions()
{
  if(ctsFtn != 0) delete ctsFtn;
  ctsFtn = 0;
};

pair<int,int>
ConjGoal:: rank() const
{
	int count = 0;
	int maxDegree = 0;

	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
	{
		pair<int,int> rnk = (*i)->rank();
		count += rnk.first;
		if(rnk.second > maxDegree || (maxDegree == 0 && rnk.second == -1)) maxDegree = rnk.second;
	};
	return make_pair( count, maxDegree );

};


pair<int,int>
DisjGoal::rank() const
{
	int count = 0;
	int maxDegree = 0;

	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
	{
		pair<int,int> rnk = (*i)->rank();
		count += rnk.first;
		if(rnk.second > maxDegree || (maxDegree == 0 && rnk.second == -1) ) maxDegree = rnk.second;
	};

	return make_pair( count, maxDegree );

};

pair<int,int>
SimpleProposition::rank() const
{
	return make_pair(0,0);
};

pair<int,int>
DerivedGoal::rank() const
{
   return make_pair(0,-1);
 /*string dpName = getDPName();
 //cout <<" ranking "<<*this<<"\n";
 map<string, pair<int,int> >::iterator i = ranks.find(dpName);

 if(i != ranks.end())
   {

     if(i->second == noRank)
       return make_pair(0,-1);
     else
       return i->second;
   };


 ranks[dpName] = noRank;

 pair<int,int> ans = deriveFormula->rank();

 ranks[dpName] = ans;


 return ans;*/
};

pair<int,int>
Comparison::rank() const
{
	if(ctsFtn == 0) return make_pair(1,0);

   if(const Polynomial* poly = dynamic_cast<const Polynomial*>(ctsFtn))
   {
      return make_pair(1,poly->getDegree());
   };
	return make_pair(1,4);
};

pair<int,int>
ImplyGoal::rank() const
{
	pair<int,int> rnkAnt = ant->rank();
	pair<int,int> rnkCons = cons->rank();
	return make_pair(rnkAnt.first + rnkCons.first , max(rnkAnt.second,rnkCons.second) );
};

pair<int,int>
NegGoal::rank() const

{
	return p->rank();
};

pair<int,int> QfiedGoal::rank() const
{
  if(!pp)
    { //dont want to keep expanding if dps are involved
      pair<bool,bool> dpComp = hasDP(qg,env,vld,false,false);

      if(dpComp.first)
	{
	  if(!dpComp.second)
	    return make_pair(0,-1);
	  else
	    return make_pair(0,1000);//after simple preds and comps
	};
      if(createLiterals) return make_pair(0,0);
      create();
    };

  return pp->rank();

};

const Proposition *
PropositionFactory::buildProposition(const goal * g,bool buildNewLiterals,const State * state)
{
	static Environment nullBindings;

	return buildProposition(g,nullBindings,buildNewLiterals,state);
};

/* //for testing purposes
void displaybs(const Environment &bs)
{
  cout << " For "<<&bs<<"\\\\\n";
  for(map<const var_symbol*,const const_symbol*>::const_iterator i = bs.begin(); i!= bs.end() ; ++i)
    {
      cout << i->first <<" -- " << i->first->getName() <<" = " << i->second->getName()<<"\\\\\n";

    };
  cout << "\\\\\n";
};
*/
string getDPName(const simple_goal* sg,Environment * bs)
{

 string propName = sg->getProp()->head->getName();

 for(parameter_symbol_list::const_iterator i = sg->getProp()->args->begin();
				i != sg->getProp()->args->end();++i)
 {
		if(dynamic_cast<const var_symbol *>(*i))


		{
			propName += bs->find(dynamic_cast<const var_symbol*>(*i))->second->getName();
		}

		else
		{
			propName += dynamic_cast<const const_symbol*>(*i)->getName();
		};
 };

 return propName;

};


string DerivedGoal::getDPName() const
{
	string propName = prop->head->getName();


	for(parameter_symbol_list::const_iterator i = prop->args->begin();
				i != prop->args->end();++i)
	{
		if(dynamic_cast<const var_symbol *>(*i))

		{
			propName += bindings.find(dynamic_cast<const var_symbol*>(*i))->second->getName();
		}
		else
		{
			propName += dynamic_cast<const const_symbol*>(*i)->getName();
		};
	};


	return propName;
};

void DerivedGoal::resetLists(const State* s)
{

  //too much calculating if a changed literal affects a DP
  //get changed literals and PNEs
 /* set<const FuncExp *> changedPNEs = s->getChangedPNEs();
  if(changedPNEs.size() > 0) evals.clear();
  else
  {
     set<const SimpleProposition *> changedLiterals = s->getChangedLiterals();
     string SPName;
     //loop thro' changed literals
     for(set<const SimpleProposition *>::const_iterator cl = changedLiterals.begin(); cl != changedLiterals.end(); ++cl)
     {

         vector<string> toErase;
         //loop thro' each cached DP eval, find out if it has been affected
         for(map<string,bool>::iterator e = evals.begin(); e != evals.end(); ++e)
         {
          map<string,set<const SimpleProposition *> >::iterator i = dependsOnSPs.find(e->first);
          //if the literal affects the DP then remove it from the cache since its value may have changed
          set<const SimpleProposition *>::const_iterator j = (i->second).find(*cl);
          if(j != (i->second).end())
          {
             toErase.push_back(e->first);
          };
         };

         //erase DPs from cached evaluations of DPs
         for(vector<string>::iterator te = toErase.begin(); te != toErase.end(); ++te)
         {
           evals.erase(*te);
         };

      };
  };
 */
   evals.clear();
   intervals.clear();
   propStrings.clear();
   preCons.clear();
};

bool

DerivedGoal::evaluate(const State * s,vector<const DerivedGoal*> DPs) const
{
 string dpName = getDPName();

 //setUp Comparisons
 if(ace != 0) const_cast<Proposition*>(deriveFormula)->setUpComparisons(ace,rhsOpen);


//cout << "Looking for " << *this << "\n";
 map<string,bool>::iterator i = evals.find(dpName);

 if(i != evals.end())
   {
   //cout << "Found it and it is " << i->second << "\n";
     return i->second;
   };

 revisit = false;
 if(visited(dpName))
  {
    for(vector<const DerivedGoal*>::iterator i = DPs.begin(); i != DPs.end(); ++i) (*i)->setRevisit(true);
    return false;
  };


 addCalledDP(dpName);

 DPs.push_back(this);
 bool ans = deriveFormula->evaluate(s,DPs);

 removeCalledDP(dpName);

 if(ans || !revisit)
 {
   evals[dpName] = ans;
 };

 return ans;
};

bool FalseProposition::evaluate(const State * s,vector<const DerivedGoal*> DPs) const
{
    for(vector<const DerivedGoal*>::iterator i = DPs.begin(); i != DPs.end(); ++i) (*i)->setRevisit(true);

    if(trueProp) return true;
    else
    return false;
};

void removeCalledDP(string dp)
{

	  for(vector<string>::iterator k = calledDPsCreate.begin(); k != calledDPsCreate.end(); ++k)
	    {

	      if(*k == dp)
		{

		  calledDPsCreate.erase(k);
		  break;
		};
	    };

};

void addCalledDP(string dp)
{


  calledDPsCreate.push_back(dp);


};


void DerivedGoal::removeCalledDP(string dp) const

{

	  for(vector<string>::iterator k = calledDPsEval.begin(); k != calledDPsEval.end(); ++k)
	    {
	      if(*k == dp)
		{
		  calledDPsEval.erase(k);
		  break;
		};
	    };

};

void DerivedGoal::addCalledDP(string dp) const
{
  calledDPsEval.push_back(dp);
};

bool DerivedGoal::visited(string dp) const
{

 //if DP has already been visited
 for(vector<string>::iterator j = calledDPsEval.begin(); j != calledDPsEval.end(); ++j)
   {
     if(*j == dp)
       {
	  return true;
       };

   };


 return false;
};

bool DerivedGoal::visited() const
{

  return visited(getDPName());
};

bool visited(string dp)
{

 //if DP has already been visited
 for(vector<string>::iterator j = calledDPsCreate.begin(); j != calledDPsCreate.end(); ++j)
   {
     if(*j == dp)
       {
	  return true;
       };
   };


 return false;
};

void QfiedGoal::create() const
{
	if(i == qg->getVars()->end())
	{

	props.push_back(vld->pf.buildProposition(qg->getGoal(),*(env->copy(vld)),createLiterals,&vld->getState()));


	}
	else
	{
	  //cout << " Processing " << (*i)->getName() << "\\\\\n";
		vector<const_symbol *> vals = vld->range(*i);
		const var_symbol * v = *i;

		++i;
		for(vector<const_symbol*>::iterator j = vals.begin();j != vals.end();++j)
		{
		  //cout << " considering value " << (*j)->getName() << "\\\\\n";
			(*env)[v] = *j;
			create();
		};
		if(i == qg->getVars()->end())
		{
			if(qg->getQuantifier()==E_FORALL)
			{
				pp = new ConjGoal(0,props,bindings);
			}
			else
			{
				pp = new DisjGoal(0,props,bindings);
			};
		};
		--i;
	};

};

void QfiedGoal::deletepp() const
{
  props.clear();


  delete pp;
  pp = 0;

  i = qg->getVars()->begin();
};


bool QfiedGoal::evaluate(const State * s,vector<const DerivedGoal*> DPs) const
{
 /* if(!pp) create();

  bool ans = pp->evaluate(s,DPs);

  deletepp();

  return ans;*/
  return evaluateQfiedGoal(s,DPs);
};

//create the Conjuncts/Disjuncts one at a time and evaluate as we go along
bool QfiedGoal::evaluateQfiedGoal(const State * s,vector<const DerivedGoal*> DPs) const
{
     bool disjConjEval;
     //get list of instansiated parameters for the qfied prop, so for forall(?z ?y), we have a grounded list for every z? and ?y
     set<var_symbol*> svs = getVariables(qg);
     vector<const_symbol_list*> constantsList = defineUndefinedParameters(newBlankConstSymbolList(const_cast<var_symbol_list*>(qg->getVars()),vld),const_cast<var_symbol_list*>(qg->getVars()),vld,svs);
     if(constantsList.empty()) return qg->getQuantifier()==E_FORALL;
     //now create a conjunction or disjunction with the qfied variables substituted
       //map<parameter_symbol*,parameter_symbol*> newvars;
       Environment & env = const_cast<Environment &>(bindings);

       for(vector<const_symbol_list*>::iterator k = constantsList.begin(); k != constantsList.end(); ++k)
       {
          //const goal * aGoal = copyGoal(qg->getGoal());

          //define mapping of parameter symbol to constant
          const_symbol_list::iterator consList = (*k)->begin();
          for(var_symbol_list::const_iterator i = qg->getVars()->begin(); i != qg->getVars()->end(); ++i)
          {
             //newvars
             env[const_cast<var_symbol*>(*i)] = *consList;
             consList++;
           };
          //evaluate conjunct/disjunct with qfied variables substituted
          //changeVars(const_cast<goal *>(aGoal),newvars);
          //cout << "Using " << *(qg->getGoal()) << "\n";
          const Proposition * propToCheck = vld->pf.buildProposition(qg->getGoal() /* aGoal */,bindings,false,s); //cout << *propToCheck <<" \n";
          //cout << "Here it is: " << *propToCheck << "\n";
          disjConjEval = propToCheck->evaluate(s,DPs);
          delete propToCheck;
          //delete aGoal;

          if(qg->getQuantifier()==E_FORALL && !disjConjEval) {deleteParameters(constantsList); return false;}
          else if(qg->getQuantifier()!=E_FORALL && disjConjEval) {deleteParameters(constantsList); return true;};
       };

     deleteParameters(constantsList);

    return (qg->getQuantifier()==E_FORALL);
};

bool QfiedGoal::markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const
{
  if(!pp) create();

  bool ans = pp->markOwnedPreconditions(a,o,w);

  deletepp();

  return ans;
};

bool FalseProposition::markOwnedPreconditions(const Action * a,Ownership & o,ownership w) const
{
  return true;
};

const Environment buildBindings(const var_symbol_table * vst,const simple_goal * sg,const Environment &bs)
{
	//create new bindings from the subset of bindings that a derived predicate has based on the actions bindings

	Environment bindings;

	var_symbol_table::const_iterator j = vst->begin();

	for(parameter_symbol_list::const_iterator i = sg->getProp()->args->begin();
				i != sg->getProp()->args->end();++i)
	{

		if(dynamic_cast<const var_symbol *>(*i))
		{
		  	bindings[j->second] = bs.find(dynamic_cast<const var_symbol*>(*i))->second;

		}
		else
		{
		  bindings[j->second] = dynamic_cast<const const_symbol*>(*i);
		};

		++j;
	};


//Environment * e = new Environment(bindings);

	return bindings;
};

const Proposition *
PropositionFactory::buildProposition(const goal * g,const Environment &bs,bool buildNewLiterals,const State * state)
{
	if(const comparison * cmp = dynamic_cast<const comparison*>(g))
	{
		return new Comparison(cmp,bs);
	};

	if(const conj_goal * cg = dynamic_cast<const conj_goal *>(g))
	{
		vector<const Proposition*> gs;
		gs.reserve(cg->getGoals()->size());
		std::transform(cg->getGoals()->begin(),
			cg->getGoals()->end(),std::back_inserter(gs),
						buildProp(this,bs,buildNewLiterals,state));

		return new ConjGoal(cg,gs,bs);
	};


	if(const disj_goal * dg = dynamic_cast<const disj_goal*>(g))
	{
		vector<const Proposition*> gs;
		gs.reserve(dg->getGoals()->size());
		std::transform(dg->getGoals()->begin(),
			dg->getGoals()->end(),std::back_inserter(gs),
						buildProp(this,bs,buildNewLiterals,state));
		return new DisjGoal(dg,gs,bs);
	};

	if(const neg_goal * ng = dynamic_cast<const neg_goal *>(g))
	{
		return new NegGoal(ng,buildProposition(ng->getGoal(),bs,buildNewLiterals,state),bs);
	};


	if(const imply_goal * ig = dynamic_cast<const imply_goal*>(g))
	{
		return new ImplyGoal(ig,buildProposition(ig->getAntecedent(),bs,buildNewLiterals,state),
								buildProposition(ig->getConsequent(),bs,buildNewLiterals,state),bs);
	};

	if(const simple_goal * sg = dynamic_cast<const simple_goal*>(g))
	{
		const goal * dg = 0;
		const var_symbol_table * vst = 0;
		map<string,pair<const goal *,const var_symbol_table *> > derivPreds = vld->getDerivRules()->getDerivPreds();


		//check to see if simple goal is in fact a derived predicate

		for(map<string,pair<const goal *,const var_symbol_table *> >::const_iterator i = derivPreds.begin(); i != derivPreds.end(); ++i)
		{

			if(sg->getProp()->head->getName() == i->first)
			{
				dg = i->second.first;
				vst = i->second.second;
				break;

			};

		};

		if(dg != 0)
		{
		  //check to see if derived predicate depends on itself

		       string dpName = getDPName(sg,(const_cast<Environment *>(&bs)));

                       if(visited(dpName))
			 {
            //map<string,const DerivedGoal*>::const_iterator dp = derivedPredicates.find(dpName);
           // cout << dp->second <<" = "<< dpName <<" dp for false prop\n";
			   if(sg->getPolarity()==E_POS)
			     {
			       return new FalseProposition(bs);
			     }
			   else
			     {

			       return new NegGoal(new neg_goal(new simple_goal(const_cast<proposition*>(sg->getProp()),E_POS)),new FalseProposition(bs),bs);
			     };
			 };

		       addCalledDP(dpName);

		       const Proposition * newProp;
		       const DerivedGoal * newDP  =  new DerivedGoal(sg->getProp(),buildProposition(dg,*(buildBindings(vst,sg,bs).copy(vld)),buildNewLiterals,state),bs);

			if(sg->getPolarity()==E_POS)
			{
				newProp =  newDP;
			}

			else
        {
          	newProp = new NegGoal(new neg_goal(new simple_goal(const_cast<proposition*>(sg->getProp()),E_POS)),newDP,bs);
        };


			removeCalledDP(dpName);



		       return newProp;
		};
    if(buildNewLiterals)
    {
      		if(sg->getPolarity()==E_POS)
      		{
      			return buildLiteral(sg->getProp(),bs);
      		}
      		else
      			return new NegGoal(new neg_goal(new simple_goal(const_cast<proposition*>(sg->getProp()),E_POS)),
      								buildLiteral(sg->getProp(),bs),bs);
    }
    else
    {
      if(state == 0)
      {
         BadAccessError bae;
         throw bae;
      };

        //if we do not want to create literals create true or false literals - which is state dependent of course, so must be used immediatley
        bool literalisTrue = false; //closed world assumption
        if(sg->getProp()->head->getName() == "=")
        {
           literalisTrue = evaluateEquality(sg->getProp(),bs);
        }
        else
        {
           string s(sg->getProp()->head->getName());
        		for(parameter_symbol_list::const_iterator i = sg->getProp()->args->begin(); i != sg->getProp()->args->end();++i)
        		{
        			if(dynamic_cast<const var_symbol*>(*i)) s += bs.find(dynamic_cast<const var_symbol*>(*i))->second->getName();
        			else s += (*i)->getName();
        		};
           map<string,const SimpleProposition*>::const_iterator i1 = literals.find(s);
           if(i1 != literals.end()) literalisTrue = i1->second->evaluate(state);
       };
       if(sg->getPolarity()!=E_POS) literalisTrue = !literalisTrue;

       if(literalisTrue)
       {
          return new FalseProposition(bs,true);
       }
       else
       {
          return new FalseProposition(bs,false);
       };
    };
	};

	if(const qfied_goal * qg = dynamic_cast<const qfied_goal*>(g))
	{
		return new QfiedGoal(vld,qg,bs,buildNewLiterals);
	};

	if(const preference * p = dynamic_cast<const preference*>(g))
	{
		return new PreferenceGoal(vld,p,buildProposition(p->getGoal(),bs,buildNewLiterals,state),bs);
	};

	return 0;
};
/*
//evaluate a literal without building it!
bool PropositionFactory::evaluate(const proposition * p,const Environment & bs,const State * state) const
{
		string s(p->head->getName());
		for(parameter_symbol_list::const_iterator i = p->args->begin();
					i != p->args->end();++i)
		{
			if(dynamic_cast<const var_symbol*>(*i))
			{
				s += bs.find(dynamic_cast<const var_symbol*>(*i))->second->getName();
			}

			else
			{

				s += (*i)->getName();
			};

		};


		map<string,const SimpleProposition*>::const_iterator i1 = literals.find(s);
		if(i1 != literals.end())
			return i1->second->evaluate(state);

		return false;
	};
*/
string SimpleProposition::getParameter(int paraNo) const
{

  int parameterNo = 1;
  	for(parameter_symbol_list::const_iterator i = prop->args->begin();
				i != prop->args->end();++i)
	{
      if(parameterNo == paraNo)
      {
      		if(dynamic_cast<const var_symbol *>(*i))
      		{
      			return bindings.find(dynamic_cast<const var_symbol*>(*i))->second->getName();
      		}

      		else
      		{
      			return dynamic_cast<const const_symbol*>(*i)->getName();
      		};
      };
      ++parameterNo;
	};

   return "";
};

bool SimpleProposition::checkParametersConstantsMatch(parameter_symbol_list* psl) const
{
  map<parameter_symbol*,const_symbol*> mapping;
  const_symbol * aConst;

  parameter_symbol_list::const_iterator ps = psl->begin();   //from event
  	for(parameter_symbol_list::const_iterator i = prop->args->begin(); //from logstate
  				i != prop->args->end();++i)
  {

     if(const var_symbol * aVariable = dynamic_cast<const var_symbol *>(*i))
     {
          aConst = const_cast<const_symbol*>(bindings.find(aVariable)->second);
     }


     else
     {
          aConst = const_cast<const_symbol*>(dynamic_cast<const const_symbol*>(*i));
     };

     map<parameter_symbol*,const_symbol*>::const_iterator cs = mapping.find(*ps);
     if(cs != mapping.end())
     {
       if(cs->second != aConst) return false;
     }
     else
     {
      if(dynamic_cast<const const_symbol*>(*ps) && *ps != aConst) return false;
      mapping[*ps] = aConst;
     };

     ++ps;
  };

  return true;
};

bool SimpleProposition::checkConstantsMatch(parameter_symbol_list* psl) const
{
  const_symbol * aConst;

  parameter_symbol_list::const_iterator ps = psl->begin();   //from event precondition
  	for(parameter_symbol_list::const_iterator i = prop->args->begin(); //from logstate
  				i != prop->args->end();++i)
  {
     if(dynamic_cast<const const_symbol*>(*ps))
     {
       if(const var_symbol * aVariable = dynamic_cast<const var_symbol *>(*i))
       {
            aConst = const_cast<const_symbol*>(bindings.find(aVariable)->second);
       }

       else
       {
            aConst = const_cast<const_symbol*>(dynamic_cast<const const_symbol*>(*i));
       };

       if(*ps != aConst) return false;
     };

     ++ps;
  };

  return true;
};

const_symbol_list * SimpleProposition::getConstants(var_symbol_list* variables,parameter_symbol_list* psl,Validator * vld) const
{
  const_symbol_list * theConstants = new const_symbol_list();
  bool variableFound;




  for(var_symbol_list::const_iterator vs = variables->begin(); vs != variables->end(); ++vs)
  {
      variableFound = false;
      parameter_symbol_list::const_iterator ps = psl->begin();   //from event
    	for(parameter_symbol_list::const_iterator i = prop->args->begin(); //from logstate
    				(i != prop->args->end() && !variableFound) ;++i)
    	{
        if(*ps == *vs)
        {
        		if(const var_symbol * aVariable = dynamic_cast<const var_symbol *>(*i))
        		{
               theConstants->push_back(const_cast<const_symbol*>(bindings.find(aVariable)->second));
        		}
            else
            {
               theConstants->push_back(const_cast<const_symbol*>(dynamic_cast<const const_symbol*>(*i)));
            };
            variableFound = true;

         };
        ++ps;
    	};

      if(!variableFound)
      {
      	theConstants->push_back(0); //vld->getAnalysis()->const_tab.symbol_get(""));
	  };
  };


  return theConstants;

};

void SimpleProposition::write(ostream & o) const
{
	o << toString();
};

string SimpleProposition::toString() const
{
	string propName = "(" + prop->head->getName();

	for(parameter_symbol_list::const_iterator i = prop->args->begin();
				i != prop->args->end();++i)
	{
		if(dynamic_cast<const var_symbol *>(*i))
		{
			propName += " " + bindings.find(dynamic_cast<const var_symbol*>(*i))->second->getName();
		}

		else
		{
			propName += " " + dynamic_cast<const const_symbol*>(*i)->getName();
		};
	};

	if(LaTeX) latexString(propName);

	propName += ")";
	return propName;
};

void DerivedGoal::write(ostream & o) const
{
	string propName = "(" + prop->head->getName();


	for(parameter_symbol_list::const_iterator i = prop->args->begin();
				i != prop->args->end();++i)
	{
		if(dynamic_cast<const var_symbol *>(*i))
		{
			propName += " " + bindings.find(dynamic_cast<const var_symbol*>(*i))->second->getName();
		}
		else

		{
			propName += " " + dynamic_cast<const const_symbol*>(*i)->getName();
		};

	};


	if(LaTeX) latexString(propName);

	o << propName << ")";
};

void ConjGoal::write(ostream & o) const
{
  string propName = "(";
  	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)

	  {
	    propName += toString(*i);
	    if((i+1) != gs.end()) propName += " AND ";

	  };
	o<< propName + ")";
};

void DisjGoal::write(ostream & o) const
{

  string propName = "(";
  	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)

	  {
	    propName += toString(*i);
	    if((i+1) != gs.end()) propName += " OR ";
	  };
	o<< propName + ")";
};

void ImplyGoal::write(ostream & o) const
{

  o << "(" <<*ant << " IMPLIES " << *cons <<")";

};

void NegGoal::write(ostream & o) const
{
  o << "(NOT " <<*p << ")";

};

void QfiedGoal::write(ostream & o) const
{
/*	switch(qg->getQuantifier())
	{
		case E_FORALL:
			o << "(forall (";
			break;
		default:
			o << "(exists (";
			break;
	};
	for(var_symbol_list::const_iterator i = qf->getVars()->begin();i != qf->getVars()->end();++i)
	{
	*/
	auto_ptr<WriteController> w(parse_category::recoverWriteController());
	auto_ptr<WriteController> p(new PrettyPrinter());
	parse_category::setWriteController(p);
  o<< *qg << "\n";
  	parse_category::setWriteController(w);
  //"(QfiedGoal)";
};

void PreferenceGoal::write(ostream & o) const
{
	o << "(preference " << pref->getName() << " " << *thePref << ")";
};

string Comparison::getExprnString(const expression * e,const Environment & bs) const
{
	if(dynamic_cast<const div_expression *>(e))
	{
		return "("+getExprnString(dynamic_cast<const div_expression*>(e)->getLHS(),bs) + " / " +
				getExprnString(dynamic_cast<const div_expression*>(e)->getRHS(),bs)+")";
	};

	if(dynamic_cast<const minus_expression *>(e))
	{
		return "("+getExprnString(dynamic_cast<const minus_expression*>(e)->getLHS(),bs) +" - " +
				getExprnString(dynamic_cast<const minus_expression*>(e)->getRHS(),bs)+")";
	};

	if(dynamic_cast<const mul_expression *>(e))
	{
     if(LaTeX) return "("+getExprnString(dynamic_cast<const mul_expression*>(e)->getLHS(),bs) + " $\\times$ "+
				getExprnString(dynamic_cast<const mul_expression*>(e)->getRHS(),bs)+")";

		return "("+getExprnString(dynamic_cast<const mul_expression*>(e)->getLHS(),bs) + " * "+
				getExprnString(dynamic_cast<const mul_expression*>(e)->getRHS(),bs)+")";
	};

	if(dynamic_cast<const plus_expression *>(e))
	{
		return "("+getExprnString(dynamic_cast<const plus_expression*>(e)->getLHS(),bs) + " + "  +
				getExprnString(dynamic_cast<const plus_expression*>(e)->getRHS(),bs)+")";
	};

	if(dynamic_cast<const num_expression*>(e))
	{
		return toString(dynamic_cast<const num_expression*>(e)->double_value());
	};

	if(dynamic_cast<const uminus_expression*>(e))
	{
		return " - "+ getExprnString(dynamic_cast<const uminus_expression*>(e)->getExpr(),bs);
	};

	if(const func_term * ft = dynamic_cast<const func_term*>(e))
	{
		string s = "(";
		s += ft->getFunction()->getName();
		for(parameter_symbol_list::const_iterator i = ft->getArgs()->begin();
				i != ft->getArgs()->end();++i)
		{
			s += " ";
			if(const var_symbol * v = dynamic_cast<const var_symbol *>(*i))
			{
				s += bs.find(v)->second->getName();
			}
			else s += (*i)->getName();
		};
		s += ")";


      if(LaTeX) return "\\exprn{"+ s + "}";
		return s;

	};

	if(const special_val_expr * sp = dynamic_cast<const special_val_expr *>(e))
	{
		if(sp->getKind() == E_TOTAL_TIME)
		{
				return "total-time";
		};

		if(sp->getKind() == E_DURATION_VAR)
		{
			return "?duration";
		};

		if(sp->getKind() == E_HASHT)
		{
			if(LaTeX)
				return "\\#t";
			else return "#t";
		}
	};
	return "?";
};

string Comparison::getExprnString(const expression * e,const Environment & bs, const State * s) const
{
	if(dynamic_cast<const div_expression *>(e))
	{
		return "("+getExprnString(dynamic_cast<const div_expression*>(e)->getLHS(),bs,s) + " / " +
				getExprnString(dynamic_cast<const div_expression*>(e)->getRHS(),bs,s)+")";
	};

	if(dynamic_cast<const minus_expression *>(e))
	{

		return "("+getExprnString(dynamic_cast<const minus_expression*>(e)->getLHS(),bs,s) +" - " +
				getExprnString(dynamic_cast<const minus_expression*>(e)->getRHS(),bs,s)+")";
	};

	if(dynamic_cast<const mul_expression *>(e))
	{
     if(LaTeX) return "("+getExprnString(dynamic_cast<const mul_expression*>(e)->getLHS(),bs,s) + " $\\times$ "+
				getExprnString(dynamic_cast<const mul_expression*>(e)->getRHS(),bs,s)+")";

		return "("+getExprnString(dynamic_cast<const mul_expression*>(e)->getLHS(),bs,s) + " * "+
				getExprnString(dynamic_cast<const mul_expression*>(e)->getRHS(),bs,s)+")";
	};

	if(dynamic_cast<const plus_expression *>(e))
	{
		return "("+getExprnString(dynamic_cast<const plus_expression*>(e)->getLHS(),bs,s) + " + "  +
				getExprnString(dynamic_cast<const plus_expression*>(e)->getRHS(),bs,s)+")";
	};

	if(dynamic_cast<const num_expression*>(e))
	{
		return toString(dynamic_cast<const num_expression*>(e)->double_value());
	};

	if(dynamic_cast<const uminus_expression*>(e))
	{

		return " - "+ getExprnString(dynamic_cast<const uminus_expression*>(e)->getExpr(),bs,s);
	};

	if(dynamic_cast<const func_term*>(e))
	{
		const FuncExp * fexp = s->getValidator()->fef.buildFuncExp(dynamic_cast<const func_term*>(e),bs);

      if(LaTeX) return "\\exprn{"+ toString(fexp)  + "}$[=" + toString(fexp->evaluate(s)) + "]$";
		return toString(fexp) + "[=" + toString(fexp->evaluate(s)) + "]";

	};

	if(const special_val_expr * sp = dynamic_cast<const special_val_expr *>(e))
	{
		if(sp->getKind() == E_TOTAL_TIME)
		{
				if(s->getValidator()->durativePlan()) return toString(s->getTime());
				return toString(s->getValidator()->simpleLength());
		};

		if(sp->getKind() == E_DURATION_VAR)
		{
			return toString(bs.duration);
		};

		if(sp->getKind() == E_HASHT)
		{
			if(LaTeX)
				*report << "The use of \\#t is not valid in this context!\n";
			else if(Verbose)
				cout << "The use of #t is not valid in this context!\n";
			SyntaxTooComplex stc;
			throw stc;
		}
	};


	UnrecognisedCondition uc;
	throw uc;
};

string Comparison::getPropAdviceString(const State* s) const
{

	string ans;
	string op;
	switch(comp->getOp())
	{


		case E_GREATER:
			op = ">";
			break;
		case E_GREATEQ:

			if(LaTeX || LaTeXRecord)
				op = "\\geq";
			else
				op = ">=";
			break;
		case E_LESS:
			op = "<";
			break;
		case E_LESSEQ:
			if(LaTeX || LaTeXRecord)
				op = "\\leq";

			else
				op = "<=";
			break;
		case E_EQUALS:
			op = "=";

	};


    	if(LaTeX || LaTeXRecord)
    		{  bool lx = LaTeX; LaTeX = true;
           ans = getExprnString(comp->getLHS(),bindings,s)+" $"+op+"$ "+getExprnString(comp->getRHS(),bindings,s);
           LaTeX = lx;
           //latexString(ans); // It's already been latexed!
          }
    	else
    		ans = getExprnString(comp->getLHS(),bindings,s)+" "+op+" "+getExprnString(comp->getRHS(),bindings,s);

	return ans;
};

const AdviceProposition * PreferenceGoal::getAdviceProp(const State * s) const
{
	return thePref->getAdviceProp(s);
};

const AdviceProposition * Proposition::getAdviceProp(const State* s) const
{
    const AdvicePropositionConj * apc = new AdvicePropositionConj();



    return apc;
};


const AdviceProposition * SimpleProposition::getAdviceProp(const State* s)  const

{
    AdvicePropositionLiteral * apl = new AdvicePropositionLiteral(false,0,false);

    if(!evaluate(s)) apl->changeAdvice(true,this,true);

    return apl;
};

const AdviceProposition * Comparison::getAdviceProp(const State* s)  const
{
    if(evaluate(s)) return new AdvicePropositionComp(false,this,"",false);
    AdvicePropositionComp * apc;

    if(ctsFtn != 0)
           apc = new AdvicePropositionComp(true,this,"Invariant "+getPropString(s)+" is only satisfied on "+toString(getIntervals(s)),false);
    else
           apc = new AdvicePropositionComp(true,this,"Satisfy "+getPropAdviceString(s),false);

    return apc;
};

const AdviceProposition * ConjGoal::getAdviceProp(const State* s)  const
{


    AdvicePropositionConj * apc = new AdvicePropositionConj();

    	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
	  {
	    if(!((*i)->evaluate(s))) apc->addAdviceProp((*i)->getAdviceProp(s));
	  };

    return apc;
};


const AdviceProposition * DisjGoal::getAdviceProp(const State* s)  const
{
    AdvicePropositionDisj * apd = new AdvicePropositionDisj();

    	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
	  {
	    apd->addAdviceProp((*i)->getAdviceProp(s));
	  };

    return apd;
};

const AdviceProposition * ImplyGoal::getAdviceProp(const State* s) const
{
    AdvicePropositionDisj * apd = new AdvicePropositionDisj();

    apd->addAdviceProp(ant->getAdviceNegProp(s));


    apd->addAdviceProp(cons->getAdviceProp(s));

    return apd;
};

const AdviceProposition * QfiedGoal::getAdviceProp(const State* s) const
{
    if(!pp) create();

   return pp->getAdviceProp(s);
};

const AdviceProposition * NegGoal::getAdviceProp(const State* s) const
{
  return p->getAdviceNegProp(s);
};


const AdviceProposition * DerivedGoal::getAdviceProp(const State* s) const
{
    AdvicePropositionDP * apdp = new AdvicePropositionDP(this, false);

    return apdp;
};

const AdviceProposition * Proposition::getAdviceNegProp(const State* s) const
{
   const AdvicePropositionConj * apc = new AdvicePropositionConj();


   return apc;
};


const AdviceProposition * SimpleProposition::getAdviceNegProp(const State* s)  const
{
    AdvicePropositionLiteral * apl = new AdvicePropositionLiteral(false,0,false);

    if(evaluate(s)) apl->changeAdvice(true,this,false);


    return apl;

};

const AdviceProposition * Comparison::getAdviceNegProp(const State* s)  const
{
    if(!evaluate(s)) return new AdvicePropositionComp(false,this,"",true);
    AdvicePropositionComp * apc;

    if(ctsFtn != 0 && !ctsFtn->isLinear())
           apc = new AdvicePropositionComp(true,this,"Do NOT satisfy invariant "+getPropString(s)+" which is satisfied on "+toString(getIntervals(s)),false);
    else
           apc = new AdvicePropositionComp(true,this,"Do NOT satisfy "+getPropAdviceString(s),false);



    return apc;
};

const AdviceProposition * ConjGoal::getAdviceNegProp(const State* s)  const
{
  AdvicePropositionDisj * apd = new AdvicePropositionDisj();

    	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)

	  {
	    apd->addAdviceProp((*i)->getAdviceNegProp(s));
	  };

    return apd;
};

const AdviceProposition * DisjGoal::getAdviceNegProp(const State* s)  const
{
    AdvicePropositionConj * apc = new AdvicePropositionConj();

    	for(vector<const Proposition *>::const_iterator i = gs.begin();i != gs.end();++i)
	  {
	    if((*i)->evaluate(s)) apc->addAdviceProp((*i)->getAdviceNegProp(s));

	  };

    return apc;
};

const AdviceProposition * ImplyGoal::getAdviceNegProp(const State* s) const
{
    AdvicePropositionConj * apc = new AdvicePropositionConj();


    if(!(ant->evaluate(s))) apc->addAdviceProp(ant->getAdviceProp(s));
    if(cons->evaluate(s)) apc->addAdviceProp(cons->getAdviceNegProp(s));


    return apc;


};

const AdviceProposition * QfiedGoal::getAdviceNegProp(const State* s) const
{
    if(!pp) create();

   return pp->getAdviceNegProp(s);
};


const AdviceProposition * NegGoal::getAdviceNegProp(const State* s) const
{
    return p->getAdviceProp(s);
};

const AdviceProposition * DerivedGoal::getAdviceNegProp(const State* s) const
{
    AdvicePropositionDP * apdp = new AdvicePropositionDP(this, true);

    return apdp;
};

AdvicePropositionConj::~AdvicePropositionConj()
{
    for(vector<const AdviceProposition *>::iterator i = adviceProps.begin(); i != adviceProps.end() ; ++i)
    {
        delete (*i);
    };


};

AdvicePropositionDisj::~AdvicePropositionDisj()
{
    for(vector<const AdviceProposition *>::iterator i = adviceProps.begin(); i != adviceProps.end() ; ++i)
    {
        delete (*i);
    };

};

void displayIndent(int indent)
{
   if(indent == 0) return;
   for(int i = 1; i <= indent; ++i)  *report << " ";

};

void AdvicePropositionConj::display(int indent) const
{

  if(adviceProps.size() == 0)
      {*report << "(No advice for conjunction!)\n"; return;}
  else if(adviceProps.size() == 1)
      {
          (*adviceProps.begin())->display(indent);
          return;
      };

  *report << "(Follow each of:\n";


  for(vector<const AdviceProposition *>::const_iterator i = adviceProps.begin(); i != adviceProps.end(); ++i)
  {
      if( i != adviceProps.begin())

          {
            displayIndent(indent+4);
            *report << "and ";
            if(dynamic_cast<const AdvicePropositionConj*>(*i) || dynamic_cast<const AdvicePropositionDisj*>(*i)) {*report <<"\n";  displayIndent(indent+4);};
          }
      else
          displayIndent(indent+4);

      (*i)->display(indent+4);


  };


  displayIndent(indent); *report << ")\n";
};


void AdvicePropositionDisj::display(int indent) const
{

  if(adviceProps.size() == 0)
      {*report << "(No advice for disjunction!)\n"; return;}
   else if(adviceProps.size() == 1)
      {
          (*adviceProps.begin())->display(indent);
          return;
      };


  *report << "(Follow one of:\n";


  for(vector<const AdviceProposition *>::const_iterator i = adviceProps.begin(); i != adviceProps.end(); ++i)
  {

      if( i != adviceProps.begin() )
          {
            displayIndent(indent+4);
            *report << "or ";
            if(dynamic_cast<const AdvicePropositionConj*>(*i) || dynamic_cast<const AdvicePropositionDisj*>(*i)) {*report <<"\n";  displayIndent(indent+4);};
            }
      else
          displayIndent(indent+4);

      (*i)->display(indent+4);

  };

  displayIndent(indent);
  *report << ")\n";

};

void AdvicePropositionLiteral::display(int indent) const
{
  if(!thereIsAdvice)
      {*report << "(No advice for literal!)\n"; return;};

  *report << "(Set " << *sp << " to ";
  if(advice == 1) *report << "true"; else *report << "false";
  *report <<")\n";


};

void AdvicePropositionDP::display(int indent) const
{
    *report << "(Satisfy derived predicate ";
    if(neg) *report << "(NOT ";
    *report << *dp;
    if(neg) *report << ")";
    *report << "!)\n";
};

void AdvicePropositionComp::display(int indent) const
{
  if(!thereIsAdvice)
      {*report << "(No advice comparison!)\n"; return;};

  if(neg) *report << "(NOT "; else *report << "(";
  *report << advice;
  *report << ")";
  *report << "\n";
};

void AdvicePropositionConj::displayLaTeX(int depth) const
{
   bool itemizing = true;
   if(depth > 3) itemizing = false;

  if(adviceProps.size() == 0)

      {*report << "No advice for conjunction!\n"; return;}
  else if(adviceProps.size() == 1)
      {
          (*adviceProps.begin())->displayLaTeX(depth);
          return;
      };
  if(!itemizing) *report << "(";
  *report << "Follow each of:\n";
    if(!itemizing) *report << "\\\\";
   else *report << "\\begin{itemize}";


  for(vector<const AdviceProposition *>::const_iterator i = adviceProps.begin(); i != adviceProps.end(); ++i)
  {
      if(itemizing) *report << "\\item ";
      (*i)->displayLaTeX(depth+1);
      if(!itemizing && i+1 != adviceProps.end()) *report << " {\\em and}\\\\ ";
  };

  if(!itemizing) *report << ")";
  else *report << "\\end{itemize}";

};


void AdvicePropositionDisj::displayLaTeX(int depth) const
{
  bool itemizing = true;
   if(depth > 3) itemizing = false;

  if(adviceProps.size() == 0)
      {*report << "No advice for disjunction!\n"; return;}
  else if(adviceProps.size() == 1)
      {
          (*adviceProps.begin())->displayLaTeX(depth);
          return;
      };

    if(!itemizing) *report << "(";
  *report << "Follow one of:\n";
   if(!itemizing) *report << "\\\\";
   else *report << "\\begin{itemize}";

  for(vector<const AdviceProposition *>::const_iterator i = adviceProps.begin(); i != adviceProps.end(); ++i)
  {
      if(itemizing) *report << "\\item ";
      (*i)->displayLaTeX(depth+1);
      if(!itemizing && i+1 != adviceProps.end()) *report << " {\\em or}\\\\ ";
  };

   if(!itemizing) *report << ")";
   else *report << "\\end{itemize}";
};

void AdvicePropositionLiteral::displayLaTeX(int depth) const
{
  if(!thereIsAdvice)
      {*report << "No advice for literal!\n"; return;};

  *report << "Set \\exprn{" << *sp << "} to ";
  if(advice == 1) *report << "true"; else *report << "false";
  *report <<"\n";

};


void AdvicePropositionDP::displayLaTeX(int depth) const
{
    *report << "Satisfy derived predicate ";

    if(neg) *report << "($\\neg$ ";
    *report << "\\exprn{" << *dp << "}";
    if(neg) *report << ")";
    *report << "!\n";

};

void AdvicePropositionComp::displayLaTeX(int depth) const
{
  if(!thereIsAdvice)
      {*report << "No advice comparison!\n"; return;};

  if(neg) *report << "(NOT ";
  *report << advice;
  if(neg) *report << ")";
  *report << "\n";
};

};
