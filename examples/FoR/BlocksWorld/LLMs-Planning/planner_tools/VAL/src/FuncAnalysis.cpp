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

#include "FuncAnalysis.h"
#include <set>
#include <algorithm>
using std::set;
using std::find;

namespace VAL {

bool FAverbose = true;

bool isSigned(const FValue & f1)
{
	switch(f1)
	{
		case E_POSITIVE:
		case E_NEGATIVE:
		case E_NONPOS:
		case E_NONNEG:
			return true;
		default:
			return false;
	};
};

bool isNeg(const FValue & f1)
{
	return f1==E_NEGATIVE || f1==E_NONPOS;
};

bool isPos(const FValue & f1)
{
	return f1==E_POSITIVE || f1==E_NONNEG;
};

bool sameSign(const FValue & f1,const FValue & f2)
{
	if(f1==E_ZERO || f2==E_ZERO)
	{
		return isSigned(f1) || isSigned(f2);
	};
	if(!isSigned(f1) || !isSigned(f2)) return false;
	
	if(isNeg(f1)) return isNeg(f2);
	return isPos(f2);
};
	
FValue mostExtreme(const FValue & f1,const FValue & f2)
{
	return min(f1,f2);
};

bool nonZero(const FValue & f1)
{
	return f1 != E_ZERO && f1 != E_NONPOS  && f1 != E_NONNEG;
};

void update(FValue & f1,FValue f2)
{
	if(f1 == E_ZERO || f2 == E_ZERO)
	{
		f1 = mostExtreme(f1,f2);
		f1 = f1<2?FValue(f1.toInt()+2):f1;
	}
	else if(sameSign(f1,f2))
	{
		f1 = max(f1,f2);
	}
	else f1 = E_ALL;
};

void operator+=(FValue & f1,FValue f2)
{
	bool x = f1.isConstant() && f2.isConstant();
	if(sameSign(f1,f2))
	{

		FValue f = mostExtreme(f1,f2);
		if(f1 == E_ZERO || f2 == E_ZERO)
		{	
			f1 = f<2?FValue(f.toInt()+2):f;
		}	
	}
	else
	{
		f1 = E_ALL;
	};
	if(x) f1.assertConst();
};

void operator*=(FValue & f1,FValue f2)
{
	bool x = f1.isConstant() && f2.isConstant();
	if(sameSign(f1,f2))
	{
		if(nonZero(f1) && nonZero(f2)) 
		{
			f1 = E_POSITIVE;
			if(x) f1.assertConst();
			return;
		};
		f1 = E_NONNEG;
	}
	else if(isSigned(f1) && isSigned(f2))
	{
		if(nonZero(f1) && nonZero(f2)) 
		{
			f1 = E_NEGATIVE;
			if(x) f1.assertConst();
			return;
		};
		f1 = E_NONPOS;
	}
	else
	{
		f1 = E_ALL;
	};
	if(x) f1.assertConst();
};

void operator-=(FValue & f1,FValue f2)
{
	f1 += -f2;
};

void operator/=(FValue & f1,FValue f2)
{
	f1 *= f2;
};


FValue operator-(FValue & f1)
{
	FValue f;
	switch(f1)
	{
		case E_POSITIVE: f = E_NEGATIVE;
			break;
		case E_NEGATIVE: f = E_POSITIVE;
			break;
		case E_NONNEG: f = E_NONPOS;
			break;
		case E_NONPOS: f = E_NONNEG;
			break;
		case E_ZERO: f = E_ZERO;
			break;
		default: f = E_ALL;
	};
	if(f1.isConstant())
	{
		f.assertConst();
	};
	return f;
};



void FuncGatherer::visit_func_term(func_term * p)
{
	eft->addDepend(EFT(p->getFunction()));
};

template<class TI>
void FuncAnalysis::doExplore(set<func_symbol*> & explored,
			vector<extended_func_symbol*> & tsort,
			bool invert,IGraph & inverted,extended_func_symbol * fn,TI s,TI e)
{
	for(;s != e;++s)
	{
		if(explored.find(*s) == explored.end())
		{
//			cout << "Exploring from " << (*s)->getName() << "\n";
			if(invert) inverted[*s].insert(fn);
			explored.insert(*s);
			exploreFrom(explored,tsort,invert,inverted,*s);
		};
	};
};

void FuncAnalysis::exploreFrom(set<func_symbol*> & explored,
						vector<extended_func_symbol*> & tsort,bool invert,
						IGraph & inverted,func_symbol * fn)
{
	extended_func_symbol * efn = EFT(fn);
	if(invert)
	{
		doExplore(explored,tsort,invert,inverted,efn,
						efn->getDeps().begin(),efn->getDeps().end());
	}
	else
	{
		doExplore(explored,tsort,invert,inverted,efn,
						inverted[efn].begin(),inverted[efn].end());
	};
	tsort.push_back(efn);
};



void extended_func_symbol::applyUpdates()
{
	AbstractEvaluator ae;
	for(Updates::iterator j = assigns.begin();j != assigns.end() && cval != E_ALL;++j)
	{
		j->second->getExpr()->visit(&ae);
		if(FAverbose)
			cout << "Was " << cval << " and now assigning " << ae() << "\n";
		update(cval,ae());
	};
	for(Updates::iterator j = increasers.begin();j != increasers.end() && cval != E_ALL;++j)
	{
		j->second->getExpr()->visit(&ae);
		if(FAverbose)
			cout << "Was " << cval << " and now increasing by " << ae() << "\n";
		cval += ae();
		if(FAverbose && cval == E_ALL && find(preconds.begin(),preconds.end(),pair<operator_*,derivation_rule*>(j->first,0)) != preconds.end())
		{
			cout << "But note that a precondition applies\n";
		};
	};
	for(Updates::iterator j = decreasers.begin();j != decreasers.end() && cval != E_ALL;++j)
	{
		j->second->getExpr()->visit(&ae);
		if(FAverbose) 
			cout << "Was " << cval << " and now decreasing by " << ae() << "\n";
		cval -= ae();
		if(FAverbose && cval == E_ALL && std::find(preconds.begin(),preconds.end(),pair<operator_*,derivation_rule*>(j->first,0)) != preconds.end())
		{
			cout << "But note that a precondition applies\n";
		};
	};
	for(Updates::iterator j = scalers.begin();j != scalers.end() && cval != E_ALL;++j)
	{
		j->second->getExpr()->visit(&ae);
		if(FAverbose)
			cout << "Was " << cval << " and now scaling by " << ae() << "\n";
		cval *= ae();
		if(FAverbose && cval == E_ALL && std::find(preconds.begin(),preconds.end(),pair<operator_*,derivation_rule*>(j->first,0)) != preconds.end())
		{
			cout << "But note that a precondition applies\n";
		};
	};
};

FuncAnalysis::FuncAnalysis(func_symbol_table & ftab)
{
	set<func_symbol *> explored;
	vector<extended_func_symbol*> topsort;
	IGraph inverted;
	
//	cout << "Ready to explore....\n";

	for(func_symbol_table::iterator i = ftab.begin();i != ftab.end();++i)
	{
		if(explored.find(i->second) == explored.end())
		{
//			cout << "Exploring from " << i->first << "\n";
			explored.insert(i->second);
			exploreFrom(explored,topsort,true,inverted,i->second);
		};
	};
#ifdef VERBOSE
	cout << "Sequence is:\n";
#endif
	explored.clear();
	for(vector<extended_func_symbol*>::reverse_iterator i = topsort.rbegin();i != topsort.rend();++i)
	{
#ifdef VERBOSE
		cout << (*i)->getName() << "\n";
#endif
		if(explored.find(*i) == explored.end())
		{
			vector<extended_func_symbol*> cmpt;
#ifdef VERBOSE
			cout << "Exploring from " << (*i)->getName() << "\n";
#endif
			explored.insert(*i);
			exploreFrom(explored,cmpt,false,inverted,*i);
			deps.push_back(cmpt);
		};
	};
#ifdef VERBOSE
	cout << "Components:\n";
	for(Dependencies::iterator i = deps.begin();i != deps.end();++i)
	{
		for(vector<extended_func_symbol*>::iterator j = i->begin();j != i->end();++j)
		{
			cout << (*j)->getName() << " ";
		};
		cout << "\n";
	};
	cout << "OK, let's analyse:\n";
#endif
	for(Dependencies::iterator i = deps.begin();i != deps.end();++i)
	{
		if(i->size() > 1)
		{
			cout << "Got a cycle component - we'll have to stop here\n";
			break;
		};
		if(!(*i)[0]->isStatic() && (*i)[0]->get() != E_ALL)
		{
			if(FAverbose)
				cout << "Re-examine behaviour of " << (*i)[0]->getName() << "\n";
			(*i)[0]->applyUpdates();
			if(FAverbose)
				cout << "Final value: " << (*i)[0]->get() << "\n";
		};
	};	
};

};

