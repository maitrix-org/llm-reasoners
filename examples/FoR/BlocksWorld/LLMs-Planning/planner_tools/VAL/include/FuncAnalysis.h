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

#ifndef __FUNCANALYSIS
#define __FUNCANALYSIS

#include <iostream>
#include <utility>
#include <vector>
#include <set>
#include <map>
using std::vector;
using std::pair;
using std::make_pair;
using std::ostream;
using std::max;
using std::min;

namespace VAL {

class extended_func_symbol;
typedef std::map<extended_func_symbol*,std::set<extended_func_symbol*> > IGraph;

};

#include "VisitController.h"

namespace VAL {

extern bool FAverbose;


enum FValueEnum {E_POSITIVE=0,E_NEGATIVE=1,E_NONNEG=2,
			E_NONPOS=3,E_ZERO=4,E_ALL=5,E_BOUNDED=6};

class FValue {
private:
	FValueEnum fve;
	bool isConst;
public:
	FValue() : fve(E_POSITIVE), isConst(false) {};
	FValue(FValueEnum e) : fve(e), isConst(false) {};
	FValue(int e) : fve((FValueEnum)(e)), isConst(false) {};
	FValue(const FValue & f) : fve(f.fve), isConst(f.isConst) {};
	operator FValueEnum() const {return fve;};
	bool isConstant() const {return isConst;};
	int toInt() const {return fve;};
	void assertConst() {isConst = true;};
};

void operator+=(FValue & f1,FValue f2);
void operator*=(FValue & f1,FValue f2);
void operator-=(FValue & f1,FValue f2);
void operator/=(FValue & f1,FValue f2);
FValue operator-(FValue & f1); 

inline ostream & operator<<(ostream & o,FValue fv)
{
	switch(fv)
	{
		case E_POSITIVE: 
			o << "strictly positive";
			break;
		case E_NEGATIVE:
			o << "strictly negative";
			break;
		case E_NONNEG:
			o << "non-negative";
			break;
		case E_NONPOS:
			o << "non-positive";
			break;
		case E_ZERO:
			o << "initially zero";
			break;
		case E_ALL:
			o << "fluctuating";
			break;
		default:
			break;
	};
	return o;
};


class FuncGatherer : public VisitController {
private:
	extended_func_symbol * eft;
	bool cont;
public:
	FuncGatherer(extended_func_symbol * e) : eft(e), cont(false) {};

	virtual void visit_plus_expression(plus_expression * p) 
	{
		p->getLHS()->visit(this);
		p->getRHS()->visit(this);
	};
	virtual void visit_minus_expression(minus_expression * p) 
	{
		p->getLHS()->visit(this);
		p->getRHS()->visit(this);
	};
	virtual void visit_mul_expression(mul_expression * p) 
	{
		p->getLHS()->visit(this);
		p->getRHS()->visit(this);
	};
	virtual void visit_div_expression(div_expression * p) 
	{
		p->getLHS()->visit(this);
		p->getRHS()->visit(this);
	};
	virtual void visit_uminus_expression(uminus_expression * p) 
	{
		p->getExpr()->visit(this);
	};
	virtual void visit_func_term(func_term * p);
	virtual void visit_assignment(assignment * p) 
	{
		p->getExpr()->visit(this);
	};	
	virtual void visit_special_val_expr(special_val_expr * s)
	{
		cont = true;
	};
	bool isCont() const
	{
		return cont;
	};

};

typedef pair<operator_*,assignment*> Updater;
typedef vector<Updater> Updates;

class extended_func_symbol : public func_symbol {
protected:
	vector<pair<operator_*, derivation_rule*> > preconds;
	vector<assignment*> initials;
	Updates assigns;
	Updates increasers;
	Updates decreasers;
	Updates scalers;
	Updates continuous;

	vector<extended_func_symbol*> dependencies;

	bool seenPos;
	bool seenNeg;
	bool seenZero;
	bool difficultInitial;
	double top;
	double bottom;
	mutable FValue cval;
	

	int goals;
	
public:
	virtual ~extended_func_symbol() {};

	extended_func_symbol(const string & nm) : func_symbol(nm), seenPos(false),
		seenNeg(false), seenZero(false), difficultInitial(false), goals(0) {};

	void addInitial(assignment * a)
	{
		initials.push_back(a);
		const num_expression * nm = dynamic_cast<const num_expression*>((a)->getExpr());
		if(!nm)
		{
			difficultInitial = true;
			return;
		};
		double d = nm->double_value();
		if(seenPos || seenNeg || seenZero) 
		{
			top = max(top,d);
			bottom = min(bottom,d);
		}
		else
		{
			top = d;
			bottom = d;
		};
		seenPos |= (d>0);
		seenNeg |= (d<0);
		seenZero |= (d==0);
	};
	void addPre(operator_* o) 
	{
		preconds.push_back(pair<operator_*, derivation_rule*>(o,0));
	};
	void addPre(derivation_rule * o) 
	{
		preconds.push_back(pair<operator_*, derivation_rule*>(0,o));
	};
	void addAssign(operator_ * o,assignment * a) 
	{
		assigns.push_back(make_pair(o,a));
		FuncGatherer fg(this);
		a->visit(&fg);
	};
	void addIncreaser(operator_ * o,assignment * a) 
	{
		FuncGatherer fg(this);
		a->visit(&fg);
		if(fg.isCont())
		{
			continuous.push_back(make_pair(o,a));
		}
		else
		{
			increasers.push_back(make_pair(o,a));
		};
	};
	bool onlyGoingDown() {
		return assigns.empty() && increasers.empty() && !decreasers.empty() && scalers.empty()
			&& continuous.empty();
	}
	bool onlyGoingUp() {
                return assigns.empty() && !increasers.empty() && decreasers.empty() && scalers.empty()
                        && continuous.empty();
        }
	void addContinuous(operator_ * o,assignment * a)
	{
		continuous.push_back(make_pair(o,a));
	};
	void addDecreaser(operator_ * o,assignment * a) 
	{
		FuncGatherer fg(this);
		a->visit(&fg);
		if(fg.isCont())
		{
			continuous.push_back(make_pair(o,a));
		}
		else
		{
			decreasers.push_back(make_pair(o,a));
		};
	};
	void addOther(operator_ * o,assignment * a) 
	{
		scalers.push_back(make_pair(o,a));
		FuncGatherer fg(this);
		a->visit(&fg);
	};
	void addDepend(extended_func_symbol * e)
	{
		dependencies.push_back(e);
	};
	void addGoal() {++goals;};
	
	bool isStatic() const
	{
		return assigns.empty() && increasers.empty() && decreasers.empty() && scalers.empty()
					&& continuous.empty();
	};
	bool isDiscrete() const
	{
		return continuous.empty();
	};
	bool isContinuous() const
	{
		return !continuous.empty();
	};
	FValue initially() const
	{
		if(difficultInitial) 
		{
			return (cval = E_ALL);
		};
		
		if(!seenPos)
		{
			if(!seenZero) return (cval = E_NEGATIVE);
			if(seenNeg) return (cval = E_NONPOS);
			return (cval = E_ZERO);
		}
		else if(!seenNeg)
		{
			if(seenZero) return (cval = E_NONNEG);
			return (cval = E_POSITIVE);
		};
		return (cval = E_ZERO);
	};
	FValue currently() const
	{
		return cval;
	};
	void applyUpdates();
	void write(ostream & o) const
	{
		o << "Report for: " << getName() << "\n";
		o << "Preconditions:\n";
		for(vector<pair<operator_*,derivation_rule*> >::const_iterator i = preconds.begin();
				i != preconds.end();++i)
		{
			if(i->first) o << "\t" << i->first->name->getName() << "\n";
			if(i->second) o << "\t" << i->second->get_head()->head->getName() << "\n";
		};
		
		if(isStatic()) 
		{
			o << "Fluent is static\n";
		}
		else if(assigns.empty() && scalers.empty())
		{
			if(increasers.empty() || decreasers.empty())
			{
				o << "Could be a one way changing value\n";
			}
			else
			{
				o << "Seems to be an additive tracking quantity\n";
			};
		
		
			o << "Assigns:\n";
			for(Updates::const_iterator i = assigns.begin();
					i != assigns.end();++i)
			{
				if(i->first) o << "\t" << i->first->name->getName() << "\n";
			};

			o << "Increasers:\n";
			for(Updates::const_iterator i = increasers.begin();
					i != increasers.end();++i)
			{
				if(i->first) o << "\t" << i->first->name->getName() << "\n";
			};

			o << "Decreasers:\n";
			for(Updates::const_iterator i = decreasers.begin();
					i != decreasers.end();++i)
			{
				if(i->first) o << "\t" << i->first->name->getName() << "\n";
			};

			o << "Scalers:\n";
			for(Updates::const_iterator i = scalers.begin();
					i != scalers.end();++i)
			{
				if(i->first) o << "\t" << i->first->name->getName() << "\n";
			};

			if(!continuous.empty())
			{
				o << "Continuous value, affected by:\n";
				for(Updates::const_iterator i = continuous.begin();
						i != continuous.end();++i)
				{
					if(i->first) o << "\t" << i->first->name->getName() << "\n";
				};
			};
		};
		o << "Initial value assignments:\n";
		for(vector<assignment*>::const_iterator i = initials.begin();
				i != initials.end();++i)
		{
			o << "\t(" << getName();
			for(parameter_symbol_list::const_iterator j = (*i)->getFTerm()->getArgs()->begin();j != (*i)->getFTerm()->getArgs()->end();++j)
			{
				o << " " << (*j)->getName();
			};
			
			o << ") = " << dynamic_cast<const num_expression*>((*i)->getExpr())->double_value() << "\n";
		};	
		if(isStatic()) 
		{
			o << "Noting";
		}
		else
		{
			o << "Dependencies:\n";
			for(vector<extended_func_symbol*>::const_iterator i = dependencies.begin();
					i != dependencies.end();++i)
			{
				o << "\t" << (*i)->getName() << "\n";
			};
			o << "Speculating";
		};
		o << " that this value is: " << initially() << "\n";
		if(goals)
		{
			o << "Value appears in " << goals << " goals\n";
		};
		
	};
	void set(FValue f) {cval = f;};
	FValue get() const {return cval;};
	void visit(VisitController * v) const
	{
		write(cout);
	};
	const vector<extended_func_symbol *> & getDeps() const
	{
		return dependencies;
	};

	template<class TI>
	pair<bool,double> getInitial(const TI t1,const TI) const
	{
		// A CascadeMap would be better.
		vector<assignment *>::const_iterator i = initials.begin();
		const vector<assignment *>::const_iterator iEnd = initials.end();

		for(;i != iEnd;++i)
		{
			const parameter_symbol_list * const argList = (*i)->getFTerm()->getArgs();
			parameter_symbol_list::const_iterator j = argList->begin();
			const parameter_symbol_list::const_iterator jEnd = argList->end();
		
			TI localt1 = t1;

			for(;j != jEnd;++j,++localt1)
			{
				if((*j)!=*localt1) break;
			};
			if(j == jEnd) 
			{
				return
					make_pair(true,dynamic_cast<const num_expression *>((*i)->getExpr())->double_value());
			};
		};
		return make_pair(false,0); // Probably better if this threw an exception
	};

	typedef vector<assignment *>::const_iterator const_iterator;
	const_iterator begin() const {return initials.begin();};
	const_iterator end() const {return initials.end();};
};

#define EFT(x) static_cast<extended_func_symbol*>(const_cast<func_symbol*>(x))

class FuncAnalysis {
private:
	typedef vector<vector<extended_func_symbol*> > Dependencies;
	Dependencies deps;

	template<class TI>
	void doExplore(std::set<func_symbol*> & explored,
					vector<extended_func_symbol*> & tsort,
					bool invert,IGraph & inverted,extended_func_symbol * fn,TI,TI);
					
	void exploreFrom(std::set<func_symbol*> & explored,
					vector<extended_func_symbol*> & tsort,
					bool invert,IGraph & inverted,func_symbol * fn);
public:
	FuncAnalysis(func_symbol_table & ftab);
					// To call after the visitor has set up the initial dependencies
					// in the extended_func_symbols.

};

class AbstractEvaluator : public VisitController {
private:
	FValue val;
public:
	AbstractEvaluator() : val(E_ALL) {};
	virtual void visit_plus_expression(plus_expression * p) 
	{
		p->getLHS()->visit(this);
		FValue val1 = val;
		p->getRHS()->visit(this);
		val += val1;
	};
	virtual void visit_minus_expression(minus_expression * p) 
	{
		p->getRHS()->visit(this);
		FValue val1 = val;
		p->getLHS()->visit(this);
		val -= val1;
	};
	virtual void visit_mul_expression(mul_expression * p) 
	{
		p->getLHS()->visit(this);
		FValue val1 = val;
		p->getRHS()->visit(this);
		val *= val1;
	};
	virtual void visit_div_expression(div_expression * p) 
	{
		p->getRHS()->visit(this);
		FValue val1 = val;
		p->getLHS()->visit(this);
		val /= val1;
	};
	virtual void visit_uminus_expression(uminus_expression * p) 
	{
		p->getExpr()->visit(this);
		val = -val;
	};
	virtual void visit_func_term(func_term * p)
	{
		val = EFT(p->getFunction())->currently();
		if(EFT(p->getFunction())->isStatic())
		{
			val.assertConst();
		};
	};
	virtual void visit_int_expression(int_expression * p)
	{
		double d = p->double_value();
		if(d < 0) 
		{
			val = E_NEGATIVE;
		}
		else if(d > 0)
		{
			val = E_POSITIVE;
		}
		else
		{
			val = E_ZERO;
		};
		val.assertConst();
	};
	virtual void visit_float_expression(float_expression * p) 
	{
		double d = p->double_value();
		if(d < 0) 
		{
			val = E_NEGATIVE;
		}
		else if(d > 0)
		{
			val = E_POSITIVE;
		}
		else
		{
			val = E_ZERO;
		};
		val.assertConst();
	};
	FValue operator()() {return val;};
};

};

#endif
