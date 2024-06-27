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

#ifndef __TIMUTILS
#define __TIMUTILS

#include <iostream>
#include <iterator>


using std::ostream;
using std::iterator;

namespace VAL {
class pddl_type;
};

template<class T>
struct ptrwriter
{
	ostream & os;
	const char * septr;
	
	ptrwriter(ostream & o,const char * sep) : os(o), septr(sep) {};
	void operator()(T * p)
	{
		os << (*p) << septr;
	};
};

namespace TIM {

template<class TI>
struct typeTransformer : 
	public 
#ifndef OLDCOMPILER
			std::iterator
#endif
#ifdef OLDCOMPILER
			std::forward_iterator
#endif
					<typename std::iterator_traits<TI>::iterator_category,VAL::pddl_type *>{
	TI ti;
	int arg;
	const VAL::pddl_type * pt;
	int cnt;

	typeTransformer(TI t,int a,const VAL::pddl_type * p) :
		ti(t), arg(a), pt(p), cnt(0) {};

	VAL::pddl_type * operator*() 
	{
		if(cnt==arg) return const_cast<VAL::pddl_type *>(pt);
		return (*ti)->type;
	};
	typeTransformer<TI> & operator++() {++ti; ++cnt; return *this;};
	bool operator==(const typeTransformer<TI> & t) const {return ti == t.ti;};
	size_t operator-(const typeTransformer<TI> & t) const
	{
		return ti - t.ti;
	};
};

template<class TI>
typeTransformer<TI> makeTT(TI t,int a,const VAL::pddl_type * p)
{
	return typeTransformer<TI>(t,a,p);
};

};

#endif
