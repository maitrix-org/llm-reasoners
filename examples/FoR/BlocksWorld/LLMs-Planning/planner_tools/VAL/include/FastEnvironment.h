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

#ifndef __FASTENV
#define __FASTENV

#include <vector>
using std::vector;

#include <list>
#include <iterator>

#include "ptree.h"

namespace VAL {

template<class U>
class IDsymbol : public U {
private:
	int symId;
public:
	IDsymbol(const string & nm,int Id) : U(nm), symId(Id) {};
	int getId() const {return symId;};
};

template<class T,class U = T>
class IDSymbolFactory : public SymbolFactory<T> {
private:
	static int cnt;
	int symId;
public:
	IDSymbolFactory() : symId(0) {cnt = 0;};
	IDSymbolFactory(int n) : symId(n) {};
	T * build(const string & nm) { ++cnt;return new IDsymbol<U>(nm,symId++);};
	int numSyms() const {return symId;};
	static int getCount() {return cnt;};
};

template<class T,class U>
int IDSymbolFactory<T,U>::cnt = 0;

class id_var_symbol_table : public var_symbol_table {
private:
	IDSymbolFactory<var_symbol> * symFac;
public:
	id_var_symbol_table() : symFac(new IDSymbolFactory<var_symbol>())
	{
		setFactory(symFac);
	};
	id_var_symbol_table(id_var_symbol_table * i) : 
		symFac(new IDSymbolFactory<var_symbol>(IDSymbolFactory<var_symbol>::getCount()))
	{
		setFactory(symFac);
	};
	int numSyms() const {return symFac->numSyms();};
};

class IDopTabFactory : public VarTabFactory {
private:
	id_var_symbol_table * idv;
public:
	var_symbol_table * buildOpTab()
	{
		idv = new id_var_symbol_table;
		return idv;
	};
	var_symbol_table * buildRuleTab()
	{
		idv = new id_var_symbol_table;
		return idv;
	};
	var_symbol_table * buildExistsTab()
	{
		return new id_var_symbol_table(idv);
	};
	var_symbol_table * buildForallTab()
	{
		return new id_var_symbol_table(idv);
	};
};

class FastEnvironment {
private:
	vector<const_symbol *> syms;
public:
	FastEnvironment(int x) : syms(x,static_cast<const_symbol*>(0)) {};
	FastEnvironment(const FastEnvironment & other) : syms(other.syms) {};

	void extend(int x)
	{
		syms.resize(syms.size()+x,static_cast<const_symbol*>(0));
	};
	
	FastEnvironment * copy() const
	{
		return new FastEnvironment(*this);
	};

	const_symbol * operator[](const symbol * s) const
	{
		if(const const_symbol * c = dynamic_cast<const const_symbol*>(s))
		{
			return const_cast<const_symbol *>(c);
		};
		return syms[static_cast<const IDsymbol<var_symbol>*>(s)->getId()];
	};

	const_symbol * & operator[](const symbol * s) 
	{
		static const_symbol * c;
		if((c = const_cast<const_symbol *>(dynamic_cast<const const_symbol*>(s))))
		{
			return c;
		};
		return syms[static_cast<const IDsymbol<var_symbol>*>(s)->getId()];
	};

	typedef vector<const_symbol *>::const_iterator const_iterator;
	const_iterator begin() const {return syms.begin();};
	const_iterator end() const {return syms.end();};
	const vector<const_symbol *> & getCore() const {return syms;};
};




template<class TI>
class LiteralParameterIterator {
private:
	FastEnvironment * env;
	TI pi;
	
public:
	LiteralParameterIterator(FastEnvironment * f,TI p) :
		env(f), pi(p) {};
	
	const_symbol * operator*()
	{
		return (*env)[*pi];
	};

	LiteralParameterIterator & operator++() 
	{
		++pi;
		return *this;
	};

	bool operator==(const LiteralParameterIterator<TI> & li) const
	{
		return pi==li.pi;
	};

	bool operator!=(const LiteralParameterIterator<TI> & li) const
	{
		return pi!=li.pi;
	};

	
	LiteralParameterIterator<TI> operator+(int x);
};

template<typename TI,typename T>
struct plusIt {
	LiteralParameterIterator<TI> operator()(TI pi,FastEnvironment * env,int x) const
	{
		for(int c = 0;c < x;++c,++pi);
		return LiteralParameterIterator<TI>(env,pi);
	};
};

template<typename TI>
struct plusIt<TI,std::random_access_iterator_tag> {
	LiteralParameterIterator<TI> operator()(TI pi,FastEnvironment * env,int x) const
	{
		return LiteralParameterIterator<TI>(env,pi+x);
	};
};

template<typename TI>
LiteralParameterIterator<TI> LiteralParameterIterator<TI>::operator+(int x)
{
	return plusIt<TI,typename std::iterator_traits<TI>::iterator_category>()(pi,env,x);
};

template<class TI>
LiteralParameterIterator<TI> makeIterator(FastEnvironment * f,TI p)
{
	return LiteralParameterIterator<TI>(f,p);
};

};

#endif
