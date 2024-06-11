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

#ifndef __CASCADER
#define __CASCADER

#include <map>
#include <iostream>

using std::map;
using std::ostream;

template<class T>
class CascadeSet {
protected:
	typedef map<T,CascadeSet<T> *> Cascader;

	bool terminal;
	Cascader cascade;

	template<class TI>
	CascadeSet(TI s,TI e) : terminal(false), cascade()
	{
		insert(s,e);
	};


public:
	CascadeSet() : terminal(false), cascade() {};

	template<class TI>
	void insert(TI s,TI e)
	{
		if(s == e) 
		{
			terminal = true;
		}
		else
		{
			typename Cascader::iterator c = cascade.find(*s);
			T t(*s);
			++s;
			if(c == cascade.end())
			{
				cascade[t] = new CascadeSet(s,e);
			}
			else
			{
				c->second->insert(s,e);
			};
		};
	};
	
	template<class TI>
	bool contains(TI s,TI e)
	{
		if(s==e)
		{
			return terminal;
		};
		typename Cascader::iterator c = cascade.find(*s);
		if(c == cascade.end()) return false;
		++s;
		return c->second->contains(s,e);
	};

	void write(ostream & o) const
	{
		static int ind = 0;
		if(terminal) 
		{
			for(int x = 0;x < ind;++x) o << " ";
			o << "--X\n";
		};
		for(typename Cascader::const_iterator c = cascade.begin();c != cascade.end();++c)
		{
			for(int x = 0;x < ind;++x) o << " ";
			cwrite(c->first,o);
			o << "\n";
			++ind;
			c->second->write(o);
			--ind;
		};
	};
};


template<class T,class U>
class CascadeMap {
protected:
	typedef map<T,CascadeMap<T,U> *> Cascader;

	U * value;
	Cascader cascade;

	template<class TI>
	CascadeMap(TI s,TI e,U * u) : value(0), cascade()
	{
		insert(s,e,u);
	};


public:
	CascadeMap() : value(0), cascade() {};

	template<class TI>
	void insert(TI s,TI e,U * u)
	{
		if(s == e) 
		{
			value = u;
		}
		else
		{
			typename Cascader::iterator c = cascade.find(*s);
			T t(*s);
			++s;
			if(c == cascade.end())
			{
				cascade[t] = new CascadeMap(s,e,u);
			}
			else
			{
				c->second->insert(s,e,u);
			};
		};
	};

	template<class TI>
	U * get(TI s,TI e) const
	{
		if(s==e)
		{
			return value;
		};
		typename Cascader::const_iterator c = cascade.find(*s);
		if(c == cascade.end()) return 0;
		++s;
		return c->second->get(s,e);
	};

	template<class TI>
	U * & myGet(TI s,TI e)
	{
		static U * dummyCase = 0;
		if(s==e)
		{
			return value;
		};
		typename Cascader::const_iterator c = cascade.find(*s);
		if(c == cascade.end()) return dummyCase;
		++s;
		return c->second->myGet(s,e);
	};

	template<class TI>
	U * partialGet(TI s,TI e) const
	{
		if(s==e)
		{
			return value;
		};
		if(*s!=0)
		{
			typename Cascader::const_iterator c = cascade.find(*s);
			if(c == cascade.end()) return 0;
			++s;
			return c->second->partialGet(s,e);
		};
		++s;
		for(typename Cascader::const_iterator c = cascade.begin();c != cascade.end();++c)
		{
			U * u = c->second->partialGet(s,e);
			if(u)
			{
				return u;
			};
		};
		return 0;
	};

	template<class TI>
	U * & forceGet(TI s,TI e)
	{
		if(s == e) 
		{
			return value;
		}
		else
		{
			typename Cascader::iterator c = cascade.find(*s);
			T t(*s);
			++s;
			if(c == cascade.end())
			{
				CascadeMap * cm = cascade[t] = new CascadeMap();
				return cm->forceGet(s,e);
			}
			else
			{
				return c->second->forceGet(s,e);
			};
		};
	};

	void write(ostream & o) const
	{
		static int ind = 0;
		if(value) 
		{
			for(int x = 0;x < ind;++x) o << " ";
			o << *value << "\n";
		};
		for(typename Cascader::const_iterator c = cascade.begin();c != cascade.end();++c)
		{
			for(int x = 0;x < ind;++x) o << " ";
			cwrite(c->first,o);
			o << "\n";
			++ind;
			c->second->write(o);
			--ind;
		};
	};

	class iterator;
	friend class iterator;
	
	class iterator : public 
#ifndef OLDCOMPILER
						std::iterator
#endif
#ifdef OLDCOMPILER
						std::forward_iterator
#endif
										<U *,size_t> {
	private:
		CascadeMap * cmap;
		bool done;
		typename Cascader::iterator c;
		typename CascadeMap::iterator * cmpi;

		void buildTrail(vector<T> & ts) const
		{
			if(c==cmap->cascade.end() || !done) return;
			ts.push_back(c->first);
			cmpi->buildTrail(ts);
		};
		
	public:
		iterator(CascadeMap * cm,bool bgn) : cmap(cm), done(bgn?false:true), 
			c(bgn?cm->cascade.begin():cm->cascade.end()), cmpi(0)
		{
			if(!done && !(cmap->value))
			{
				++(*this);
			};
		};
		~iterator()
		{
			delete cmpi;
		};
		bool operator==(const CascadeMap<T,U>::iterator & cmi) const
		{
			return cmap == cmi.cmap && done == cmi.done && c==cmi.c &&
						(cmpi?(cmi.cmpi?(*cmpi == *cmi.cmpi):false):true);
		};
		bool operator!=(const CascadeMap<T,U>::iterator & cmi) const
		{
			return !(operator==(cmi));
		};
		iterator & operator++()
		{
			if(!done)
			{
				done = true;
				if(c != cmap->cascade.end())
				{
					cmpi = new iterator(c->second->begin());
				};
				return *this;
			};
			if(cmpi)
			{
				++(*cmpi);
				if(!(cmpi->cmpi))
				{
					++c;
					delete cmpi;
					if(c==cmap->cascade.end())
					{
						cmpi = 0;
					}
					else
					{
						cmpi = new iterator(c->second->begin());
					};
				};
			};
			return *this;
		};
		U * operator *() 
		{
			if(!done)
			{
				return cmap->value;
			};
			return **cmpi;
		};
		vector<T> trail() const
		{
			vector<T> ts;
			buildTrail(ts);
			return ts;
		};
	};

	iterator begin() {return iterator(this,true);};
	iterator end() {return iterator(this,false);};
		
};

#endif
