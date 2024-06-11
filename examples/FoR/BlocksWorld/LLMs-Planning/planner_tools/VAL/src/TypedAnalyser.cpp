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

#include "TypedAnalyser.h"

namespace VAL {

ostream & operator<<(ostream & o,const SimplePropStore * sp)
{
	sp->write(o);
	return o;
};

ostream & operator<<(ostream & o,const SimplePropStore & sp)
{
	sp.write(o);
	return o;
};

void cwrite(const pddl_type * p,ostream & o)
{
	o << p->getName();
};


CompoundPropStore::CompoundPropStore(int c,vector<pair<pddl_type *,vector<const pddl_type *> > > & tps,TMap & t,extended_pred_symbol * e,Associater * a) : 
		PropStore(), stores(c)
{
	int arity = tps.size();
	vector<pddl_type *> tps1;
	for(int i = 0;i < c;++i)
	{
		tps1.clear();
		TypeExtractor tex(tps,i);
		while(!(tex == TypeExtractor(arity)))
		{
			pddl_type * tt = *tex;
//			cout << "Got " << tt->getName() << "\n";
			tps1.push_back(tt);
			++tex;
		};
//		if(e->getName()=="can-carry")
//		{
//			t.write(cout);
//		};
		
		SimplePropStore * s = t.get(tps1.begin(),tps1.end());
		if(!s)
		{	
//			cout << "About to find\n";
			extended_pred_symbol * f = a->find(e,tps1.begin(),tps1.end());
			if(!f->getParent())
			{
				e->getParent()->add(f);
			};
//			cout << "Ready to record ";
//			f->writeName(cout);
//			cout << "\n";
			s = new SimplePropStore(f);
//			TypeExtractor te(tps,i);
//			cout << "building for ";
//			while(!(te == TypeExtractor(arity)))
//			{
//				cout << (*te)->getName() << " ";
//				++te;
//			};
//			cout << "\n";
			
			t.insert(tps1.begin(),tps1.end(),s);
		};
		stores[i] = s;
		records.insert(tps1.begin(),tps1.end(),s);
	};
};
	
PropStore * extended_pred_symbol::records() const
{
	if(!props)
	{
		const_cast<extended_pred_symbol*>(this)->props =
				parent->buildPropStore(const_cast<extended_pred_symbol*>(this),types.begin(),types.end());//types);
	};
	return props;
};

PropStore * extended_pred_symbol::getAt(double t) const
{
	if(timedInitials.find(t) == timedInitials.end())
	{
		extended_pred_symbol * tmp = const_cast<extended_pred_symbol *>(this);
		PropStore * nps = parent->buildPropStore(tmp,types,t);
		tmp->timedInitials[t] = nps;
		return nps;
	}
	else
	{
		return timedInitials.find(t)->second;
	};
};

vector<double> extended_pred_symbol::getTimedAchievers(Environment * f,const proposition * prop) const
{
	vector<double> tms;
	if(records()->get(f,prop))
	{
		tms.push_back(0);
	};
	for(map<double,PropStore *>::const_iterator i = timedInitials.begin();i != timedInitials.end();++i)
	{
		if(i->second->get(f,prop))
		{
			tms.push_back(i->first);
		};
	};
	return tms;
};

PropInfoFactory * PropInfoFactory::pf = 0;


auto_ptr<EPSBuilder> Associater::buildEPS(new EPSBuilder());

// Associater associates predicates with their various type-specific versions.
// So, if a predicate is overloaded to work with multiple types this will store
// a tree structured association, indexed by types, down to an extended_pred_symbol
// that is the substitute predicate symbol for the specific type sequence.
Associater * Associater::handle(proposition * p)
{	
//	cout << "A-handle " << p->head->getName() << "\n";
	Associater * a = this;
	if(p->args->empty())
	{
		LeafAssociater * lf = dynamic_cast<LeafAssociater*>(this);
		if(lf)
		{
			p->head = lf->get();
			return this;
		};
		a =0;
	};
		
	parameter_symbol_list::iterator i = p->args->begin();
	while(i != p->args->end())
	{
		pddl_type * t = (*i)->type;
//		cout << "T " << t->getName() << "\n";
			
		++i;
		Associater * aa = a->lookup(t);
		if(!aa)
		{
			if(i == p->args->end())
			{
				aa = new LeafAssociater(p->head,p);
			}
			else
			{
				aa = new NodeAssociater();
			};
			a->set(t,aa);
		};
		a = aa;
	};

	if(a) 
	{
		if (!a->get()) {
			std::cerr << "A problem has been encountered with your domain/problem file.\n";
			std::cerr << "-------------------------------------------------------------\n";
			std::cerr << "Unfortunately, a bug has been encountered in your domain and problem file,\n";
			std::cerr << "and the planner has to terminate.  The proposition:\n\n\t";
			{
				std::cerr << "(" << p->head->getName();
				if (p->args) {
					parameter_symbol_list::iterator pItr = p->args->begin();
					const parameter_symbol_list::iterator pEnd = p->args->end();
					for (; pItr != pEnd; ++pItr) {
						std::cerr << " " << (*pItr)->getName();
					}
				}
				std::cerr << ")\n\n";
			}
			std::cerr << "... cannot be mapped to any of the predicate declarations in the domain,\n";
			std::cerr << "due to having the wrong number of arguments and/or wrong argument types.\n";
			exit(0);
		} else {
			HPS(p->head)->add(a->get());
			p->head = a->get();
			return this;
		}
	}
	else
	{
		a = new LeafAssociater(p->head,p);
		if (!a->get()) {
			std::cerr << "A problem has been encountered with your domain/problem file.\n";
			std::cerr << "-------------------------------------------------------------\n";
			std::cerr << "Unfortunately, a bug has been encountered in your domain and problem file,\n";
			std::cerr << "and the planner has to terminate.  The proposition:\n\n\t";
			{
				std::cerr << "(" << p->head->getName();
				if (p->args) {
					parameter_symbol_list::iterator pItr = p->args->begin();
					const parameter_symbol_list::iterator pEnd = p->args->end();
					for (; pItr != pEnd; ++pItr) {
						std::cerr << " " << (*pItr)->getName();
					}
				}
				std::cerr << ")\n\n";
			}
			std::cerr << "... cannot be mapped to any of the predicate declarations in the domain,\n";
			std::cerr << "due to having the wrong number of arguments and/or wrong argument types.\n";
			exit(0);
		}
		if(p->args->empty()) HPS(p->head)->add(a->get());
		p->head = a->get();
		return a;
	};
};

void CompoundPropStore::notify(void(extended_pred_symbol::*f)(operator_ *,const proposition *),
									operator_ * o,const proposition * p)
{
//	cout << "Notifying compound about " << *p << "\n";
	for(vector<SimplePropStore *>::iterator i = stores.begin();i != stores.end();++i)
	{
//		cout << "?";
		extended_pred_symbol * ep = (*i)->getEP();
		if(ep) 
		{
//			cout << "Notification\n";
			(ep->*f)(o,p);
		};
	};
};

void cwrite(const parameter_symbol * p,ostream & o)
{
	o << p->getName();
};

PropInfo * SimplePropStore::partialGet(FastEnvironment * f,const proposition * prop) const
{
	return records.partialGet(makeIterator(f,prop->args->begin()),
							makeIterator(f,prop->args->end()));
};

PropInfo * CompoundPropStore::partialGet(FastEnvironment * f,const proposition * prop) const
{
	for(vector<SimplePropStore *>::const_iterator s = stores.begin();s != stores.end();++s)
	{
		PropInfo * u = (*s)->partialGet(f,prop);
		if(u) return u;
	};	
	return 0;
};

};

