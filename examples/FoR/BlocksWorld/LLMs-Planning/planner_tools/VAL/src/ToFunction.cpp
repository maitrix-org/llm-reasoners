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

#include "ToFunction.h"
#include "FastEnvironment.h"
#include "SASActions.h"
#include "instantiation.h"
#include "SimpleEval.h"

#define SASOUTPUT if(use_sasoutput)

using std::cerr;

using namespace VAL;
using namespace Inst;

namespace SAS {

bool use_sasoutput = false;

ostream & operator<<(ostream & o,const ValHolder & vh)
{
	vh.write(o);
	return o;
};

ostream & operator<<(ostream & o,const ValueStructure & vs)
{
	vs.write(o);
	return o;
};

ostream & operator << (ostream & o,const ValueElement & ve)
{
	ve.write(o);
	return o;
};

ostream & operator<<(ostream & o,const VElement & v)
{
	v.write(o);
	return o;
};

int PlaceHolder::idGen = 0;

bool findLinkedValue(ValHolder * vh,const vector<ValueElement *> & vs)
{
	PlaceHolder * ph = dynamic_cast<PlaceHolder *>(vh);
	if(!ph) return false;
	for(PlaceHolder::const_iterator i = ph->begin();i != ph->end();++i)
	{
		for(vector<ValueElement*>::const_iterator j = vs.begin();j != vs.end();++j)
		{
//			if(!(*j)->getPS()) continue;
//		cout << "Comparing " << **i << " and " << *((*j)->getPS()) << "\n";
			if(*i == (*j)->getPS()) return true;
/*			PropertyState::PSIterator p = (*i)->begin();
			for(;p != (*i)->end();++p)
			{
				PropertyState::PSIterator q = ((*j)->getPS())->begin();
				for(;q != (*j)->getPS()->end();++q)
				{
					if((*p)->equivalent(*q)) break;
				};
				if(q == (*j)->getPS()->end()) break;
			};
			if(p == (*i)->end()) 
			{
				cout << "Got a match\n";
				cout << *i << " and " << (*j)->getPS() << "\n";
				return true;
			};
*/
		};
	};
	return false;
};

vector<ValueElement *> constructValue(Range & range,TIMobjectSymbol * tob)
{
	vector<ValueElement*> vs;
	for(Range::const_iterator i = range.begin();i != range.end();++i)
	{
// Now considering a state space
		vector<VElement*> vals, tmpvals;
		const PropertyState * ps = 0;
		for(ValuesUnion::const_iterator j = i->begin();j != i->end();++j)
		{
// Now considering a state
			Values::const_iterator k = j->begin();
			ps = (*k)->getState();
			
			for(;k != j->end();++k)
			{
// Now considering a property
// If the object is not in this state, we have to move on to the next state.
// Object is probably in this state if it has the corresponding initial property.
// If it has that then we can extract the initial proposition and the arguments
// we need. If it doesn't have it, then we know it is not in this state.
				Property * wanted = (*k)->getProp();
				if(!wanted) 
				{
//				cout << "Looking\n";
					if(findLinkedValue(*k,vs))
					{
//						cout << "Adding a value\n";
						tmpvals.push_back(new PlaceHolderElement(dynamic_cast<const PlaceHolder*>(*k)));
					};
					continue;
				};

				proposition * got = tob->find(wanted);
				if(!got) break;
				if(got->args->size() == 1)
				{
					vals.push_back(new PElement(wanted));
				}
				else
				{
					int c = 0;
					for(parameter_symbol_list::iterator pr = got->args->begin();
							pr != got->args->end();++pr,++c)
					{
						if(c != wanted->aPosn())
							vals.push_back(new ObElement(TOB(*pr)));
					};
				};
			};
			if(k == j->end() && !vals.empty())
			{
// Then we found we were in this state and so we can't be in any other.
				break;
			};
			ps = 0;
		};
		if(vals.empty()) 
		{
//			cout << "Really should consider another possibility\n";
			vals = tmpvals;			
		};
		vs.push_back(new ValueElement(ps,vals));
	};
	return vs;
};

void ValueStructure::initialise()
{
	SASOUTPUT {
	cout << "Initialising for " << pt->getName() << " in range: " << *this << "\n\n"; };

	vector<VAL::const_symbol *> cs(theTC->range(pt));
	for(vector<VAL::const_symbol*>::iterator i = cs.begin();i != cs.end();++i)
	{
		if((rngs[TOB(*i)] = constructValue(range,TOB(*i))).size() > 0)
		{
			SASOUTPUT {
			cout << "Constructed an initial value for " << *TOB(*i) << ": (";
			for_each(rngs[TOB(*i)].begin(),rngs[TOB(*i)].end()-1,ptrwriter<ValueElement>(cout,","));
			cout << **(rngs[TOB(*i)].end()-1);
			cout << ")\n"; };
		}
		else
		{
			SASOUTPUT {cout << "No initial value for " << *TOB(*i) << "\n";};
		};
	};
	SASOUTPUT {cout << "\n";};
};

void FunctionStructure::initialise()
{
	for(FunctionRep::iterator i = frep.begin();i != frep.end(); ++i)
	{
		i->second.initialise();
	};
};


set<PropertySpace *> relevant(pddl_type * tp)
{
	return TA->relevant(tp);
};

void constructValues(Values & vals,const PropertyState * pst,const pddl_type * pt,PropertySpace * psp)
{
	for(PropertyState::PSIterator pp = pst->begin();pp != pst->end();++pp)
	{
		if((*pp)->familySize() == 1)
		{
			vals.push_back(new NullHolder(pt,pst,psp,*pp));
		}
		else
		{
			vector<pddl_type *> tps;
			vector<pddl_typed_symbol *>::const_iterator tti = (*pp)->root()->tcBegin();
			for(int ti = 0;ti < (*pp)->familySize();++ti,++tti)
			{
				if(ti != (*pp)->aPosn())
				{
// Caution: watch out for either types!
					tps.push_back((*tti)->type);
				};
			};
			vals.push_back(new TypesHolder(pt,pst,psp,*pp,tps));
		};
	};
};

FunctionStructure::FunctionStructure() : levels(0), othercounts(1,0)
{
	for(pddl_type_list::const_iterator i = current_analysis->the_domain->types->begin();
			i != current_analysis->the_domain->types->end();++i)
	{
		if(theTC->isLeafType(*i))
		{
			set<PropertySpace *> s = relevant(*i);
			if(!s.empty())
			{
				ValueStructure vstr(*i);
				for(set<PropertySpace*>::iterator j = s.begin();j != s.end();++j)
				{
					if((int) (theTC->range(*i).size()) > (*j)->oend() - (*j)->obegin())
					{
						cout << "This space contains only a subset of the objects of the type\n"
							<< "Suggest creating a sub-type\n";
					};

				
					ValuesUnion vu;
					for(PropertySpace::SIterator ps = (*j)->begin();ps != (*j)->end();++ps)
					{
						Values vs;
						constructValues(vs,*ps,*i,*j);
						vu.push_back(vs,*ps);
					};
					vstr.add(vu);
				};
				frep.insert(make_pair(*i,vstr));
//				cout << (*i)->getName() << " -> " << vstr << "\n";
			}
			else
			{
				noStates.push_back(*i);
//				cout << "No state for " << (*i)->getName() << "\n";
			};
		};
	};
};

int WildElement::idgen = 0;

struct getMe {
	Property * p;
	getMe(Property * pp) : p(pp) {};
	bool operator()(const pair<Property*,proposition*> & pp) 
	{
//	cout << "Seeing " << *(pp.first) << "\n";
		return p->equivalent(pp.first);

/*
 *		if(p==pp.first) return true;

		vector<Property*> ps = pp.first->matchers();
		vector<Property*>::const_iterator j = std::find(ps.begin(),ps.end(),p);
		return j != ps.end();
*/
	};
};

int countRelevant(PropertySpace * ps,const vector<pair<Property *,proposition*> > & conds)
{
	int c = 0;
	for(vector<pair<Property *,proposition*> >::const_iterator i = conds.begin();i != conds.end();++i)
	{
		c += (find(i->first->begin(),i->first->end(),ps) != i->first->end());
	};
	return c;
};

vector<ValueElement *> constructValue(const ValueStructure & vvs,
								const vector<pair<Property *,proposition*> > & conds,
								const vector<ValueElement*> & pvals)
{
	vector<ValueElement*> vs;
	for(Range::const_iterator i = vvs.getRange().begin();i != vvs.getRange().end();++i)
	{
// Now considering a state space
		vector<VElement*> vals;
		const PropertyState * ps = 0;
		const PlaceHolder * potentiallyLinked = 0;
		for(ValuesUnion::const_iterator j = i->begin();j != i->end();++j)
		{
// Now considering a state
			Values::const_iterator k = j->begin();
			ps = (*k)->getState();
//			int relevantConds = countRelevant((*k)->getSpace(),conds);
			int cnt = 0;
			
			for(;k != j->end();++k)
			{
// Now considering a property
// If the object is not in this state, we have to move on to the next state.
// Object is probably in this state if it has the corresponding initial property.
// However there is a possibility that the object is in a different state that shares
// this property.
// 
// If it has that then we can extract the initial proposition and the arguments
// we need. 
// 
// If it doesn't have it, then we can suspect that it is not in this state - but it might 
// just be incompletely specified. The key is that the object must be in some state and 
// we know that all the propositions will have to be accounted for by the state we identify.
// So, we'll count how many propositions are accounted for and then check that they were
// all handled by the end.
				Property * wanted = (*k)->getProp();
				if(!wanted) 
				{
					potentiallyLinked = dynamic_cast<const PlaceHolder*>(*k);
					if(findLinkedValue(*k,vs))
					{
//						cout << "Adding a value\n";
						++cnt;
						vals.push_back(new PlaceHolderElement(potentiallyLinked));
					};
					continue;
				};
				//cout << "I want " << *wanted << "\n";
				// If the property is not wanted then we are looking at a special value case.
				// This is not a really bad problem for preconditions, but it is much more of
				// a problem for postconditions. To test it properly we could use a trick: put the 
				// associated states before this one, so that by the time we get here we know the
				// associated state will be already stored - that way we can find it and correctly
				// set this value!
				//cout << "Want " << *wanted << "\n";
				vector<pair<Property*,proposition*> >::const_iterator fnd = 
									find_if(conds.begin(),conds.end(),getMe(wanted));
				if(fnd == conds.end()) 
				{
//					cout << "Failed to find it\n";
// Then we didn't find a match. So we put in a null and move on.
					if(pvals.size() > vs.size() && pvals[vs.size()]->size() > vals.size())
					{
						vals.push_back((*(pvals[vs.size()]))[vals.size()]->copy());
					}
					else
					{
						vals.push_back(new WildElement(*i));
					};
					continue;
				};
				proposition * got = fnd->second;
				++cnt;
				if(got->args->size() == 1)
				{
					vals.push_back(new PElement(wanted));
				}
				else
				{
					int c = 0;
					for(parameter_symbol_list::iterator pr = got->args->begin();
							pr != got->args->end();++pr,++c)
					{
						if(c != wanted->aPosn())
							vals.push_back(new VarElement(*pr));
					};
				};
			};
			//cout << "Found " << cnt << " of " << relevantConds << "\n";
			if(cnt)// == relevantConds)
			{
// Then we found we were in this state, accounting for all propositions,
// and so we can't be in any other, unless it shares the common set of propositions. We could
// carry on and see if we can find another match.
//				if(cnt < (int)conds.size()) cout << "Still some conditions to account for\n";
				break;
			}
			else
			{
				for(vector<VElement*>::const_iterator xx = vals.begin();xx != vals.end();++xx)
					delete (*xx);
				vals.clear();
			};
			ps = 0;
		};
		if(potentiallyLinked && !ps)
		{
			SASOUTPUT {cout << "Potential link for a value\n";};
		};
		vs.push_back(new ValueElement(ps,vals));
	};
	return vs;
};

class ConditionGatherer : public VisitController {
private:
	vector<vector<pair<Property*,proposition *> > > gathered;
	vector<vector<ValueElement*> > values;
	vector<proposition *> theStatics;
	vector<proposition *> others;
public:
	ConditionGatherer(int n) : gathered(n) {};
	ConditionGatherer(const ConditionGatherer & cg) : 
		gathered(cg.gathered.size()), values(cg.values), 
		theStatics(cg.theStatics), others() 
	{};
	
	virtual void visit_simple_goal(simple_goal * p) 
	{p->getProp()->visit(this);};
	virtual void visit_qfied_goal(qfied_goal * p) 
	{cout << "Cannot handle quantified preconditions yet!\n"; exit(0);};
	virtual void visit_conj_goal(conj_goal * p) 
	{p->getGoals()->visit(this);};
	virtual void visit_disj_goal(disj_goal * p) 
	{cout << "Cannot handle disjunctive preconditions yet!\n"; exit(0);};
	virtual void visit_timed_goal(timed_goal * p) 
	{cout << "CAUTION: Temporal goal\n";
	p->getGoal()->visit(this); 
	cout << "Done Temporal goal\n";};
	virtual void visit_imply_goal(imply_goal * p) 
	{cout << "Cannot handle implicative preconditions yet!\n";exit(0);};
	virtual void visit_neg_goal(neg_goal * p) 
	{cout << "Cannot handle negative preconditions yet (although should be able to manage !=)\n";};
	virtual void visit_comparison(comparison * p) 
	{cout << "No metric or special comparisons yet!\n"; exit(0);};
	virtual void visit_proposition(proposition * p) 
	{
		if(EPS(p->head)->isStatic())
		{
			theStatics.push_back(p);
			return;
		};
		if(p->args->empty())
		{
			others.push_back(p);
			return;
		};
		parameter_symbol_list::const_iterator ps = p->args->begin();
		for(unsigned int i = 0;i < p->args->size();++i,++ps)
		{
			cout << "Handle " << *(TPS(p->head)->property(i)) << "\n";
// The following assumes that a property will be single valued for all parameters (of any types)
// that may instantiate a particular position. This looks for any one single valued property 
// matching the property from the proposition (which can be for a more general type than the 
// properties that were originally marked as single valued).
			vector<Property*> ms;
			for(holding_pred_symbol::PIt pit = EPS(p->head)->getParent()->pBegin();
					pit != EPS(p->head)->getParent()->pEnd();++pit)
			{
				ms.push_back(TPS(*pit)->property(i));

			};
			for(vector<Property*>::const_iterator prp = ms.begin();prp != ms.end();++prp)
			{
			cout << "Considering " << **prp << "\n";
				if((*prp)->isSingleValued())
				{
					cout << "Think I should allocate this to parameter " << 
						(static_cast<const IDsymbol<var_symbol>*>(*ps)->getId()) << "\n";
					gathered[(static_cast<const IDsymbol<var_symbol>*>(*ps)->getId())].
									push_back(make_pair(TPS(p->head)->property(i),p));
					break;
				};
			};
		};

	};
	virtual void visit_simple_effect(simple_effect * p) 
	{
		p->prop->visit(this);
	};
	virtual void visit_effect_lists(effect_lists * p) 
	{
		for(pc_list<simple_effect*>::iterator i = p->add_effects.begin();i != p->add_effects.end();++i)
		{
			(*i)->visit(this);
		};
	};
	void collect(const operator_ * op,FunctionStructure * fs,bool stateForAll,VMap & valueFor)
	{
		vector<vector<ValueElement*> > prevalues(op->parameters->size());
		if(!values.empty()) 
		{
			prevalues.clear();
			prevalues.swap(values);
		};
		
		SASOUTPUT {cout << (stateForAll?"Precondition":"Postcondition") 
				<< " states for " << *TAS(op->name) << "\n";};
		int c = 0;
		for(var_symbol_list::const_iterator ps = op->parameters->begin();
									ps != op->parameters->end();++ps,++c)
		{
			if(!stateForAll && !TAS(op->name)->hasRuleFor(c)) continue;
//			cout << "For parameter " << (*ps)->getName() << " of type " 
//							<< (*ps)->type->getName() << "\n";
			vector<const pddl_type *> tps,tmptps;
			if(!fs->hasFluent((*ps)->type))
			{
				//values.push_back(vector<ValueElement*>());
//				cout << "This type has no state\n";
				tmptps = theTC->leaves((*ps)->type);
				for(vector<const pddl_type *>::const_iterator xx = tmptps.begin();xx != tmptps.end();++xx)
				{
					if(fs->hasFluent(*xx))
					{
//						cout << "Should consider type " << (*xx)->getName() << "\n";
						tps.push_back(*xx);
					};
				};
			}
			else
			{

				/* Generally we should ascend the entire hierarchy looking for fluents.
				*/
				const pddl_type * ttp = (*ps)->type;
				while(ttp)
				{
					if(fs->hasFluent(ttp)) 
					{
						tps.push_back(ttp);
//						cout << "Pushing type " << ttp->getName() << " for " << (*ps)->getName() << "\n";
					};
					ttp = ttp->type;
				};
				
			};
			if(tps.empty())
			{
				values.push_back(vector<ValueElement*>());
				continue;
			};
			for(vector<const pddl_type *>::const_iterator atp = tps.begin();atp != tps.end();++atp)
			{
				values.push_back(constructValue(fs->forType(*atp),
								gathered[(static_cast<const IDsymbol<var_symbol>*>(*ps)->getId())],
								prevalues[values.size()]));
				int cc = 0;
				for(vector<ValueElement*>::const_iterator ve = values[values.size()-1].begin();
						ve != values[values.size()-1].end();++ve,++cc)
				{
					if((*ve)->size() > 0)
					{
						SASOUTPUT {cout << (*atp)->getName() << " [" << cc << "] " << (*ps)->getName() << " = "
							<< **ve << "\n";};
						valueFor[*ps].push_back(new ValueRep(*atp,cc,*ve));
					};
				};
			};
		};
		SASOUTPUT {
		if(stateForAll && !theStatics.empty())
		{
			cout << "\nStatic conditions:\n";
			for(vector<proposition*>::const_iterator s = theStatics.begin();s != theStatics.end();++s)
			{
				cout << "(" << (*s)->head->getName();
				for(parameter_symbol_list::const_iterator pm = (*s)->args->begin();
						pm !=(*s)->args->end();++pm)
				{
					cout << " " << (*pm)->getName();
				};
				cout << ")\n";
			};
		};
		if(!others.empty())
		{
			cout << "Other conditions:\n";
			for(vector<proposition*>::const_iterator s = others.begin();s != others.end();++s)
			{
				cout << "(" << (*s)->head->getName();
				for(parameter_symbol_list::const_iterator pm = (*s)->args->begin();
						pm !=(*s)->args->end();++pm)
				{
					cout << " " << (*pm)->getName();
				};
				cout << ")\n";
			};
		};
		};
	};
	SASActionTemplate * completeAction(operator_ * op,const VMap & pre,const VMap & post,
											ConditionGatherer & cg) const
	{
		return new SASActionTemplate(op,pre,post,theStatics,others,cg.others);
	};
};

void FunctionStructure::processActions()
{
	for(operator_list::const_iterator i = current_analysis->the_domain->ops->begin();
				i != current_analysis->the_domain->ops->end();++i)
	{
		SASOUTPUT {cout << "===========\n" << *TAS((*i)->name) << "\n";};
/*		for(TIMactionSymbol::RCiterator r = TAS((*i)->name)->begin();r != TAS((*i)->name)->end();++r)
		{
			cout << **r << "\n";
		};
*/
		ConditionGatherer pg((*i)->parameters->size());
		(*i)->precondition->visit(&pg);
		VMap pres;
		pg.collect(*i,this,true,pres);
		
		ConditionGatherer eff(pg);
		(*i)->effects->visit(&eff);
		VMap posts;
		eff.collect(*i,this,false,posts);
		sasActionTemplates[*i] = pg.completeAction(*i,pres,posts,eff);
	};
	SASOUTPUT {cout << "\n\n\n\n";
	for(SASActionTemplates::const_iterator i = sasActionTemplates.begin();
								i != sasActionTemplates.end();++i)
	{
		cout << *(i->second) << "\n";
	};};
};

void FunctionStructure::buildLayers()
{
	SimpleEvaluator::setInitialState();
	for(operator_list::const_iterator os = current_analysis->the_domain->ops->begin();
						os != current_analysis->the_domain->ops->end();++os)
    {
    
    	cout << (*os)->name->getName() << "\n";
    	int s = instantiatedOp::howMany();
    	instantiatedOp::instantiate(*os,current_analysis->the_problem,*theTC);
    	cout << instantiatedOp::howMany() << " so far\n";
    	startOp[*os] = make_pair(s,instantiatedOp::howMany());
    };
    for(OpStore::iterator i = instantiatedOp::opsBegin();i != instantiatedOp::opsEnd();++i)
	{
		unsatisfiedPrecs.push_back(sasActionTemplates[(*i)->forOp()]->preCount());
	};
};

struct ValueStruct {
	const pddl_type * tp;
	const ValuesUnion & vu;
	const PropertyState * ps;

	ValueStruct(const pddl_type * p,const ValuesUnion & v,const PropertyState * s) :
		tp(p), vu(v), ps(s) {};

	ValueStruct & operator=(const ValueStruct & vs)
	{
		tp = vs.tp;
		const_cast<ValuesUnion &>(vu) = const_cast<ValuesUnion &>(vs.vu);
		ps = vs.ps;
		return *this;
	};
		
};

void FunctionStructure::normalise()
{
	for(operator_list::const_iterator op = current_analysis->the_domain->ops->begin();
				op != current_analysis->the_domain->ops->end();++op)
	{
		int c = 0;
		for(var_symbol_list::const_iterator ps = (*op)->parameters->begin();
									ps != (*op)->parameters->end();++ps,++c)
		{
			if(!TAS((*op)->name)->hasRuleFor(c)) continue;
//			cout << "For parameter " << (*ps)->getName() << " of type " 
//							<< (*ps)->type->getName() << "\n";
			vector<ValueStruct> toReduce,toLeave;
			const PropertyState * pst = 0;
			for(TIMactionSymbol::RCiterator r = TAS((*op)->name)->begin();r != TAS((*op)->name)->end();++r)
			{
				if((*r)->paramNum()==c)
				{
					vector<const pddl_type *> tps = theTC->leaves((*ps)->type);
					if(tps.empty()) tps.push_back((*ps)->type);
					vector<const pddl_type *> rtps;
					for(vector<const pddl_type *>::const_iterator xx = tps.begin();xx != tps.end();++xx)
					{
						if(hasFluent(*xx))
						{
							//cout << "Should consider type " << (*xx)->getName() << "\n";
							rtps.push_back(*xx);
						};
					};
					
					for(vector<const pddl_type *>::const_iterator tp = rtps.begin();tp != rtps.end();++tp)
					{
						for(Range::const_iterator vu = frep.find(*tp)->second.getRange().begin();
											vu != frep.find(*tp)->second.getRange().end();++vu)
						{
							
							ValuesUnion::const_iterator v = (*vu).begin();
							for(;v != (*vu).end();++v)
							{
								if(v.forState() && (*r)->applicableIn(v.forState()))
								{
									break;
								};
							};
							if(v != (*vu).end())
							{
								if((*r)->getLHS()->size() < v.forState()->size())
								{
//									cout << "Rule " << **r << " applies to partial state description in "
//											<< *(v.forState()) << "\n";
									//restructure((*r)->getLHS(),v.forState(),*tp);
									toReduce.push_back(ValueStruct(*tp,*vu,v.forState()));
									if(!pst) 
									{
										pst = (*r)->getLHS();
									};
								}
								else
								{
									toLeave.push_back(ValueStruct(*tp,*vu,v.forState()));
								};
							};
						};
					};
				};
			};
			if(!toReduce.empty())
			{
				restructure(toReduce,toLeave,pst);
			};
// The following block seems unnecessary because the action processing handles the multiple subtypes
// case (although there could still be unforeseen problems with that, so we'll keep this here for
// the moment!

			if(!hasFluent((*ps)->type))
			{
//				cout << "This type has no state\n";
				vector<const pddl_type *> tps = theTC->leaves((*ps)->type);
				vector<const pddl_type *> rtps;
				for(vector<const pddl_type *>::const_iterator xx = tps.begin();xx != tps.end();++xx)
				{
					if(hasFluent(*xx))
					{
						//cout << "Should consider type " << (*xx)->getName() << "\n";
						rtps.push_back(*xx);
					};
				};
				if(!rtps.empty()) 
				{
					restructure(*op,*ps,rtps);
				};
			};

		};
	};
};

bool equivalent(const Values & v1,const Values & v2)
{
	return v1.equivalent(v2);
};

struct alreadyIn {
	const ValuesUnion & vals;
	alreadyIn(const ValuesUnion & v) : vals(v) {};

	bool operator()(const pair<const PropertyState *,Values> & v) const
	{
		for(ValuesUnion::const_iterator i = vals.begin();i != vals.end();++i)
		{
			if(v.second.equivalent(*i))
			{
				return true;
			};
		}
		return false;
	};
};

ValuesUnion::ValuesUnion(const ValuesUnion & v1,const ValuesUnion & v2) : valuesUnion(v1.valuesUnion)
{
	remove_copy_if(v2.valuesUnion.begin(),v2.valuesUnion.end(),back_inserter(valuesUnion),
					alreadyIn(v1));
};

bool ValuesUnion::intersectsWith(const ValuesUnion & v) const
{
	for(ValuesUnion::const_iterator i = begin();i != end();++i)
	{
		for(ValuesUnion::const_iterator j = v.begin();j != v.end();++j)
		{
			if(equivalent(*i,*j)) return true;
		};
	};
	return false;
};

void ValueStructure::liftFrom(ValueStructure & vs1,ValueStructure & vs2)
{
	Range r1,r2;
	for(Range::const_iterator i = vs1.getRange().begin();i != vs1.getRange().end();++i)
	{
		Range::const_iterator j = vs2.getRange().begin();
		for(;j != vs2.getRange().end();++j)
		{
			if(i->intersectsWith(*j))
			{
				range.push_back(ValuesUnion(*i,*j));
				break;
			}
			else
			{
				r2.push_back(*j);
			};
		};
		if(j == vs2.getRange().end())
		{
			r1.push_back(*i);
		};
		vs2.range = r2;
	};
	vs1.range = r1;
};

void FunctionStructure::restructure(const operator_ * op,const var_symbol * prm,
											const vector<const pddl_type *> & rtps)
{

 	SASOUTPUT {cout << "Looking for shared state structure in types: ";
	for(vector<const pddl_type *>::const_iterator i = rtps.begin(); i != rtps.end(); ++i)
	{
		cout << (*i)->getName() << " ";
	};
	cout << "\n";};
	if(rtps.size() > 2) 
	{
		cerr << "Not sure how to handle so many sub-types for this abstraction process!\n"
			<< "Review: FunctionStructure::restructure in ToFunction.cpp\n";
		exit(0);
	};
	ValueStructure & vs1 = frep.find(rtps[0])->second;
	ValueStructure & vs2 = frep.find(rtps[1])->second;
	ValueStructure newvs(prm->type);
	newvs.liftFrom(vs1,vs2);
	frep.insert(make_pair(prm->type,newvs));
	if(vs1.getRange().size()==0)
	{
		frep.erase(rtps[0]);
	};
	if(vs2.getRange().size()==0)
	{
		frep.erase(rtps[1]);
	};
};

void FunctionStructure::restructure(const vector<ValueStruct> & toReduce,const vector<ValueStruct> & toLeave,
								const PropertyState * ps)
{
/*
 	for(vector<ValueStruct>::const_iterator i = toLeave.begin();i != toLeave.end();++i)
	{
		cout << "Leave " << *(i->ps) << " for " << i->tp->getName() << "\n";
	};
	for(vector<ValueStruct>::const_iterator i = toReduce.begin();i != toReduce.end();++i)
	{
		cout << "Reduce " << *(i->ps) << " for " << i->tp->getName() << "\n";
	};
*/
	vector<Property*> toSeparate;
	if(!toLeave.empty())
	{
		for(ValuesUnion::const_iterator i = toLeave[0].vu.begin();i != toLeave[0].vu.end();++i)
		{
			copy(i.forState()->begin(),i.forState()->end(),back_inserter(toSeparate));
		};
//		cout << "Closure of properties: ";
//		for_each(toSeparate.begin(),toSeparate.end(),ptrwriter<Property>(cout," "));
//		cout << "\n";
	}
	else
	{
//		cout << "Hmmm...we need to find the closure some other way!\n";
		set<Property*> props;
		copy(ps->begin(),ps->end(),inserter(props,props.begin()));
		TA->close(props,toReduce[0].tp);
		copy(props.begin(),props.end(),back_inserter(toSeparate));
//		cout << "Have found: ";
//		for_each(toSeparate.begin(),toSeparate.end(),ptrwriter<Property>(cout," "));
//		cout << "\n";
	};
	for(vector<ValueStruct>::const_iterator i = toReduce.begin();i != toReduce.end();++i)
	{
		ValuesUnion newvu1,newvu2;
		for(ValuesUnion::const_iterator j = i->vu.begin();j != i->vu.end();++j)
		{
			Values newvl1,newvl2;
			vector<Property *> prps1,prps2;
//			cout << "Want to intersect with " << *(j.forState()) << "\n";
			for(PropertyState::PSIterator p = j.forState()->begin();p != j.forState()->end();++p)
			{
			// The j state is the one we want to split. The q values are the properties we are trying
			// to separate from the j state.
				vector<Property*>::const_iterator q = toSeparate.begin();
				for(;q != toSeparate.end();++q)
				{
					//cout << "Compare " << **p << " and " << **q << "\n";
					if((*p)->equivalent(*q))
					{
//						cout << **p << " and " << **q << " match\n";
						prps2.push_back(*p);
						break;
					};
				};
				if(q == toSeparate.end())
				{
					prps1.push_back(*p);
				};
			};
			const PropertyState * newpst;
			const PropertyState * reducedPSt;
			PropertySpace * psp = (*(j->begin()))->getSpace();
			if(!prps2.empty())
			{
				newpst = PropertyState::getPS(TA,i->tp,prps2.begin(),prps2.end());
				reducedPSt = PropertyState::getPS(TA,i->tp,prps1.begin(),prps1.end());
//				cout << "Split states into: " << *newpst << " and " << *reducedPSt << "\n";
				constructValues(newvl1,newpst,i->tp,psp);
				newvu1.push_back(newvl1,newpst);
				constructValues(newvl2,reducedPSt,i->tp,psp);
				newvu2.push_back(newvl2,reducedPSt);
				
			}
			else
			{
				reducedPSt = j.forState();
				newpst = j.forState();
//				cout << "State " << *reducedPSt << " not split\n";
				newvu2.push_back(*j,j.forState());
				
				if(!newvu1.hasPlaceHolder(j.forState()))
				{
					newvl1.push_back(new PlaceHolder(i->tp,j.forState(),psp));
					newvu1.push_back(newvl1,0);
				};
			};
			
		};
//		cout << "Planning to replace original ValuesUnion " << i->vu << " with " << newvu1 << " and " 
//				<< newvu2 << "\n";
		frep.find(i->tp)->second.update(i->vu,newvu2,newvu1);
	};
};

void Range::update(const ValuesUnion & oldvu,const ValuesUnion & newvu1,const ValuesUnion & newvu2)
{
	for(unsigned int i = 0;i < size();++i)
	{
		if(&(range[i]) == &oldvu)
		{
			range[i] = newvu1;
			range.push_back(newvu2);
			return;
		};
	};

};

void ValueStructure::update(const ValuesUnion & oldvu,const ValuesUnion & newvu1,const ValuesUnion & newvu2)
{
	range.update(oldvu,newvu1,newvu2);
};

bool ValuesUnion::hasPlaceHolder(const PropertyState * ps)
{
	for(vector<pair<const PropertyState *,Values> >::iterator i = valuesUnion.begin();i != valuesUnion.end();++i)
	{
		if(dynamic_cast<PlaceHolder*>((i->second)[0]))
		{
			(i->second)[0]->add(ps);
			return true;
		};
	};
	return false;
};

void ValueStructure::setUpInitialState(Reachables & reachables)
{
	for(ElementRanges::iterator i = rngs.begin();i != rngs.end();++i)
	{
		reachables[pt][i->first] = new RangeRep(pt,i->first,i->second);
	};
};

void FunctionStructure::setUpInitialState()
{
	for(FunctionRep::iterator i = frep.begin();i != frep.end();++i)
	{
		i->second.setUpInitialState(reachables);
	};
	for(pc_list<simple_effect*>::iterator i = current_analysis->the_problem->initial_state->add_effects.begin();
				i != current_analysis->the_problem->initial_state->add_effects.end();++i)
	{
		if((*i)->prop->args->empty() && !EPS((*i)->prop->head)->isStatic())
		{
			others.push_back((*i)->prop);
		};
	};
};

bool FunctionStructure::growOneLevel()
{
	bool activated = false;
	for(Reachables::iterator i = reachables.begin();i != reachables.end();++i)
	{
		for(map<const TIMobjectSymbol *,RangeRep *>::iterator j = i->second.begin();
									j != i->second.end();++j)
		{
			j->second->cap();
		};
	};
	int last = othercounts[othercounts.size()-1];
// Record size now because we could be about to change it.
	int sz = others.size();
	othercounts.push_back(sz);
	for(int i = last;i < sz;++i)
	{
//		cout << "Handling " << others[i]->head->getName() << "\n";
		for(extended_pred_symbol::OpProps::const_iterator j = EPS(others[i]->head)->posPresBegin();
				j != EPS(others[i]->head)->posPresEnd();++j)
		{
			for(int k = startFor(j->op);k != endFor(j->op);++k)
			{
				if(!(--(unsatisfiedPrecs[k])))
				{
					cout << "Enacting " << *instantiatedOp::getInstOp(k) << "\n";
					sasActionTemplates[j->op]->enact(instantiatedOp::getInstOp(k)->getEnv(),reachables,others);
					activated = true;
				};
			};
		};

	};
	for(Reachables::iterator i = reachables.begin();i != reachables.end();++i)
	{
		for(map<const TIMobjectSymbol *,RangeRep *>::iterator j = i->second.begin();
									j != i->second.end();++j)
		{
//			bool preact = activated;
			activated |= j->second->growOneLevel(this);
//			if(!preact && activated) 
//			{
//				cout << "Done it with " << j->first->getName() << "\n";
//			};
		};
	};
//	cout << "Done: " << activated << "\n";
	return activated;

};

ValueElement::ValueElement(ValueElement * vel,FastEnvironment * fe) : pst(vel->pst)
{
	for(vector<VElement*>::iterator i = vel->value.begin();i != vel->value.end();++i)
	{
		value.push_back((*i)->build(fe));
	};
};

VElement * VarElement::build(FastEnvironment * fe)
{
	return new ObElement(TOB((*fe)[var]));

};

bool FunctionStructure::tryMatchedPre(int k,instantiatedOp * iop,const var_symbol * var,
											SASActionTemplate * sasact,ValueRep * vrep)
{
//	cout << *iop << " has " << unsatisfiedPrecs[k] << " precs to satisfy\n";
	if(sasact->checkPre(this,iop->getEnv(),var,vrep))
	{
//	cout << "One down\n";
		if(--(unsatisfiedPrecs[k])) return false;
		SASOUTPUT {cout << "Enacting " << *iop << "\n";};
		sasact->enact(iop->getEnv(),reachables,others);
		return true;
	}
	return false;
};

};

