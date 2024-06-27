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

#include "typecheck.h"
#include "ptree.h"
#include <functional>
#include "main.h"
#include "Exceptions.h"
#include <algorithm>

namespace VAL {

void TypeHierarchy::addDown(const PTypeRef & t1,const PTypeRef & t2)
{
	GIC i = downGraph.find(&t1);
	GIC j = downGraph.find(&t2);
	if(i == downGraph.end())
	{
		i = graph.find(&t1);
		downGraph[i->first] = Nodes();
	};
	if(j == downGraph.end())
	{
		j = graph.find(&t2);
		downGraph[j->first] = Nodes();
		j = downGraph.find(&t2);
	};
	downGraph[&t1].insert(j->first);
};

TypeHierarchy::TypeHierarchy(const analysis * a) : leafNodes()
{
	if(!a || !a->the_domain) 
	{
		ParseFailure pf;
		throw(pf);
	};
	
	if(!a->the_domain->types) return;
	for(pddl_type_list::const_iterator i = a->the_domain->types->begin();
			i != a->the_domain->types->end();++i)
	{
		if((*i)->type)
		{
			add(PTypeRef(*i),PTypeRef((*i)->type));
			addDown(PTypeRef((*i)->type),PTypeRef(*i));
		}
		else
		{	
			if((*i)->either_types)
			{
				for(pddl_type_list::const_iterator j = (*i)->either_types->begin();
					j != (*i)->either_types->end();++j)
				{
					add(PTypeRef(*i),PTypeRef(*j));
					addDown(PTypeRef(*j),PTypeRef(*i));
				};
			}
			else
			{
				PTypeRef pt(*i);
				Graph::iterator j = graph.find(&pt);
				if(j == graph.end())
				{
					TypeRef * c = pt.clone();
					graph[c] = set<const TypeRef*>();
					downGraph[c] = set<const TypeRef*>();
				};
			};
		};
	};
};

TypeHierarchy::~TypeHierarchy()
{
	for(Graph::const_iterator i = graph.begin();i != graph.end();++i)
	{
		delete (i->first);
	};
};




/* gi is the Graph::iterator starting point aiming for t.
 * vs are the places visited so far.
 * gs is the original starting point.
 */

bool TypeHierarchy::closure(Graph & gr,GI & gi,Nodes & vs,GI & gs,const TypeRef * t)
{	 
	if(gi == gr.end()) 
	{
		return false;
	};
	if(*(gi->first) == *t) 
	{
		for(Nodes::const_iterator i = vs.begin();i != vs.end();++i)
		{
			gs->second.insert(*i);
			//insert(vs.begin(),vs.end());
		};
		//gs->second.insert(vs.begin(),vs.end());
		return true;
	};

	for(Nodes::iterator n = gi->second.begin();n != gi->second.end();++n)
	{
		if(vs.find(*n) == vs.end())
		{
			vs.insert(*n);
			GI tmp = gr.find(*n);
			if(closure(gr,tmp,vs,gs,t)) return true;
		};
	};
	return false;
};

vector<const pddl_type *> TypeChecker::leaves(const pddl_type * t)
{
	typedef set<const TypeRef *> Nodes;
	PTypeRef pt(t);
	const Nodes & ns = th.leaves(pt);
	vector<const pddl_type *> vs(ns.size());

	int j = 0;
	for(Nodes::const_iterator i = ns.begin();i != ns.end();++i,++j)
	{
		vs[j] = ***i;
	};

	return vs;
};

bool TypeChecker::isLeafType(const pddl_type * t)
{
	//typedef set<const TypeRef *> Nodes;
	PTypeRef pt(t);
	return th.leaves(pt).empty();
};

vector<const pddl_type *> TypeChecker::accumulateAll(const pddl_type * t)
{
	return th.accumulateAll(t);
};

vector<const pddl_type *> TypeHierarchy::accumulateAll(const pddl_type * t)
{
	vector<const pddl_type *> nds(1,t);
	PTypeRef tt(t);
	GI gi = downGraph.find(&tt);
	if(gi == downGraph.end()) return nds;
	Nodes ns;
	PTypeRef pt(0);
	closure(downGraph,gi,ns,gi,&pt);
	for(Nodes::const_iterator i = ns.begin();i != ns.end();++i)
	{
		nds.push_back(***i);
	};
	return nds;
};

const set<const TypeRef *> & TypeHierarchy::leaves(PTypeRef & t)
{
	static Nodes empty;

	GI gi = leafNodes.find(&t);
	if(gi != leafNodes.end()) 
	{
		return (gi->second);
	};
	
	gi = downGraph.find(&t);
	if(gi == downGraph.end()) return empty;
	Nodes ns;
	PTypeRef pt(0);
	closure(downGraph,gi,ns,gi,&pt);
	Nodes ms;
	for(Nodes::const_iterator i = ns.begin();i != ns.end();++i)
	{
		Nodes xs;
		gi = downGraph.find(*i);
		closure(downGraph,gi,xs,gi,&pt);
		if(xs.empty()) ms.insert(*i);
	};
	leafNodes[downGraph.find(&t)->first] = ms;
	return leafNodes[downGraph.find(&t)->first];
};

void UTypeRef::addContents(TypeHierarchy * th) const
{
	for(std::set<const pddl_type*>::const_iterator i = pts.begin();
				i != pts.end();++i)
	{
		th->add(PTypeRef(*i),*this);
	};
};

bool TypeHierarchy::reachable(const TypeRef & t1,const TypeRef & t2)
{
	if(t1 == t2) return true;

	Graph::iterator i = graph.find(&t1);
	if(i == graph.end()) 
	{
		return false;
	};
	
	Graph::const_iterator j = graph.find(&t2);
	if(j == graph.end() && t2.expected()) 
	{
		return false;
	};
	t2.addContents(this);
	j = graph.find(&t2);
	
	if(i->second.find(j->first) != i->second.end()) return true;

	Nodes ns;
	return closure(graph,i,ns,i,j->first);
};

void TypeHierarchy::add(const PTypeRef & t1,const TypeRef & t2)
{
	Graph::const_iterator i = graph.find(&t1);
	Graph::const_iterator j = graph.find(&t2);
	if(j == graph.end())
	{
		TypeRef * c = t2.clone();
		graph[c] = set<const TypeRef*>();	
		j = graph.find(&t2);
	};
	if(i == graph.end())
	{
		TypeRef * c = t1.clone();
		graph[c] = set<const TypeRef*>(); 
	};
	graph[&t1].insert(j->first);
};

struct badchecker {
	TypeChecker * thea;
	
	badchecker(TypeChecker * a) : thea(a) {};

	bool operator() (const operator_ * a) const
	{
		return !thea->typecheckAction(a);
	};
	bool operator() (const goal * g) const
	{
		return !thea->typecheckGoal(g);
	};
	bool operator() (const plan_step * p)
	{
		return !thea->typecheckActionInstance(p);
	};
	bool operator() (const effect * e)
	{
		return !thea->typecheckEffect(e);
	};
	bool operator() (const derivation_rule * d)
	{
		return !thea->typecheckDerivationRule(d);
	};
};


struct matchFunc {
	const func_symbol * f;

	matchFunc(const func_symbol * fs) : f(fs) {};

	bool operator() (const func_decl * fd) const
	{
		if(!fd)
		{
			if(Verbose) *report << "Problematic function declaration!\n";
			ParseFailure pe;
			throw(pe);
		};
		const func_symbol * fdf = fd->getFunction();
		if(!fdf)
		{
			if(Verbose) *report << *fd << " problematic function declaration!\n";
			ParseFailure pe;
			throw(pe);
		};

		return f == fdf;
	};
};

struct matchPred {
	const pred_symbol * p;

	matchPred(const pred_symbol * ps) : p(ps) {};

	bool operator() (const pred_decl * pd) const
	{
		return p == pd->getPred();
	};
};

struct matchOp {
	const operator_symbol * f;

	matchOp(const operator_symbol * fs) : f(fs) {};

	bool operator() (const operator_ * op) const
	{
		return f == op->name;
	};
};

/* What does it all mean?
 * 
 * Let c be a constant symbol, v a variable symbol and t be a type symbol.
 * 
 * t1 - t2 means t1 is a subset of t2.
 * We allow multiple subset relations (multiple inheritance).
 * c - t means c is a member of t.
 * v - t means instantiations of v must be of type t.
 * 
 * Therefore, subType(c - t1,v - t2) iff t1 - t2. Note that this relationship might 
 * be via a path of transitive subset relations.
 * 
 * v - (either t1 t2) is equivalent to v - (t1 U t2).
 * c - (either t1 t2) is not given a meaning, but we use the either field to record
 *                    multiple types for constants. Thus, c - t1 and c - t2 is legal
 *                    and is recorded internally using c - (either t1 t2). The meaning
 *                    of the multiple declarations is that c is of both types t1 and t2,
 *                    so can be substituted for any variable of either type. 
 * t - (either t1 t2) is also not given a meaning (although the current parser will
 * 		actually interpret it as equivalent to "t - t1 t - t2").
 */


bool TypeChecker::subType(const pddl_typed_symbol * tp1,const pddl_typed_symbol * tp2)
{
	if(!isTyped) return true;
	if(tp1->type)
	{
		if(tp2->type)
		{
			return th.reachable(PTypeRef(tp1->type),PTypeRef(tp2->type));
		}
		else
		{
			if(tp2->either_types)
			{
				return th.reachable(PTypeRef(tp1->type),UTypeRef(tp2->either_types));
			};
			if(Verbose) *report << tp2->getName() << " has bad type definition\n";
			TypeException te;
			throw(te);
		};		
	}
	else
	{
		if(!tp1->either_types) 
		{
			if(Verbose) *report << "Object with unknown type: " << tp1->getName() << "\n";
			TypeException te;
			throw(te);
		};

		// The situation is now complicated by the fact that variables and constants
		// must be treated differently. Constants have either types representing 
		// conjunctions of types, while variables use them for disjunctions. 

		if(!tp1->either_types)
		{
			if(Verbose) *report << tp1->getName() << " badly typed!\n";
			TypeException te;
			throw(te);
		};
		
		if(dynamic_cast<const const_symbol*>(tp1))
		{
			// The following confirms that a constant is a subtype of a given type by
			// checking that one of its types is appropriate. 

			
			for(pddl_type_list::const_iterator i = tp1->either_types->begin();
						i != tp1->either_types->end();++i)
			{
				if(subType(*i,tp2)) return true;
			};
			return false;
		}
		else
		{
		
			for(pddl_type_list::const_iterator i = tp1->either_types->begin();
						i != tp1->either_types->end();++i)
			{
				if(!subType(*i,tp2)) return false;
			};
			return true;
		};
	};
};

bool TypeChecker::subType(const pddl_type * t,const pddl_typed_symbol * s)
{
	if(!isTyped) return true;
	if(s->type)
	{
		return th.reachable(PTypeRef(t),PTypeRef(s->type));
	};

	if(!s->either_types)
	{
		if(Verbose) *report << s->getName() << " badly typed!\n";
		TypeException te;
		throw(te);
	};

	return th.reachable(PTypeRef(t),UTypeRef(s->either_types));
};

bool TypeChecker::subType(const pddl_type * t1,const pddl_type * t2)
{
	if(!isTyped) return true;

	if(!t1 || !t2)
	{
		TypeException te;
		throw(te);
	};
	
	return th.reachable(PTypeRef(t1),PTypeRef(t2));
};
	
bool TypeChecker::typecheckProposition(const proposition * p)
{
	if(!isTyped || !thea->the_domain->predicates) return true;
	pred_decl_list::const_iterator prd = 
		std::find_if(thea->the_domain->predicates->begin(),thea->the_domain->predicates->end(),
				matchPred(p->head));
	if(prd==thea->the_domain->predicates->end())
	{
		if(p->head->getName()=="=") return true;
		if(Verbose) *report << "Predicate " << p->head->getName() << " not found\n";
		return false;
	};
	int idx = 1;

	var_symbol_list::const_iterator arg = (*prd)->getArgs()->begin();
	const var_symbol_list::const_iterator argEnd = (*prd)->getArgs()->end();
	
	parameter_symbol_list::const_iterator i = p->args->begin();
	const parameter_symbol_list::const_iterator e = p->args->end();

	for(;i != e && arg != argEnd;++i,++arg,++idx)
	{
		if(!subType(*i,*arg)) 
		{
			if(Verbose) {
				*report << "Type problem with proposition (";
				*report << p->head->getName();
				parameter_symbol_list::const_iterator it = p->args->begin();
				parameter_symbol_list::const_iterator et = p->args->end();
				for(;it != et;++it) {
					*report << " " << (*it)->getName();
				}
				*report << ") - parameter " << idx << " is incorrectly typed\n";			
			}
			return false;
		};
	};
	if (i != e) {
		if(Verbose) {
			*report << "Problem with proposition (";
			*report << p->head->getName();
			parameter_symbol_list::const_iterator it = p->args->begin();
			parameter_symbol_list::const_iterator et = p->args->end();
			for(;it != et;++it) {
				*report << " " << (*it)->getName();
			}
			*report << ") - too many parameters\n";
		}
		return false;
	}

	if (arg != argEnd) {
		if(Verbose) {
			*report << "Problem with proposition (";
			*report << p->head->getName();
			parameter_symbol_list::const_iterator it = p->args->begin();
			parameter_symbol_list::const_iterator et = p->args->end();
			for(;it != et;++it) {
				*report << " " << (*it)->getName();
			}
			*report << ") - too few parameters\n";
		}
		return false;
	}

	return true;
};

bool TypeChecker::typecheckFuncTerm(const func_term * ft)
{
	if(!isTyped) return true;
	func_decl_list::const_iterator fd = 
		std::find_if(thea->the_domain->functions->begin(),thea->the_domain->functions->end(),
				matchFunc(ft->getFunction()));
	if(fd==thea->the_domain->functions->end())
		return false;
	var_symbol_list::const_iterator arg = (*fd)->getArgs()->begin();
	const var_symbol_list::const_iterator argEnd = (*fd)->getArgs()->end();
	parameter_symbol_list::const_iterator i = ft->getArgs()->begin();
	const parameter_symbol_list::const_iterator e = ft->getArgs()->end();
	int idx = 1;
	for(; i != e && arg != argEnd;++i,++arg,++idx)
	{
		if(!subType(*i,*arg)) 
		{
			if(Verbose) {
				*report << "Type problem with function term (";
				*report << ft->getFunction()->getName();
				parameter_symbol_list::const_iterator it = ft->getArgs()->begin();
				parameter_symbol_list::const_iterator et = ft->getArgs()->end();
				for(;it != et;++it) {
					*report << " " << (*it)->getName();
				}
				*report << ") - parameter " << idx << " is incorrectly typed\n";
			}
			return false;
		};
	};
	if (arg != argEnd) {
		if(Verbose) {
			*report << "Problem with function term (";
			*report << ft->getFunction()->getName();
			parameter_symbol_list::const_iterator it = ft->getArgs()->begin();
			parameter_symbol_list::const_iterator et = ft->getArgs()->end();
			for(;it != et;++it) {
				*report << " " << (*it)->getName();
			}
			*report << ") - too few parameters\n";
		}
	}
	if (i != e) {
		if(Verbose) {
			*report << "Problem with function term (";
			*report << ft->getFunction()->getName();
			parameter_symbol_list::const_iterator it = ft->getArgs()->begin();
			parameter_symbol_list::const_iterator et = ft->getArgs()->end();
			for(;it != et;++it) {
				*report << " " << (*it)->getName();
			}
			*report << ") - too many parameters\n";
		}
	}

	return true;
};

bool TypeChecker::typecheckActionInstance(const plan_step * p)
{
	if(!isTyped) return true;
	operator_list::const_iterator op = 
		std::find_if(thea->the_domain->ops->begin(),thea->the_domain->ops->end(),
				matchOp(p->op_sym));
	if(op==thea->the_domain->ops->end())
		return false;
	var_symbol_list::const_iterator arg = (*op)->parameters->begin();
	const_symbol_list::const_iterator e = p->params->end();
	for(const_symbol_list::const_iterator i = p->params->begin();
			i != e;++i,++arg)
	{
		if(!subType(*i,*arg))
		{
			if(Verbose) *report << "Type problem in action " << *p << "\n";
			return false;
		};
	};
	return true;
};

bool TypeChecker::typecheckExpression(const expression * exp)
{
	if(!isTyped) return true;
	if(const binary_expression * be = dynamic_cast<const binary_expression *>(exp))
	{
		return typecheckExpression(be->getLHS()) && typecheckExpression(be->getRHS());
	};
	if(const uminus_expression * ue = dynamic_cast<const uminus_expression *>(exp))
	{
		return typecheckExpression(ue->getExpr());
	};
	if(const func_term * ft = dynamic_cast<const func_term *>(exp))
	{
		return typecheckFuncTerm(ft);
	};
	return true;
};
		

bool TypeChecker::typecheckGoal(const goal * g)
{
	if(!isTyped) return true;
	if(const preference * p = dynamic_cast<const preference *>(g))
	{
		return typecheckGoal(p->getGoal());
	};
	if(const constraint_goal * cg = dynamic_cast<const constraint_goal *>(g))
	{
		bool b1 = true;
		if(cg->getTrigger()) 
		{
			b1 = typecheckGoal(cg->getTrigger());
		};
		return b1 && typecheckGoal(cg->getRequirement());
	};
	if(const simple_goal * sg = dynamic_cast<const simple_goal *>(g))
	{
		return typecheckProposition(sg->getProp());
	};
	if(const conj_goal * cg = dynamic_cast<const conj_goal *>(g))
	{
		if(cg->getGoals()->end() == std::find_if(cg->getGoals()->begin(),cg->getGoals()->end(),badchecker(this)))
			return true;
		return false;
	};
	if(const disj_goal * dg = dynamic_cast<const disj_goal *>(g))
	{
		if(dg->getGoals()->end() == std::find_if(dg->getGoals()->begin(),dg->getGoals()->end(),badchecker(this)))
			return true;
		return false;
	};
	if(const imply_goal * ig = dynamic_cast<const imply_goal *>(g))
	{
		return typecheckGoal(ig->getAntecedent()) && typecheckGoal(ig->getConsequent());
	};
	if(const neg_goal * ng = dynamic_cast<const neg_goal *>(g))
	{
		return typecheckGoal(ng->getGoal());
	};
	if(const timed_goal * tg = dynamic_cast<const timed_goal *>(g))
	{
		return typecheckGoal(tg->getGoal());
	};
	if(const qfied_goal * qg = dynamic_cast<const qfied_goal *>(g))
	{
		return typecheckGoal(qg->getGoal());
	};
	if(const comparison * c = dynamic_cast<const comparison *>(g))
	{
		return typecheckExpression(c->getLHS()) && typecheckExpression(c->getRHS());
	};
	if(const constraint_goal * cg = dynamic_cast<const constraint_goal *>(g))
	{
		return (!cg->getRequirement() || typecheckGoal(cg->getRequirement())) &&
				(!cg->getTrigger() || typecheckGoal(cg->getTrigger()));
	};
	return false;
};

bool TypeChecker::typecheckEffect(const effect * e)
{
	if(!isTyped) return true;
	if(const simple_effect * se = dynamic_cast<const simple_effect *>(e))
	{
		return typecheckProposition(se->prop);
	};
	if(const cond_effect * ce = dynamic_cast<const cond_effect *>(e))
	{
		return typecheckGoal(ce->getCondition()) &&
				typecheckEffects(ce->getEffects());
	};
	if(const forall_effect * fe = dynamic_cast<const forall_effect *>(e))
	{
		return typecheckEffects(fe->getEffects());
	};
	if(const timed_effect * te = dynamic_cast<const timed_effect *>(e))
	{
		return typecheckEffects(te->effs);
	};
	if(const assignment * ass = dynamic_cast<const assignment *>(e))
	{
		return typecheckFuncTerm(ass->getFTerm()) && typecheckExpression(ass->getExpr());
	};
	return false;
};

bool TypeChecker::typecheckEffects(const effect_lists * es)
{
	if(!isTyped) return true;
	return 
		(es->add_effects.end() == std::find_if(es->add_effects.begin(),es->add_effects.end(),badchecker(this)))
			&&
			(es->del_effects.end() == std::find_if(es->del_effects.begin(),es->del_effects.end(),badchecker(this)))
			&&
			(es->forall_effects.end() == std::find_if(es->forall_effects.begin(),es->forall_effects.end(),badchecker(this)))
			&&
			(es->cond_effects.end() == std::find_if(es->cond_effects.begin(),es->cond_effects.end(),badchecker(this)))
			&&
			(es->assign_effects.end() == std::find_if(es->assign_effects.begin(),es->assign_effects.end(),badchecker(this)))
			&&
			(es->timed_effects.end() == std::find_if(es->timed_effects.begin(),es->timed_effects.end(),badchecker(this)));
};

bool TypeChecker::typecheckAction(const operator_ * act)
{
	if(!isTyped) return true;
	if(Verbose) *report << "Type-checking " << act->name->getName() << "\n";
	if(!typecheckGoal(act->precondition)) {
		if(Verbose) *report << "Conditions fail type-checking.\n";
		return false;
	}
	if(!typecheckEffects(act->effects)) {
		if(Verbose) *report << "Effects fail type-checking.\n";
		return false;
	}
	if(const durative_action * da = dynamic_cast<const durative_action *>(act))
	{
		if (!typecheckGoal(da->dur_constraint)) {
			if(Verbose) *report << "Duration constraint fails type-checking.\n";
			return false;
		};
	};
	if(Verbose) *report << "...action passes type checking.\n";
	return true;
};

bool TypeChecker::typecheckDerivationRule(const derivation_rule * d)
{
	if(!isTyped) return true;
	if(Verbose) *report << "Type-checking derivation rule for " << (d->get_head()->head->getName()) << "\n";
	pred_decl_list::iterator i = thea->the_domain->predicates->begin();
	for(;i != thea->the_domain->predicates->end();++i)
	{
		if((*i)->getPred()==d->get_head()->head)
		{
			(*i)->setTypes(d->get_head());
			break;
		};
	};
	return i != thea->the_domain->predicates->end() && typecheckGoal(d->get_body());
};

bool TypeChecker::typecheckDomain()
{
	if(!isTyped) return true;
	return thea->the_domain->ops->end() == 
		std::find_if(thea->the_domain->ops->begin(),
						thea->the_domain->ops->end(),badchecker(this))
						&&
			thea->the_domain->drvs->end() ==
			std::find_if(thea->the_domain->drvs->begin(),
						thea->the_domain->drvs->end(),badchecker(this))
						&&
			  (!thea->the_domain->constraints || 
			  		typecheckGoal(thea->the_domain->constraints));
};

bool TypeChecker::typecheckProblem()
{
	if(!isTyped) return true;
	if(!thea || !thea->the_problem) 
	{
		ParseFailure pf;
		throw(pf);
	};
	if (thea->the_problem->the_goal && !typecheckGoal(thea->the_problem->the_goal)) {
		if (Verbose) *report << "Type-checking goal failed\n";
		return false;
	}
	if (!typecheckEffects(thea->the_problem->initial_state)) {
		if (Verbose) *report << "Type-checking initial state failed\n";
		return false;
	}
	if (thea->the_problem->constraints && !typecheckGoal(thea->the_problem->constraints)) {
		if (Verbose) *report << "Type-checking constraints failed\n";
		return false;
	}
	return true;
};

bool TypeChecker::typecheckPlan(const plan * p)
{
	if(!isTyped) return true;
	return p->end() == std::find_if(p->begin(),p->end(),badchecker(this));
};

vector<const_symbol *> TypeChecker::range(const var_symbol * v) 
{
	vector<const_symbol *> l;
	for(const_symbol_table::const_iterator i = thea->const_tab.begin();
			i != thea->const_tab.end();++i)
	{
		if(subType(i->second,v)) l.push_back(i->second);
	};
	
	return l;
};

vector<const_symbol *> TypeChecker::range(const parameter_symbol * v) 
{
	vector<const_symbol *> l;
	for(const_symbol_table::const_iterator i = thea->const_tab.begin();
			i != thea->const_tab.end();++i)
	{
		if(subType(i->second,v)) l.push_back(i->second);
	};
	
	return l;
};


vector<const_symbol *> TypeChecker::range(const pddl_type * t) 
{
	var_symbol v("");
	v.type = const_cast<pddl_type*>(t); // OK - we will keep v const.
	v.either_types = 0;
	return range(&v);
};


};
