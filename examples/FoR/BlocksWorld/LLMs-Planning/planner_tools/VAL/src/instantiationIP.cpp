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

#include <cstdio>
#include <iostream>
#include <fstream>
#include "ptree.h"
#include <FlexLexer.h>
#include "instantiationIP.h"
#include "SimpleEval.h"
#include "DebugWriteController.h"
#include "typecheck.h"

using std::ifstream;
using std::cerr;

ostream & operator<<(ostream & o,const instantiatedOp & io)
{
	io.write(o);
	return o;
};

ostream & operator<<(ostream & o,const Literal & io)
{
	io.write(o);
	return o;
};

void instantiatedOp::writeAll(ostream & o) 
{
	for(OpStore::const_iterator i = instOps.begin();
				i != instOps.end();++i)
	{
		o << **i << "\n";
	};
};

bool Verbose = false;

OpStore instantiatedOp::instOps;
map<pddl_type *,vector<const_symbol*> > instantiatedOp::values;



void instantiatedOp::instantiate(const operator_ * op,const problem * prb,TypeChecker & tc)
{
	FastEnvironment e(static_cast<const id_var_symbol_table*>(op->symtab)->numSyms());
	vector<vector<const_symbol*>::const_iterator> vals(op->parameters->size());
	vector<vector<const_symbol*>::const_iterator> starts(op->parameters->size());
	vector<vector<const_symbol*>::const_iterator> ends(op->parameters->size());
	vector<var_symbol *> vars(op->parameters->size());
	int i = 0;
	int c = 1;
	for(var_symbol_list::const_iterator p = op->parameters->begin();
			p != op->parameters->end();++p,++i)
	{
		if(values.find((*p)->type) == values.end()) 
		{
			values[(*p)->type] = tc.range(*p);
		};
		vals[i] = starts[i] = values[(*p)->type].begin();
		ends[i] = values[(*p)->type].end();
		if(ends[i]==starts[i]) return;
		e[(*p)] = *(vals[i]);
		vars[i] = *p;
		c *= values[(*p)->type].size();
	};
	cout << c << " candidates to consider\n";
	if(!i)
	{
		SimpleEvaluator se(&e);
		op->visit(&se);
		if(!se.reallyFalse())
		{
			FastEnvironment * ecpy = e.copy();
			instOps.push_back(new instantiatedOp(op,ecpy));
		};
		return;
	};
	--i;
	while(vals[i] != ends[i])
	{
		SimpleEvaluator se(&e);
		const_cast<operator_*>(op)->visit(&se);
		if(!se.reallyFalse())
		{
			FastEnvironment * ecpy = e.copy();
			instOps.push_back(new instantiatedOp(op,ecpy));
		};
/*
 *		else
		{
			cout << "Killed\n" << op->name->getName() << "(";
			for(var_symbol_list::const_iterator a = op->parameters->begin();
					a != op->parameters->end();++a)
			{
				cout << e[*a]->getName() << " ";
			};
			cout << ")\n";
		};
*/
		int x = 0;
		++vals[0];
		e[vars[0]] = *(vals[0]);
		while(x < i && vals[x] == ends[x])
		{
			vals[x] = starts[x];
			e[vars[x]] = *(vals[x]);
			++x;
			++vals[x];
			e[vars[x]] = *(vals[x]);
		};
	};
};

LiteralStore instantiatedOp::literals;




class Collector : public VisitController {
private:
	bool adding;
	const operator_ * op;
	FastEnvironment * fe;
	LiteralStore & literals;
	LiteralStore & pres;
	LiteralStore & adds;
	LiteralStore & dels;
public:
	Collector(const operator_ * o,FastEnvironment * f,LiteralStore & l,
				LiteralStore & ps,LiteralStore & as,LiteralStore & ds) :
		adding(true), op(o), fe(f), literals(l), pres(ps), adds(as), dels(ds)
	{};
	
	virtual void visit_simple_goal(simple_goal * p) 
	{
		Literal * l = new Literal(p->getProp(),fe);
		literals.insert(l);
		pres.insert(l);
	};
	virtual void visit_qfied_goal(qfied_goal * p) 
	{p->getGoal()->visit(this);};
	virtual void visit_conj_goal(conj_goal * p) 
	{p->getGoals()->visit(this);};
	virtual void visit_disj_goal(disj_goal * p) 
	{p->getGoals()->visit(this);};
	virtual void visit_timed_goal(timed_goal * p) 
	{p->getGoal()->visit(this);};
	virtual void visit_imply_goal(imply_goal * p) 
	{
		p->getAntecedent()->visit(this);
		p->getConsequent()->visit(this);
	};
	virtual void visit_neg_goal(neg_goal * p) 
	{
		p->getGoal()->visit(this);
	};
	virtual void visit_simple_effect(simple_effect * p) 
	{
		Literal * l = new Literal(p->prop,fe);
		literals.insert(l);
		if(adding)
		{
			adds.insert(l);
		}
		else
		{
			dels.insert(l);
		};
	};
	virtual void visit_forall_effect(forall_effect * p) 
	{
		p->getEffects()->visit(this);
	};
	virtual void visit_cond_effect(cond_effect * p) 
	{
		p->getCondition()->visit(this);
		p->getEffects()->visit(this);
	};
	virtual void visit_timed_effect(timed_effect * p) 
	{
		p->effs->visit(this);
	};
	virtual void visit_timed_initial_literal(timed_initial_literal * p)
	{
		p->effs->visit(this);
	};
	virtual void visit_effect_lists(effect_lists * p) 
	{
		p->add_effects.pc_list<simple_effect*>::visit(this);
		p->forall_effects.pc_list<forall_effect*>::visit(this);
		p->cond_effects.pc_list<cond_effect*>::visit(this);
		p->timed_effects.pc_list<timed_effect*>::visit(this);
		bool whatwas = adding;
		adding = !adding;
		p->del_effects.pc_list<simple_effect*>::visit(this);
		adding = whatwas;
	};
	virtual void visit_operator_(operator_ * p) 
	{
		adding = true;
		p->effects->visit(this);
	};
	virtual void visit_action(action * p)
	{
		visit_operator_(static_cast<operator_*>(p));
	};
	virtual void visit_durative_action(durative_action * p) 
	{
		visit_operator_(static_cast<operator_*>(p));
	};
	virtual void visit_problem(problem * p) 
	{
		p->initial_state->visit(this);
		p->the_goal->visit(this);
	};
};

void instantiatedOp::createAllLiterals(problem * p) 
{
	Collector c(0,0,literals);
	p->visit(&c);
	for(OpStore::iterator i = instOps.begin(); i != instOps.end(); ++i)
	{
		(*i)->collectLiterals();
	};
};

void LiteralStore::insert(Literal * lit)
{
	if(!literals[lit->getHead()].contains(lit->begin(),lit->end()))
	{
		literals[lit->getHead()].insert(lit->begin(),lit->end());
		allLits.push_back(lit);
	};
};

void instantiatedOp::collectLiterals()
{
	Collector c(op,env,literals,pres,adds,dels);
	op->visit(&c);
};

void instantiatedOp::writeAllLiterals(ostream & o) 
{
	literals.write(o);
};

void LiteralStore::write(instantiatedOp* op,ostream & o) const
{
	for(deque<Literal*>::const_iterator i = allLits.begin();i != allLits.end();++i)
	{
		o << *op << "  " << **i << "\n";
	};
};

void instantiatedOp::writePres(ostream & o) const
{
	pres.write(this,o);
};

void instantiatedOp::writeAdds(ostream & o) const
{
	adds.write(this,o);
};

void instantiatedOp::writeDels(ostream & o) const
{
	dels.write(this,o);
};

extern int yyparse();
extern int yydebug;

parse_category* top_thing=NULL;


TypeChecker * theTC;

int PropInfo::x = 0;


analysis* current_analysis;
char * current_filename;

yyFlexLexer* yfl;

int main(int argc,char * argv[])
{
	analysis an_analysis;
    current_analysis= &an_analysis;
    IDopTabFactory * fac = new IDopTabFactory;
    an_analysis.setFactory(fac);
    an_analysis.pred_tab.replaceFactory<holding_pred_symbol>();
    an_analysis.func_tab.replaceFactory<extended_func_symbol>();
    yydebug=0; // Set to 1 to output yacc trace 

    yfl= new yyFlexLexer;

	for(int i = 1;i < 3;++i)
	{
		cout << "File: " << argv[i] << '\n';
		current_filename = argv[i];
		ifstream thefile(argv[i]);
		if (!thefile)
		{
		    // Output a message now
		    cout << "Failed to open\n";
		    
		    // Log an error to be reported in summary later
		    line_no= 0;
		    log_error(E_FATAL,"Failed to open file");
		}
		else
		{
		    line_no= 1;

		    // Switch the tokeniser to the current input stream
		    yfl->switch_streams(&thefile,&cout);
		    yyparse();

		    // Output syntax tree
		    //if (top_thing) top_thing->display(0);
		}
    }
    // Output the errors from all input files
    current_analysis->error_list.report();
    delete yfl;

	TypeChecker tc(current_analysis);
	if(!tc.typecheckDomain())
    {
    	cerr << "Type problem in domain description!\n";
    	exit(1);
    };
    if(!tc.typecheckProblem())
	{
		cerr << "Type problem in problem specification!\n";
		exit(1);
	};
	theTC = &tc;
    TypePredSubstituter a;
    current_analysis->the_problem->visit(&a);
   	current_analysis->the_domain->visit(&a);
	Analyser an;
	an_analysis.the_problem->visit(&an);
	an_analysis.the_domain->visit(&an);
	SimpleEvaluator::setInitialState();
    for(operator_list::const_iterator os = an_analysis.the_domain->ops->begin();os != an_analysis.the_domain->ops->end();++os)
    {
//    	cout << (*os)->name->getName() << "\n";
    	instantiatedOp::instantiate(*os,an_analysis.the_problem,tc);
//    	cout << instantiatedOp::howMany() << " so far\n";
    };
//    cout << instantiatedOp::howMany() << "\n";
	cout << "set actions :=\n";
    instantiatedOp::writeAll(cout);
	
	cout << "\n\nset :\n";
    instantiatedOp::createAllLiterals(current_analysis->the_problem);
    instantiatedOp::writeAllLiterals(cout);
}
