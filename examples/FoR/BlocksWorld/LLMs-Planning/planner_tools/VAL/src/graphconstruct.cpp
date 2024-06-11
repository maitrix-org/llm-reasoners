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

#include "graphconstruct.h"
#include "ptree.h"
#include "FuncAnalysis.h"
#include <fstream>
#include "State.h"
#include "InstPropLinker.h"
#include "Evaluator.h"
#include "Validator.h"

using namespace VAL;

namespace Inst {

class PlanGraph::BVEvaluator : public VAL::VisitController {
private:
// The BVEvaluator is going to own this bval.
	BoundedValue * bval;
	PlanGraph & pg;
	VAL::FastEnvironment * fenv;

	bool continuous;
	
public:
	BVEvaluator(PlanGraph & p,VAL::FastEnvironment * fe) : 
		bval(0), pg(p), fenv(fe), continuous(false) 
	{};
	~BVEvaluator() 
	{
		delete bval;
	};
	bool isContinuous() const {return continuous;};
	BoundedValue * getBV() 
	{
		BoundedValue * bv = bval;
		bval = 0;
		return bv;
	};
    virtual void visit_plus_expression(plus_expression * pe) 
    {
    	pe->getRHS()->visit(this);
		BoundedValue * br = bval;
		bval = 0;
		pe->getLHS()->visit(this);
		br = (*bval += br);
		if(br != bval)
		{
			delete bval;
			bval = br;
		};
    };
    virtual void visit_minus_expression(minus_expression * me) 
    {
		me->getRHS()->visit(this);
		BoundedValue * br = bval;
		bval = 0;
		me->getLHS()->visit(this);
		br = (*bval -= br);
		if(br != bval)
		{
			delete bval;
			bval = br;
		};
    };
    virtual void visit_mul_expression(mul_expression * pe) 
    {
    	pe->getRHS()->visit(this);
		BoundedValue * br = bval;
		bval = 0;
		pe->getLHS()->visit(this);
		if(continuous)
		{
			bval = br;
		}
		else
		{
			br = (*bval *= br);
			if(br != bval)
			{
				delete bval;
				bval = br;
			};
		};
    };
    virtual void visit_div_expression(div_expression * pe) 
    {
    	pe->getRHS()->visit(this);
		BoundedValue * br = bval;
		bval = 0;
		pe->getLHS()->visit(this);
		br = (*bval /= br);
		if(br != bval)
		{
			delete bval;
			bval = br;
		};
    };
    virtual void visit_uminus_expression(uminus_expression * um) 
    {
		um->getExpr()->visit(this);
		bval->negate();
    };
    virtual void visit_int_expression(int_expression * ie)
    {
		bval = new PointValue(ie->double_value());
    };
    virtual void visit_float_expression(float_expression * fe) 
    {
		bval = new PointValue(fe->double_value());
    };
    virtual void visit_special_val_expr(special_val_expr *) 
    {continuous = true;};
    virtual void visit_func_term(func_term * ft) 
    {
    	PNE pne(ft,fenv);
    	FluentEntry * fe = pg.fluents.find(instantiatedOp::getPNE(&pne));
		bval = fe?fe->getBV()->copy():new Undefined();
    };
};

class SpikeEvaluator : public VisitController {
private: 
	Spike<PropEntry> & spes;
	Spike<FluentEntry> & sfes;
	FastEnvironment* f;
	bool evaluation;
	PlanGraph & pg;
	
	pred_symbol * equality;
public:
	SpikeEvaluator(PlanGraph & p,Spike<PropEntry> & s1,Spike<FluentEntry> & s2, FastEnvironment * fe): 
		spes(s1), sfes(s2), f(fe), evaluation(true), pg(p),
		equality(current_analysis->pred_tab.symbol_probe("=")) {
	}
	
	virtual void visit_simple_goal(simple_goal * s){		
		if(EPS(s->getProp()->head)->getParent() == this->equality){
			evaluation = ((*f)[s->getProp()->args->front()] == 
						(*f)[s->getProp()->args->back()]);
		
		
			if(s->getPolarity() == E_NEG)
			{
				evaluation = !evaluation;
			};
			return;
		}
		else {
			Literal e(s->getProp(),f);
			Literal* lptr = instantiatedOp::getLiteral(& e);
			PropEntry* eid = spes.find(lptr);
			if(eid){
				if(s->getPolarity() == E_NEG){
					evaluation = eid->gotDeleters();
				}
				else {
					evaluation = eid->gotAchievers();
				}
			}
			else {
				if(s->getPolarity() == E_NEG){
					evaluation = true;
					
				}
				else {
					evaluation = false;
				}
			};
		};
			
	};

	bool getEvaluation() const {return evaluation;};	
		
	virtual void visit_qfied_goal(qfied_goal * qg){
		cout << "Not currently handling quantified goals\n";
	}
	virtual void visit_conj_goal(conj_goal * c){
		for(goal_list::const_iterator i = c->getGoals()->begin();
			i != c->getGoals()->end();++i)
		{
			(*i)->visit(this);
			if(!evaluation){
				return;
			}
		}
	};
	virtual void visit_disj_goal(disj_goal * c){	
		cout << "Not dealing with disjunctive goals\n";
	};
	virtual void visit_timed_goal(timed_goal * t){
		cout << "Not currently handling timed goals\n";
	}
	virtual void visit_imply_goal(imply_goal * ig){	
		cout << "Not dealing with implications\n";
	};

	virtual void visit_neg_goal(neg_goal * ng){
		ng->getGoal()->visit(this);
		evaluation = !evaluation;
	}
	virtual void visit_comparison(comparison * c){
		// Evaluate the parts and combine according to 
		// rearrangement then do the comparison with a 
		// bounds check.
		PlanGraph::BVEvaluator bve(pg,f);
		c->getLHS()->visit(&bve);
		BoundedValue * bvl = bve.getBV();
		c->getRHS()->visit(&bve);
		BoundedValue * bvr = bve.getBV();
		BoundedValue * bvres = (*bvl -= bvr);
		switch(c->getOp())
		{
			case E_GREATER:
				evaluation = !bvres->gotUB() || bvres->getUB() > 0;
				break;
			case E_GREATEQ:
				evaluation = !bvres->gotUB() || bvres->getUB() >= 0;
				break;
			case E_LESS:
				evaluation = !bvres->gotLB() || bvres->getLB() < 0;
				break;
			case E_LESSEQ:
				evaluation = !bvres->gotLB() || bvres->getLB() <= 0;
				break;
			case E_EQUALS:
				evaluation = (!bvres->gotLB() || bvres->getLB() <= 0) &&
								(!bvres->gotUB() || bvres->getUB() >= 0);
				break;
			default:
				break;
		};
		
	}
	
	virtual void visit_action(action * op){	
		op->precondition->visit(this);
	};
	virtual void visit_event(event * e){	
		e->precondition->visit(this);
	};
	virtual void visit_process(process * p){	
		p->precondition->visit(this);
	};
	virtual void visit_durative_action(durative_action * da) {
		cout << "Not dealing with duratives\n";
	};
};

class SpikeSupporter : public VisitController {
private: 
	Spike<PropEntry> & spes;
	Spike<FluentEntry> & sfes;
	FastEnvironment* f;
	ActEntry * ae;

	GraphFactory * myFac;

	bool context;

	pred_symbol * equality;
public:
	SpikeSupporter(Spike<PropEntry> & s1,Spike<FluentEntry> & s2, FastEnvironment * fe,ActEntry * a,GraphFactory * mf): 
		spes(s1), sfes(s2), f(fe), ae(a), myFac(mf), context(true),
		equality(current_analysis->pred_tab.symbol_probe("=")) {
	}
	
	virtual void visit_simple_goal(simple_goal * s){		
		if(EPS(s->getProp()->head)->getParent() != this->equality){
			Literal e(s->getProp(),f);
			Literal* lptr = instantiatedOp::getLiteral(& e);
			PropEntry* eid = spes.findInAll(lptr);
			if(eid){
			  if((context && s->getPolarity() == E_NEG) ||
			     (!context && s->getPolarity()==E_POS)){
					ae->addSupportedByNeg(eid);
					cout << "Support by neg: " << *ae << " with " << *eid << "\n";
				}
				else {
					ae->addSupportedBy(eid);
				}
			}
			else {
				eid=myFac->makePropEntry(lptr);
			// make the entry for eid
				
				ae->addSupportedByNeg(eid);
				spes.insertAbsentee(eid);	
				
				
			};
		};
			
	};

	virtual void visit_conj_goal(conj_goal * c){
		for(goal_list::const_iterator i = c->getGoals()->begin();
			i != c->getGoals()->end();++i)
		{
			(*i)->visit(this);
		}
	};

	virtual void visit_comparison(comparison * c){
		//cout << "Er....what?\n";
	}

	virtual void visit_neg_goal(neg_goal * ng)
	{
		context = !context;
		ng->getGoal()->visit(this);
	};
	
	virtual void visit_action(action * op){	
		op->precondition->visit(this);
	};
	virtual void visit_event(event * e){	
		e->precondition->visit(this);
	};
	virtual void visit_process(process * p){	
		p->precondition->visit(this);
	};
	virtual void visit_durative_action(durative_action * da) {
		cout << "Not dealing with duratives\n";
	};
};


void FluentEntry::write(ostream & o) const
{
	thefluent->write(o);
	o << "[";
	for(vector<Constraint *>::const_iterator i = constrs.begin();i != constrs.end();++i)
	{
		(*i)->write(o);
		o << " ";
	};
	o << "]\nBounded Range: " << *bval << "\n";
};



BoundedValue * BoundedInterval::operator+=(const BoundedValue * bv)
{
	if(!finitelbnd || !bv->gotLB())
	{
		finitelbnd = false;
	}
	else
	{
		lbnd += bv->getLB();
	};
	if(!finiteubnd || !bv->gotUB())
	{
		finiteubnd = false;
	}
	else
	{
		ubnd += bv->getUB();
	};
	return this;
};

BoundedValue * BoundedInterval::operator-=(const BoundedValue * bv)
{
	if(!finitelbnd || !bv->gotUB())
	{
		finitelbnd = false;
	}
	else
	{
		lbnd -= bv->getUB();
	};
	if(!finiteubnd || !bv->gotLB())
	{
		finiteubnd = false;
	}
	else
	{
		ubnd -= bv->getLB();
	};
	return this;
};

BoundedValue * BoundedInterval::operator*=(const BoundedValue * bv)
{
	if(!finitelbnd || !bv->gotLB())
	{
		finitelbnd = false;
	}
	else
	{
		lbnd *= bv->getLB();
	};
	if(!finiteubnd || !bv->gotUB())
	{
		finiteubnd = false;
	}
	else
	{
		ubnd *= bv->getUB();
	};
	return this;
};

BoundedValue * BoundedInterval::operator/=(const BoundedValue * bv)
{
/*	if(!finitelbnd || !bv->gotLB())
	{
		finitelbnd = false;
	}
	else
	{
		lbnd += bv->getLB();
	};
	if(!finiteubnd || !bv->gotUB())
	{
		finiteubnd = false;
	}
	else
	{
		ubnd += bv->getUB();
	};
	return this;
*/
// This case must be handled properly...
	cout << "WARNING: Division not managed properly, yet!\n";
	finitelbnd = finiteubnd = false;
	return this;
};

BoundedValue * PointValue::operator+=(const BoundedValue * bv)
{
	BoundedInterval * bi = new BoundedInterval(val,val);
	*bi += bv;
	return bi;
};

BoundedValue * PointValue::operator-=(const BoundedValue * bv)
{
	BoundedInterval * bi = new BoundedInterval(val,val);
	*bi -= bv;
	return bi;
};

BoundedValue * PointValue::operator*=(const BoundedValue * bv)
{
	BoundedInterval * bi = new BoundedInterval(val,val);
	*bi *= bv;
	return bi;
};

BoundedValue * PointValue::operator/=(const BoundedValue * bv)
{
	BoundedInterval * bi = new BoundedInterval(val,val);
	*bi /= bv;
	return bi;
};

BoundedValue * PlanGraph::update(BoundedValue * bv,const VAL::expression * exp,const VAL::assign_op op,VAL::FastEnvironment * fe)
{
	BVEvaluator bve(*this,fe);
	exp->visit(&bve);
	BoundedValue * b = bve.getBV();
	cout << "Evaluated to " << *b << "\n";
	switch(op)
	{
		case E_ASSIGN:
			bv = b->copy();
			break;
		case E_INCREASE:
			if(bve.isContinuous())
			{
				if(!b->gotLB() || b->getLB() < 0)
				{
					bv = bv->infLower();
				};
				if(!b->gotUB() || b->getUB() > 0)
				{
					bv = bv->infUpper();
				};
			}
			else
			{
				bv = (*bv += b);
			};
			break;
		case E_DECREASE:
			if(bve.isContinuous())
			{
				if(!b->gotLB() || b->getLB() < 0)
				{
					bv = bv->infUpper();
				};
				if(!b->gotUB() || b->getUB() > 0)
				{
					bv = bv->infLower();
				};
			}
			else
			{
				bv = (*bv -= b);
			};
			break;
		case E_SCALE_UP:
			bv = (*bv *= b);
			break;
		case E_SCALE_DOWN:
			bv = (*bv /= b);
			break;
		default:
			break;
	};
	delete b;
	return bv;
};

void Constraint::write(ostream & o) const 
{
	o << *bval;
};

void InitialValue::write(ostream & o) const
{
	o << "Initially " << *bval;
};

void UpdateValue::write(ostream & o) const
{
	o << "Updated by " << *(updater->getIO()) << " at " << (updater->getWhen()) 
	   //<< " with effect: " <<  *exp
	    << " to " << *bval;
};

void FluentEntry::addUpdatedBy(ActEntry * ae,const VAL::expression * expr,const VAL::assign_op op,PlanGraph * pg)
{
	cout << "Performing BV calc on " << *bval << "\n";
	BoundedValue * vv = bval->copy();
	BoundedValue * v = pg->update(vv,expr,op,ae->getIO()->getEnv());
	cout << "Got " << *v << "\n";
	if(vv != v)
	{
		delete vv;
	};
	Constraint * c = new UpdateValue(ae,expr,op,v);
	constrs.push_back(c);
	if(!tmpaccum) 
	{
		tmpaccum = bval->copy();
	};
	cout << "tmpaccum is " << *tmpaccum << "\n";
	BoundedValue * nv = tmpaccum->accum(v);
	if(nv != tmpaccum)
	{
		delete tmpaccum;
	};
	tmpaccum = nv;
};

BoundedValue * PointValue::accum(const BoundedValue * bv)
{
	if(bv->contains(val))
	{
		return bv->copy();
	}
	else
	{
		BoundedValue * b = new BoundedInterval(val,val);
		b->accum(bv);
		return b;
	};
};
	
PlanGraph::PlanGraph(GraphFactory * f) : myFac(f),
	inactive(instantiatedOp::opsBegin(),instantiatedOp::opsEnd())
{
// Set up the initial state in the proposition spike...
	for(pc_list<simple_effect*>::const_iterator i = 
				current_analysis->the_problem->initial_state->add_effects.begin();
				i != current_analysis->the_problem->initial_state->add_effects.end();++i)
	{
		Literal lit((*i)->prop,0);
		Literal * lit1 = instantiatedOp::getLiteral(&lit);
		PropEntry * p = myFac->makePropEntry(lit1);
		props.addEntry(p);
	};
	props.finishedLevel();

	for(pc_list<assignment*>::const_iterator i = 
				current_analysis->the_problem->initial_state->assign_effects.begin();
				i != current_analysis->the_problem->initial_state->assign_effects.end();++i)
	{
		PNE pne((*i)->getFTerm(),0);
		PNE * pne1 = instantiatedOp::getPNE(&pne);
		FluentEntry * fl = myFac->makeFluentEntry(pne1);
		fluents.addEntry(fl);
		fl->addInitial((EFT(pne1->getHead())->getInitial(pne1->begin(),pne1->end())).second);
	};
	fluents.finishedLevel();
	//copy(instantiatedOp::opsBegin(),instantiatedOp::opsEnd(),front_inserter(inactive));
};

Constraint::~Constraint() 
{
	delete bval;
};

void FluentEntry::transferValue() 
{
	if(!tmpaccum) return;
	delete bval;
	bval = tmpaccum;
	tmpaccum = 0;
};

struct IteratingActionChecker : public VisitController {
	bool iterating;

	IteratingActionChecker() : iterating(false) {};

	virtual void visit_forall_effect(forall_effect * fa) 
	{
		cout << "Not handling for all effects yet (IteratingActionChecker)!\n";
	};
	virtual void visit_cond_effect(cond_effect *) 
	{
		cout << "Not handling conditional effects yet (IteratingActionChecker)!\n";
	};
//	virtual void visit_timed_effect(timed_effect *) {};
	virtual void visit_effect_lists(effect_lists * effs) 
	{
		for(VAL::pc_list<assignment *>::iterator i = effs->assign_effects.begin();i != effs->assign_effects.end();++i)
		{
			(*i)->visit(this);
		};
	};
	virtual void visit_assignment(assignment * a) 
	{
		switch(a->getOp())
		{
			case E_INCREASE:
			case E_DECREASE:
			case E_SCALE_UP:
			case E_SCALE_DOWN:
				iterating = true;
			default:
				break;
		};
	};
};


void DurationHolder::readDurations(const string & nm)
{
	std::ifstream dursFile(nm.c_str());
	string a;
	string ax;
	string s;
	dursFile >> a;
	ax = a;
	vector<int> args;
	while(!dursFile.eof())
	{
		dursFile >> s;
		if(s == "=")
		{
			relevantArgs[a] = args;
			args.clear();
			double d;
			dursFile >> d;
			dursFor[ax] = new DurationConstraint(new PointValue(d));
			dursFile >> a;
			ax = a;
		}
		else
		{
			int arg;
			dursFile >> arg;
			args.push_back(arg);
			ax += " ";
			ax += s;
		};
	};	
};

DurationHolder ActEntry::dursFor;

void DurationConstraint::write(ostream & o) const
{
	o << "Duration for ";
	if(start) o << *(start->getIO()) << " ";
	if(inv) o << *(inv->getIO()) << " ";
	if(end) o << *(end->getIO()) << " ";
	o << "is " << *bval;
};

DurationConstraint * DurationHolder::lookUp(const string & nm,instantiatedOp * io)
{
	vector<int> args = relevantArgs[nm];
	string s = nm;
	for(vector<int>::iterator i = args.begin();i != args.end();++i)
	{
		s += " ";
		s += io->getArg(*i)->getName();
	};
	return dursFor[s];
};

ActEntry::ActEntry(instantiatedOp * io) : theact(io), iterating(false), atype(ATOMIC),
	dur(0)
{
	IteratingActionChecker iac;
	io->forOp()->effects->visit(&iac);
	iterating = iac.iterating;
	
	string s = io->forOp()->name->getName();
	if(s.length() < 6) return;
	string tl = s.substr(s.length()-4,4);
	if(tl == "-inv")
	{
		cout << "Found an invariant action " << *io << "\n";
		atype = INV;
		tl = s.substr(0,s.length()-4);
		dur = dursFor.lookUp(tl,io);
		dur->setInv(this);
	}
	else if(tl == "-end")
	{
		cout << "Found an end action " << *io << "\n";
		atype = END;
		tl = s.substr(0,s.length()-4);
		dur = dursFor.lookUp(tl,io);
		dur->setEnd(this);
	}
	else if(s.length() > 6 && s.substr(s.length()-6,6) == "-start")
	{
		cout << "Found a start action " << *io << "\n";
		atype = START;
		tl = s.substr(0,s.length()-6);
		dur = dursFor.lookUp(tl,io);
		dur->setStart(this);
	};
};

bool PlanGraph::extendPlanGraph()
{
	for(vector<ActEntry *>::iterator i = iteratingActs.begin();i != iteratingActs.end();++i)
	{
		iterateEntry(*i);
	};

	bool levelOut = true;
	for(InstOps::iterator i = inactive.begin();i!= inactive.end();){
		cout << "Considering: " << **i << "\n";
		if(activated((*i))){
			ActEntry* io = acts.addEntry(myFac->makeActEntry((*i)));
			cout << "Activated: " << (*(*i)) << "\n";
			activateEntry(io);
			InstOps::iterator j = i;
			
			++i;
			inactive.erase(j);
			
			levelOut = false;
		}
		else ++i;
	}

	
	
// Determine which actions are now activated and add them to spike.
// 
// Then add their postconditions to the proposition spike, ensuring we only add new ones.

	acts.finishedLevel();
	props.finishedLevel();
	fluents.finishedLevel();
	for(Spike<FluentEntry>::SpikeIterator i = fluents.begin();i != fluents.end();++i)
	{
		(*i)->transferValue();
	};
	return levelOut;
};

void PlanGraph::extendToGoals()
{
	VAL::FastEnvironment bs(0);
	while(true)
	{
		extendPlanGraph();
		SpikeEvaluator spiv(*this,props,fluents,&bs);
		current_analysis->the_problem->the_goal->visit(&spiv);
		if(spiv.getEvaluation()) break;
	};
};

void PlanGraph::iterateEntry(ActEntry * io)
{
	for(instantiatedOp::PNEEffectsIterator e = io->getIO()->PNEEffectsBegin();e!=io->getIO()->PNEEffectsEnd();++e){
		FluentEntry* eid = fluents.find((*e));
		cout << "Fluent effect updated: " << (*(*e)) << "\n";
		if(!eid){
			eid = fluents.addEntry(myFac->makeFluentEntry((*e)));
		};
		
		eid->addUpdatedBy(io,e.getUpdate(),e.getOp(),this);
		io->addUpdates(eid);
	}
};

void
PlanGraph::activateEntry(ActEntry * io){
	for(instantiatedOp::PropEffectsIterator e = io->getIO()->addEffectsBegin();e!=io->getIO()->addEffectsEnd();++e){
		PropEntry* eid = props.find((*e));
		if(!eid){
			eid=props.addEntry(myFac->makePropEntry((*e)));
			cout << "Prop effect added: " << (*(*e)) << "\n";
		};
		eid->addAchievedBy(io);
		io->addAchieves(eid);
	}
	for(instantiatedOp::PropEffectsIterator e = io->getIO()->delEffectsBegin();e!=io->getIO()->delEffectsEnd();++e){
		PropEntry* eid = props.find((*e));
		if(!eid){
			eid=props.addEntry(myFac->makePropEntry((*e)));
			cout << "Prop effect deleted: " << (*(*e)) << "\n";
		}
		eid->addDeletedBy(io);
		io->addDeletes(eid);
	}
	
	iterateEntry(io);

	SpikeSupporter spipp(props,fluents,io->getIO()->getEnv(),io,myFac);

	io->getIO()->forOp()->visit(&spipp);
	
	if(io->isIterating())
	{
		iteratingActs.push_back(io);
	};
};

// Method to check whether an action is to be activated at a given level.
bool
PlanGraph::activated(instantiatedOp* io){
	SpikeEvaluator spiv(*this,props,fluents,io->getEnv());
	io->forOp()->visit(&spiv);
	return spiv.getEvaluation();
}
	
void ActEntry::write(ostream & o) const
{
	o << *theact;
	if(atype != ATOMIC && dur)
	{
		o << " " << *dur;
	};
};

void PlanGraph::write(ostream & o) const
{
	o << "Propositions:\n";
	props.write(o);
	o << "Actions:\n";
	acts.write(o);
	o << "Fluents:\n";
	fluents.write(o);
};

int PropEntry::counter = 0;

bool ActEntry::isActivated(const vector<bool> & actives) const
{
	for(vector<PropEntry *>::const_iterator i = supports.begin();i != supports.end();++i)
	{
		cout << "Checking +" << **i << " = " << actives[(*i)->getID()] << "\n";
		if(!actives[(*i)->getID()]) return false;
	};
	for(vector<PropEntry *>::const_iterator i = negSupports.begin();i != negSupports.end();++i)
	{
		cout << "Checking -" << **i << " = " << actives[(*i)->getID()] << "\n";
		if(actives[(*i)->getID()]) return false;
	};
	return true;
};

bool ActEntry::isActivated(Validator * v,const State * s) const 
{
	Evaluator ev(v,s,theact);
	theact->forOp()->visit(&ev);
	return ev();
};

bool ActEntry::isRelevant(Validator * v,const State * s) const
{
	Evaluator ev(v,s,theact,true);
	theact->forOp()->visit(&ev);
	return ev();
};

vector<ActEntry *> PlanGraph::applicableActions(Validator * v,const State * s)
{
	int lastActiveLayer = 0;
	for(State::const_iterator i = s->begin();i != s->end();++i)
	{
		Literal * lit = toLiteral(*i);
		PropEntry * pe = props.findInAll(lit);
		lastActiveLayer = std::max(lastActiveLayer,pe->getWhen());
	};
	vector<ActEntry *> actives;
	for(Spike<ActEntry>::SpikeIterator i = acts.begin();i != acts.toLevel(lastActiveLayer);++i)
	{
		cout << "Considering " << **i << "\n";
		if((*i)->isActivated(v,s))
		{
			actives.push_back(*i);
		};
	};

	return actives;



/* This version was used to translate a State into a vector of bools
 * that could be used for reference against preconditions.
 * The problem is that it doesn't handle metric expressions, so we
 * have switched to evaluation in the state.
 *
 	vector<bool> activations(props.size(),false);
	int lastActiveLayer = 0;
	for(State::const_iterator i = s->begin();i != s->end();++i)
	{
		Literal * lit = toLiteral(*i);
		PropEntry * pe = props.findInAll(lit);
		activations[pe->getID()] = true;
		cout << "Set " << *pe << " active\n";
		lastActiveLayer = std::max(lastActiveLayer,pe->getWhen());
	};
	vector<ActEntry *> actives;
	for(Spike<ActEntry>::SpikeIterator i = acts.begin();i != acts.toLevel(lastActiveLayer);++i)
	{
		cout << "Considering " << **i << "\n";
		if((*i)->isActivated(activations))
		{
			actives.push_back(*i);
		};
	};

	return actives;
*/
};

vector<ActEntry *> PlanGraph::relevantActions(Validator * v,const State * s)
{
	int lastActiveLayer = 0;
	for(State::const_iterator i = s->begin();i != s->end();++i)
	{
		Literal * lit = toLiteral(*i);
		PropEntry * pe = props.findInAll(lit);
		lastActiveLayer = std::max(lastActiveLayer,pe->getWhen());
	};
	vector<ActEntry *> actives;
	for(Spike<ActEntry>::SpikeIterator i = acts.begin();i != acts.toLevel(lastActiveLayer);++i)
	{
//		cout << "Considering " << **i << "\n";
		if((*i)->isRelevant(v,s))
		{
			actives.push_back(*i);
		};
	};

	return actives;
};
};
