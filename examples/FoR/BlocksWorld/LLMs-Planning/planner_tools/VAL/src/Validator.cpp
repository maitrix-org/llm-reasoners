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

/*-----------------------------------------------------------------------------
  VAL - The Automatic Plan Validator for PDDL+

  $Date: 2009-02-05 10:50:24 $
  $Revision: 1.2 $

  Maria Fox, Richard Howey and Derek Long - PDDL+ and VAL
  Stephen Cresswell - PDDL Parser

  maria.fox@cis.strath.ac.uk
  derek.long@cis.strath.ac.uk
  stephen.cresswell@cis.strath.ac.uk
  richard.howey@cis.strath.ac.uk

  By releasing this code we imply no warranty as to its reliability
  and its use is entirely at your own risk.

  Strathclyde Planning Group
  http://planning.cis.strath.ac.uk
 ----------------------------------------------------------------------------*/
#include <iostream>
#include <string>
#include <algorithm>
#include <math.h>
#include "Validator.h"
#include "Action.h"
#include "Utils.h"
#include "typecheck.h"
#include "Exceptions.h"
#include "RobustAnalyse.h"
#include "random.h"
#include <sstream>






using std::stringstream;
using std::make_pair;

//#define map std::map
//#define vector std::vector

namespace VAL {

  bool stepLengthDefault;
  bool makespanDefault;

string getName(plan_step* ps)
{
  string actionName = ps->op_sym->getName();

      for(typed_symbol_list<const_symbol>::const_iterator j = ps->params->begin();
      			j != ps->params->end(); ++j)
      		{
      			actionName += (*j)->getName();
      		};

  return actionName;
};


struct compareActionEndPoints {

	compareActionEndPoints() {};

	bool operator()(pair<double,string> a1,pair<double,string> a2) const
	{

		if(a1.first < a2.first) return true;

		return false;
	};
};

void changeActionTime(const plan * aPlan, string actionName, double newTime)
{

  for(pc_list<plan_step*>::const_iterator i = aPlan->begin(); i != aPlan->end(); ++i)
    {
        if(getName(*i) == actionName)
        {
             (*i)->start_time = newTime;  (*i)->start_time_given = true;
        };

    };

};



DerivationRules::DerivationRules(const derivations_list * d,const operator_list * o) : ops(o),derivPreds()
{
   drvs = new derivations_list();

	for(pc_list<derivation_rule *>::const_iterator i = d->begin(); i != d->end();++i)
	{


     map<string,pair<const goal *,const var_symbol_table *> >::iterator dp = derivPreds.find((*i)->get_head()->head->getName());
	  if(dp != derivPreds.end())
	  {
          if(dp->second.second->size() != (*i)->get_head()->args->size())
          {
   	  		if(LaTeX) *report << "\\\\";
   			*report << "Derived predicates of the same name must have the same number of parameters!\n";
   			if(LaTeX) *report << "\\end{document}\n";
   			DerivedPredicateError dpe;
   			throw dpe;
          };

           //change variables  of second dp to match that of the first
          //parameter_symbol_list::const_iterator k = dp->second.second->begin();
          map<string,var_symbol*>::const_iterator k = dp->second.second->begin();
          map<parameter_symbol*,parameter_symbol*> newvars;

          for(parameter_symbol_list::iterator j = (*i)->get_head()->args->begin(); j != (*i)->get_head()->args->end();++j)
   	    {
            if(dynamic_cast<const var_symbol *>(*j))
   		  {
              newvars[const_cast<parameter_symbol*>(*j)] = k->second;
            };
              k++;
   	    };

         goal * agoal = const_cast<goal*>(dp->second.first);
         goal * bgoal = (*i)->get_body();
         changeVars(bgoal,newvars);

         goal_list * gl = new goal_list();
   		//gl->push_back((*i)->get_body());
   		gl->push_back(agoal);
         gl->push_back(bgoal);

   		const disj_goal * dg = new disj_goal(gl);  repeatedDPDisjs.push_back(dg);

          for(pc_list<derivation_rule *>::iterator h = drvs->begin(); h != drvs->end();++h)
	      {
                if((*h)->get_head()->head->getName() == dp->first) {(*h)->set_body(const_cast<disj_goal*>(dg)); break;};
          };
         derivPreds[dp->first] = make_pair(dg,dp->second.second);


	  }
    else
    {
   	  //build para list for derived predicate painfully
   	  var_symbol_table * vst = new var_symbol_table();
   	  char count = 41; //to ensure ordering of bindings is correct when vst is looped thro which is a map which orders its loop with its key, (we dont want the name of parameter just var sym address and its order of appearance)

   	  for(parameter_symbol_list::const_iterator j = (*i)->get_head()->args->begin(); j != (*i)->get_head()->args->end();++j)
   	  {
   		if(dynamic_cast<const var_symbol *>(*j))
   		{
   		  string s = toString(count++);
   		  (*vst)[s] = const_cast<var_symbol*>(dynamic_cast<const var_symbol *>(*j));

   		};



   	  };



         drvs->push_back(*i);
   		derivPreds[(*i)->get_head()->head->getName()] = make_pair((*i)->get_body(),vst);

    };

	};
};

//this changes the bindings of a DP to match the bindings of its disjuctive counterpart as given by the map
void changeVars(goal * g,map<parameter_symbol*,parameter_symbol*> varMap)
{

  if(dynamic_cast<const comparison*>(g))
	{
      const comparison * comp = dynamic_cast<const comparison*>(g);
		changeVars(const_cast<expression*>(comp->getLHS()),varMap);
      changeVars(const_cast<expression*>(comp->getRHS()),varMap);
	};

	if(dynamic_cast<const conj_goal *>(g))
	{
        const conj_goal * cg = dynamic_cast<const conj_goal *>(g);


        for(pc_list<goal*>::const_iterator i = cg->getGoals()->begin(); i != cg->getGoals()->end(); ++i)
			{
				changeVars(*i,varMap);
			};

	};


	if(dynamic_cast<const disj_goal*>(g))
	{
        const disj_goal * dg = dynamic_cast<const disj_goal *>(g);

        for(pc_list<goal*>::const_iterator i = dg->getGoals()->begin(); i != dg->getGoals()->end(); ++i)
			{
				changeVars(*i,varMap);
			};

	};

	if(dynamic_cast<const neg_goal *>(g))
	{
		changeVars(const_cast<goal*>(dynamic_cast<const neg_goal *>(g)->getGoal()),varMap);
	};



	if(dynamic_cast<const imply_goal*>(g))
	{
		const imply_goal * ig = dynamic_cast<const imply_goal*>(g);
		changeVars(const_cast<goal*>(ig->getAntecedent()),varMap);
		changeVars(const_cast<goal*>(ig->getConsequent()),varMap);
	};

	if(dynamic_cast<const simple_goal*>(g))
	{
		const simple_goal * sg = dynamic_cast<const simple_goal*>(g);

	   for(parameter_symbol_list::iterator i = sg->getProp()->args->begin(); i != sg->getProp()->args->end();++i)
   	  {
        if(dynamic_cast<const var_symbol *>(*i))
   		  {
               // *i = varMap[const_cast<var_symbol*>(dynamic_cast<const var_symbol *>(*i))];
                 map<parameter_symbol*,parameter_symbol*>::const_iterator nv = varMap.find(const_cast<parameter_symbol*>(*i));
                if( nv != varMap.end()) *i = nv->second;
           };
         };
  };


	if(dynamic_cast<const qfied_goal*>(g))
	{
      changeVars(const_cast<goal*>(dynamic_cast<const qfied_goal*>(g)->getGoal()),varMap);
	};


};

void changeVars(expression * e,map<parameter_symbol*,parameter_symbol*> varMap)
{

    if(dynamic_cast<const binary_expression*>(e))
	{
      const binary_expression * be = dynamic_cast<const binary_expression*>(e);
		changeVars(const_cast<expression*>(be->getLHS()),varMap);
      changeVars(const_cast<expression*>(be->getRHS()),varMap);
	};

  if(dynamic_cast<const uminus_expression*>(e))
	{
      const uminus_expression * ue = dynamic_cast<const uminus_expression*>(e);


		changeVars(const_cast<expression*>(ue->getExpr()),varMap);
	};

    if(dynamic_cast<const func_term*>(e))
	{
      const func_term * fe = dynamic_cast<const func_term*>(e);
      parameter_symbol_list *param_list =   const_cast<parameter_symbol_list*>(fe->getArgs());

      for(parameter_symbol_list::iterator i = param_list->begin(); i != param_list->end();++i)
   	  {
        if(dynamic_cast<const var_symbol *>(*i))
   		  {
                //*i = varMap[const_cast<parameter_symbol*>(*i)];
                 map<parameter_symbol*,parameter_symbol*>::const_iterator nv = varMap.find(const_cast<parameter_symbol*>(*i));
                if( nv != varMap.end()) *i = nv->second;
           };
         };
	};
};

bool DerivationRules::stratification() const
{

	map<pair<derivation_rule *,derivation_rule *>, unsigned int> analyseDPs;
	//order DPs
	for(pc_list<derivation_rule *>::const_iterator i = drvs->begin(); i != drvs->end();++i)
	{


		for(pc_list<derivation_rule *>::const_iterator j = drvs->begin(); j != drvs->end();++j)
		{
			analyseDPs[make_pair(*i,*j)] = 0;
		};
	};

	for(pc_list<derivation_rule *>::const_iterator j = drvs->begin(); j != drvs->end();++j)
	{
		for(pc_list<derivation_rule *>::const_iterator i = drvs->begin(); i != drvs->end();++i)
		{
			unsigned int ijoccurNNF = occurNNF(*i,*j); //i appears in j?

			if( ijoccurNNF == 2 )
				analyseDPs[make_pair(*i,*j)] = 2;
			else if( ijoccurNNF == 1 )
				{
					if(analyseDPs[make_pair(*i,*j)] < 1) analyseDPs[make_pair(*i,*j)] = 1;
				};
		};
	};

	for(pc_list<derivation_rule *>::const_iterator i = drvs->begin(); i != drvs->end();++i)
	{
		for(pc_list<derivation_rule *>::const_iterator j = drvs->begin(); j != drvs->end();++j)
		{
		 	for(pc_list<derivation_rule *>::const_iterator k = drvs->begin(); k != drvs->end();++k)
			{
				if( analyseDPs[make_pair(*i,*j)] > 0 && analyseDPs[make_pair(*j,*k)] > 0 )
				{
					unsigned int maxijjk;
					if( analyseDPs[make_pair(*i,*j)] > analyseDPs[make_pair(*j,*k)] )
						maxijjk = analyseDPs[make_pair(*i,*j)];
					else
						maxijjk = analyseDPs[make_pair(*j,*k)];

					if(maxijjk > analyseDPs[make_pair(*i,*k)] ) analyseDPs[make_pair(*i,*k)]  = maxijjk;
				};
			};



		};
	};

	//check DPs can be stratified
	for(pc_list<derivation_rule *>::const_iterator i = drvs->begin(); i != drvs->end();++i)
	{
		if(analyseDPs[make_pair(*i,*i)] == 2) return false;
	};



	if(!Verbose || drvs->size() == 0) return true;


	//extract stratification

	vector< pair<unsigned int,vector<string> > > stratification;

	unsigned int level = 1;
	map< derivation_rule *, unsigned int> remaining;



	for(pc_list<derivation_rule *>::const_iterator i = drvs->begin(); i != drvs->end();++i)
	{
		remaining[*i] = 1;
	};

	while(true){

		vector<string> stratum;
		bool stratfin = true;
		vector< derivation_rule *> toRemove;

		for(map< derivation_rule *, unsigned int>::const_iterator i = remaining.begin(); i != remaining.end();++i)
		{
			if(remaining[i->first] == 1) {stratfin = false; break;};
		};

		if(stratfin) break;

		for(map< derivation_rule *, unsigned int>::const_iterator j = remaining.begin(); j != remaining.end();++j)
		{
			if(remaining[j->first] == 1)

			{
				bool ijr2 = true;
				for(map< derivation_rule *, unsigned int>::const_iterator i = remaining.begin(); i != remaining.end();++i)
				{
					if(remaining[i->first] == 1 && analyseDPs[make_pair(i->first,j->first)] == 2) ijr2 = false;

				};

				if(ijr2)


				{
					stratum.push_back(j->first->get_head()->head->getName());
					toRemove.push_back(j->first);
				};
			};
		};

		for(vector<derivation_rule *>::const_iterator k = toRemove.begin(); k != toRemove.end(); ++k) remaining[*k] = 0;

		stratification.push_back(make_pair(level,stratum));


		++level;

	};

	if(LaTeX)
	{
		*report<< "\\subsection{Stratification}\n";
		for(vector< pair<unsigned int,vector<string> > >::const_iterator i = stratification.begin(); i != stratification.end(); )

		{

			*report << "{\\bf Strata "<<i->first<<":}\\\\\n ";

			for(vector<string>::const_iterator s = i->second.begin(); s != i->second.end(); ++s)
			{
				*report << *s;
				if( s+1 != i->second.end() ) *report <<", ";
			};

          if( ++i != stratification.end()) *report << "\\\\";
			*report << "\n";

		};
	}
	else if(Verbose)

	{
		cout<< "Stratification\n";
		for(vector< pair<unsigned int,vector<string> > >::const_iterator i = stratification.begin(); i != stratification.end(); ++i)

		{
			cout << "Strata "<<i->first<<": ";

			for(vector<string>::const_iterator s = i->second.begin(); s != i->second.end(); ++s)
			{
				cout << *s;
				if( s+1 != i->second.end() ) cout <<", ";
			};

			cout << "\n\n";
		};




	};

	return true;
};

//first DP appears in DP, negatively, positively or neither?
//0 = does not appear, 2 = appears negatively, 1 = appears positively
unsigned int DerivationRules::occurNNF(derivation_rule * drv1,derivation_rule * drv2) const
{
	const goal * g = NNF(new goal(*drv2->get_body()));


   bool ans = occur(drv1->get_head()->head->getName(),g);
   delete g;

	return  ans;
};

//0 = does not appear, 2 = appears negatively, 1 = appears positively (as an atom within formula)
unsigned int DerivationRules::occur(string s,const goal * g) const
{

	if(dynamic_cast<const comparison*>(g))
	{
		return 0;
	};

	if(dynamic_cast<const conj_goal *>(g))

	{

		const conj_goal * cg = dynamic_cast<const conj_goal *>(g);


		unsigned int ans = 0;
		for(pc_list<goal*>::const_iterator i = cg->getGoals()->begin(); i != cg->getGoals()->end(); ++i)
		{
			unsigned int occ = occur(s,*i);
			if(occ > ans) ans = occ;
			if(ans == 2) break;
		};

		return ans;
	};

	if(dynamic_cast<const disj_goal*>(g))
	{

		const disj_goal * dg = dynamic_cast<const disj_goal*>(g);

		unsigned int ans = 0;
		for(pc_list<goal*>::const_iterator i = dg->getGoals()->begin(); i != dg->getGoals()->end(); ++i)
		{


			unsigned int occ = occur(s,*i);
			if(occ > ans) ans = occ;
			if(ans == 2) break;
		};

		return ans;
	};

	if(dynamic_cast<const neg_goal *>(g))

	{
		const neg_goal * ng = dynamic_cast<const neg_goal *>(g);

		if(dynamic_cast<const simple_goal*>(ng->getGoal()) && occur(s,ng->getGoal()) == 1)
			return 2;




		return occur(s,ng->getGoal());
	};

	//shouldn't need this if g is in NNF


	if(dynamic_cast<const imply_goal*>(g))
	{
		const imply_goal * ig = dynamic_cast<const imply_goal*>(g);


		neg_goal * ng = new neg_goal(const_cast<goal *>(ig->getAntecedent()));
		goal_list * gl = new goal_list();
		goal * agoal = const_cast<goal *>(ig->getConsequent());
		gl->push_back(ng);
		gl->push_back(agoal);
		const disj_goal * dg = new disj_goal(gl);

		return occur(s,dg);
	};

	if(dynamic_cast<const simple_goal*>(g))
	{
		const simple_goal * sg = dynamic_cast<const simple_goal*>(g);

		if(isDerivedPred(sg->getProp()->head->getName()))

		{
			if( s == sg->getProp()->head->getName())

			{
				if(sg->getPolarity()==E_POS)
				{
					return 1;
				}
				else
					return 2;
			};
		};

		return 0;
	};


	if(dynamic_cast<const qfied_goal*>(g))
	{
		const qfied_goal * qg = dynamic_cast<const qfied_goal*>(g);


		return occur(s,qg->getGoal());



	};


	return 0;



};



const goal * DerivationRules::NNF(const goal * gl) const
{
	const goal * g = gl;

	if(dynamic_cast<const neg_goal *>(g))
	{

		const goal * ng = (dynamic_cast<const neg_goal *>(g))->getGoal();

		if(dynamic_cast<const conj_goal *>(ng))
		{


			const conj_goal * cg = dynamic_cast<const conj_goal *>(ng);


			goal_list * gl = new goal_list();

			for(pc_list<goal*>::const_iterator i = cg->getGoals()->begin(); i != cg->getGoals()->end(); ++i)
			{
				goal * agoal = const_cast<goal *>(NNF(new neg_goal(new goal(*const_cast<goal*>(*i)))));
				gl->push_back(agoal);
			};

			return new disj_goal(gl);
		};

		if(dynamic_cast<const disj_goal*>(ng))
		{


			const disj_goal * dg = dynamic_cast<const disj_goal*>(ng);

			goal_list * gl = new goal_list();

			for(pc_list<goal*>::const_iterator i = dg->getGoals()->begin(); i != dg->getGoals()->end(); ++i)
			{
				goal * agoal = const_cast<goal *>(NNF(new neg_goal(new goal(*const_cast<goal*>(*i)))));
				gl->push_back(agoal);

			};

			return new conj_goal(gl);
		};

		if(dynamic_cast<const neg_goal *>(ng))
		{
			const neg_goal * nng = dynamic_cast<const neg_goal *>(ng);

			return NNF(nng->getGoal());
		};



		if(dynamic_cast<const imply_goal*>(ng))
		{
			const imply_goal * ig = dynamic_cast<const imply_goal*>(ng);

			neg_goal * nng = new neg_goal(const_cast<goal *>(ig->getConsequent()));
			goal_list * gl = new goal_list();
			goal * agoal = new goal(*const_cast<goal *>(ig->getAntecedent()));


			gl->push_back(nng);
			gl->push_back(agoal);
			const conj_goal * cg = new conj_goal(gl);

			return NNF(cg);
		};



		if(dynamic_cast<const simple_goal*>(ng))
		{
			return new goal(*g);

		};


		if(dynamic_cast<const qfied_goal*>(ng))
		{
			const qfied_goal * qg = dynamic_cast<const qfied_goal*>(ng);


			const qfied_goal * ans;
			if(qg->getQuantifier() == E_EXISTS)
			 	ans = new qfied_goal(E_FORALL,new var_symbol_list(*const_cast<var_symbol_list*>(qg->getVars())),


			 				const_cast<goal*>(NNF(new neg_goal(const_cast<goal*>(qg->getGoal())))),
			 				new var_symbol_table(*const_cast<var_symbol_table*>(qg->getSymTab())));
			else
				ans = new qfied_goal(E_EXISTS,new var_symbol_list(*const_cast<var_symbol_list*>(qg->getVars())),
							const_cast<goal*>(NNF(new neg_goal(const_cast<goal*>(qg->getGoal())))),
							new var_symbol_table(*const_cast<var_symbol_table*>(qg->getSymTab())));


			return ans;


		};
	};


	if(dynamic_cast<const imply_goal*>(g))


	{


		const imply_goal * ig = dynamic_cast<const imply_goal*>(g);

		neg_goal * ng = new neg_goal(const_cast<goal *>(ig->getAntecedent()));
		goal_list * gl = new goal_list();;
		goal * agoal = new goal(*const_cast<goal *>(ig->getConsequent()));
		gl->push_back(ng);
		gl->push_back(agoal);
		const disj_goal * dg = new disj_goal(gl);

		return NNF(dg);
	};

	if(dynamic_cast<const conj_goal *>(g))
	{


		const conj_goal * cg = dynamic_cast<const conj_goal *>(g);


		goal_list * gl = new goal_list();

		for(pc_list<goal*>::const_iterator i = cg->getGoals()->begin(); i != cg->getGoals()->end(); ++i)
		{
			gl->push_back(const_cast<goal*>(NNF(*i)));
		};

		return new conj_goal(gl);
	};


	if(dynamic_cast<const disj_goal*>(g))
	{

		const disj_goal * dg = dynamic_cast<const disj_goal*>(g);


		goal_list * gl = new goal_list();

		for(pc_list<goal*>::const_iterator i = dg->getGoals()->begin(); i != dg->getGoals()->end(); ++i)
		{
				gl->push_back(new goal(*const_cast<goal*>(NNF(*i))));
		};

		return new disj_goal(gl);
	};


	if(dynamic_cast<const qfied_goal*>(g))

	{


		const qfied_goal * qg = dynamic_cast<const qfied_goal*>(g);

		const qfied_goal * ans;
		if(qg->getQuantifier() == E_EXISTS)
		 	ans = new qfied_goal(E_EXISTS,new var_symbol_list(*const_cast<var_symbol_list*>(qg->getVars())),
		 				const_cast<goal*>(NNF(const_cast<goal*>(qg->getGoal()))),
		 				new var_symbol_table(*const_cast<var_symbol_table*>(qg->getSymTab())));
		else
			ans = new qfied_goal(E_FORALL,new var_symbol_list(*const_cast<var_symbol_list*>(qg->getVars())),
						const_cast<goal*>(NNF(const_cast<goal*>(qg->getGoal()))),
		 				new var_symbol_table(*const_cast<var_symbol_table*>(qg->getSymTab())));

		return ans;


	};

	if(dynamic_cast<const simple_goal*>(g))
	{
		return new goal (*g);
	};

	return g;
};

bool DerivationRules::isDerivedPred(string s) const

{
	bool ans = false;

	map<string,pair<const goal *,const var_symbol_table *> >::const_iterator i = derivPreds.find(s);

	if(i  != derivPreds.end())  ans = true;

	return ans;
};


bool DerivationRules::effects(const effect_lists* efflist) const
{
	for(pc_list<simple_effect*>::const_iterator e1 = efflist->add_effects.begin(); e1 != efflist->add_effects.end();++e1)
	{
		if(isDerivedPred((*e1)->prop->head->getName())) return false;
	};

	for(pc_list<simple_effect*>::const_iterator e2 = efflist->del_effects.begin(); e2 != efflist->del_effects.end();++e2)
	{
		if(isDerivedPred((*e2)->prop->head->getName())) return false;
	};

	for(pc_list<forall_effect*>::const_iterator e3 = efflist->forall_effects.begin(); e3 != efflist->forall_effects.end();++e3)
	{
		if(!effects((*e3)->getEffects())) return false;
	};

	for(pc_list<cond_effect*>::const_iterator e4 = efflist->cond_effects.begin(); e4 != efflist->cond_effects.end();++e4)
	{
		if(!effects((*e4)->getEffects())) return false;

	};

	for(pc_list<assignment*>::const_iterator e5 = efflist->assign_effects.begin(); e5 != efflist->assign_effects.end();++e5)
	{
		if(isDerivedPred((*e5)->getFTerm()->getFunction()->getName())) return false;
	};

	return true;

};

bool DerivationRules::effects() const
{
  //check that derived predicates are not used in effects
   for(pc_list<operator_*>::const_iterator i = ops->begin(); i != ops->end();++i)
  {

	for(pc_list<simple_effect*>::const_iterator e1 = (*i)->effects->add_effects.begin(); e1 != (*i)->effects->add_effects.end();++e1)
	{
		if(isDerivedPred((*e1)->prop->head->getName())) return false;
	};

	for(pc_list<simple_effect*>::const_iterator e2 = (*i)->effects->del_effects.begin(); e2 != (*i)->effects->del_effects.end();++e2)
	{
		if(isDerivedPred((*e2)->prop->head->getName())) return false;
	};

	for(pc_list<forall_effect*>::const_iterator e3 = (*i)->effects->forall_effects.begin(); e3 != (*i)->effects->forall_effects.end();++e3)
	{
		if(!effects((*e3)->getEffects()))  return false;
	};

	for(pc_list<cond_effect*>::const_iterator e4 = (*i)->effects->cond_effects.begin(); e4 != (*i)->effects->cond_effects.end();++e4)
	{
		if(!effects((*e4)->getEffects())) return false;


	};

	for(pc_list<assignment*>::const_iterator e5 = (*i)->effects->assign_effects.begin(); e5 != (*i)->effects->assign_effects.end();++e5)
	{
		if(isDerivedPred((*e5)->getFTerm()->getFunction()->getName())) return false;
	};


   };

  return true;
};


DerivationRules::~DerivationRules()
{

  drvs->clear(); delete drvs;

  for(map<string,pair<const goal *,const var_symbol_table *> >::iterator i = derivPreds.begin(); i != derivPreds.end(); ++i)
  {
        delete i->second.second;
  };

  for(vector<const disj_goal *>::iterator j = repeatedDPDisjs.begin(); j != repeatedDPDisjs.end(); ++j)
  {
        const_cast<goal_list*>((*j)->getGoals())->clear();
        delete (*j);
  };

};


Validator::~Validator()
{
	for(vector<Action*>::iterator i = actionRegistry.begin();i != actionRegistry.end();++i)

	{
		delete (*i);
	};

   for(map<const FuncExp *,FEGraph *>::iterator j = graphs.begin(); j != graphs.end(); ++j)
   {


        delete j->second;
     };

	graphs.clear();


	Environment::collect(this);
	delete finalInterestingState;
};

bool DerivationRules::checkDerivedPredicates() const
{
	if(!effects())
	{

		if(LaTeX) *report << "\\\\";
		*report << "A derived predicate appears as an effect!\n";

		return false;
	};



	if(!stratification())


	{
		if(LaTeX) *report << "\\\\";
		*report << "The set of derived predicates do not form a stratified set!\n";

		return false;


	};

	return true;
};

FEGraph::~FEGraph()


{


};

Gantt::~Gantt()
{
  for(map<int, map<int, GanttElement *> >::iterator i = chartRows.begin(); i != chartRows.end(); ++i)
  {
     for(map<int, GanttElement *>::iterator j = i->second.begin(); j != i->second.end(); ++j)
       {
          delete j->second;
         };
    };


};

const double FEGraph::graphMaxH = 4;

const double FEGraph::graphMaxV = 2;

const int FEGraph::pointSize = 100;
const double Gantt::graphH = 4.5;
const double Gantt::graphV = 7.2;
const int Gantt::pointSize = 100;

bool Validator::execute()
{
	bool isOK = true;
  if(theplan.length() == 0) return isOK;
  //	cout << "STATE CURRENTLY: " << state << "\n";
	if(LaTeX)
	{
    setMaxTime(); //needed for drawing graphs
		*report << "\\begin{tabbing}\n";
		*report << "\\headingtimedetails \n";
	}
  else if(Verbose) cout << "Plan Validation details\n-----------------------\n";
	if(finalInterestingState) delete finalInterestingState;
	finalInterestingState = 0;
	followUp = theplan.end();
   bool isReg;

  isOK = events.triggerInitialEvents(this,thisStep.getTime());



  //main loop of plan execution
	while(thisStep != theplan.end())
	{
			if(LaTeX) *report << "\\\\\n";
      else if(Verbose) cout << "\n";

			if(isOK || ContinueAnyway)
			{
        		isReg = (*thisStep)->isRegularHappening();

        		if(isReg && events.hasEvents()) events.updateEventsForMutexCheck(this);

// Might need to modify this compound test if we want to identify repair
// strategies for later parts of a failed plan, since this appears to
// shortcircuit the state progression when trajectory constraints fail
				isOK = tjm.checkAtState(state) && isOK;
				isOK = state.progress(*(thisStep)) && isOK;

        		if(isReg && events.hasEvents() && (isOK || ContinueAnyway))
        			isOK = events.triggerEventsOnInterval(this,false) && isOK;

       			thisStep++;

				if(!finalInterestingState && thisStep != theplan.end() &&
								(*thisStep)->isAfterPlan())
				{
					finalInterestingState = new State(state);
					followUp = thisStep;
				};


			}
			else

			{
				break;
			};

	};

	if(LaTeX) *report << "\\end{tabbing}\n";
	thisStep.deleteActiveFEs();

	return isOK;
	//thisStep==theplan.end();

};

double Validator::getNextHappeningTime() const
{
    Plan::const_iterator aStep = thisStep;

    aStep++;


    if(aStep != theplan.end())
    {
      return aStep.getTime();
    }

      return 0;
};

double Validator::getCurrentHappeningTime() const

{
   Plan::const_iterator aStep = thisStep;
   return aStep.getTime();
};

bool Validator::isLastHappening() const
{
    Plan::const_iterator aStep = thisStep;
    aStep++;

    return (aStep == theplan.end());
};

//for applying event Happenings
bool Validator::executeHappening(const Happening * h)
{
   if(LaTeX) *report << "\\\\\n ";
   return state.progress(h);
};

bool Validator::executeHappeningCtsEvent(const Happening * h)
{
   if(LaTeX) *report << "\\\\\n ";


   return state.progressCtsEvent(h);
};

void Validator::displayPlan() const
{
	try
	{
		if(LaTeX)
			*report << theplan << "\n";
		else
			cout << "Plan to validate:\n\n" << theplan << "\n";
	}
	catch(BadAccessError)
	{
		//ok here just listing the actions....!
	}

};



struct comparePS {


	comparePS() {};

	bool operator()(plan_step * ps1,plan_step * ps2) const
	{

		if(ps1->start_time < ps2->start_time) return true;


		return false;
	};
};


void Validator::displayInitPlanLaTeX(const plan * p) const
{
	//Intended for LaTeX use only

	vector<plan_step*> vps;
	int countNoTimeStamp = 0;


	string act,r;

	for(pc_list<plan_step*>::const_iterator i = p->begin(); i != p->end() ; ++i)
	{
		if(!((*i)->start_time_given)) (*i)->start_time = ++countNoTimeStamp;
		vps.push_back(*i);
	};


	std::sort(vps.begin(),vps.end(),comparePS());

	*report << "Plan size: " << vps.size();
   if(p->getTime() >= 0) *report << "\\\\Planner run time: "<< p->getTime();



   *report <<"\n";
	*report << "\\begin{tabbing}\n";
	*report << "\\headingtimeaction \n";

	for(vector<plan_step*>::const_iterator i1 = vps.begin(); i1 != vps.end() ; ++i1)
	{
		*report << "\\atime{"<<(*i1)->start_time << "}";

		act = " \\> \\listrow{\\action{("+(*i1)->op_sym->getName();

		for(typed_symbol_list<const_symbol>::const_iterator j = (*i1)->params->begin();
				j != (*i1)->params->end(); ++j)
		{

			act += " " + (*j)->getName();


		};

		latexString(act);




		*report << act << ")";

		if((*i1)->duration_given) *report << " ["<<(*i1)->duration<<"]";
		*report << "}}\\\\\n";
	};


	*report << "\\end{tabbing}\n";

};

void Validator::displayInitPlan(const plan * p) const
{

	vector<plan_step*> vps;
	int countNoTimeStamp = 0;

	string act,r;

	for(pc_list<plan_step*>::const_iterator i = p->begin(); i != p->end() ; ++i)
	{
		if(!((*i)->start_time_given)) (*i)->start_time = ++countNoTimeStamp;
		vps.push_back(*i);
	};


	std::sort(vps.begin(),vps.end(),comparePS());

	cout << "Plan size: " << vps.size()<<"\n";

	for(vector<plan_step*>::const_iterator i1 = vps.begin(); i1 != vps.end() ; ++i1)
	{

		cout << " "<<(*i1)->start_time << ": ";

		act = "("+(*i1)->op_sym->getName();


		for(typed_symbol_list<const_symbol>::const_iterator j = (*i1)->params->begin();
			j != (*i1)->params->end(); ++j)
		{
			act += " " + (*j)->getName();
		};

		cout << act << ")";

		if((*i1)->duration_given) cout << " ["<<(*i1)->duration<<"]";

		cout << "\n";
	};

};

void Validator::displayInvariantWarnings() const
{
	for(vector<string>::const_iterator i = invariantWarnings.begin(); i != invariantWarnings.end();++i)
	{
		*report << *i <<"\n";

	};


};

bool Validator::checkGoal(const goal * g)
{
	if(!g) return true;
	const Proposition * p = pf.buildProposition(g);
  //cout << "no of events = "; events.displaynoevents(); //for testing
  //cout << "Checking " << *p << " in " << state << " and getting " << p->evaluate(&state) << "\n";
	DerivedGoal::resetLists(&state);
	bool b = p->evaluate(&state) && tjm.checkFinalState(state);
    if(!b && (Verbose || ErrorReport))
    {
      errorLog.addGoal(p,&state);
      return b;
    }
    else
    {
		if(followUp != theplan.end())
		{
     /*
			// More convenient not to report the re-examination of follow-up states.
			bool v = Verbose;
			Verbose = false;

			bool ltx = LaTeX;
			LaTeX = false;

			while(followUp != theplan.end())
			{
			// First find a state where goal is true (we know there is one, because it
			// is true at the end of the plan).
				while(!p->evaluate(finalInterestingState))
				{
					finalInterestingState->progress(*(followUp++));
				};
				State s(*finalInterestingState);
			// Now check it stays true...
				while(followUp != theplan.end() && p->evaluate(&s))
				{

					s.progress(*(followUp++));
				};
				if(followUp != theplan.end())
				{
					delete finalInterestingState;
					finalInterestingState = new State(s);
				};
			};

			Verbose = v;
			LaTeX = ltx; */
		};


    };

  p->destroy();

	return b;
};

bool Validator::durativePlan() const
{
	return durative;

};


int Validator::simpleLength() const

{
	if(stepLength) return stepcount;  // Default is step count.

	return theplan.length();

};


void Validator::computeMetric(const State * s,vector<double> & v) const
{
	static Environment nullEnv;
	pc_list<expression *>::const_iterator j = metric->expr->begin();
	for(unsigned int i = 0;i < v.size();++i,++j)
	{
		v[i] = s->evaluate(*j,nullEnv);
	}
};

vector<double> Validator::finalValue() const
{
	vector<double> value(metric?metric->opt.size():1);
	bool doneSL = false;

	if(metric && !(makespanDefault && durative))
	{
		if(finalInterestingState)
		{
			computeMetric(finalInterestingState,value);
			return value;
		}
		computeMetric(&state,value);
	}
	else if(durative && makespanDefault)
	{

		value[0] = theplan.lastHappening()->getTime();
	}
	else if(stepLength)
	{
		value[0] = stepcount;  // Default is step count.
		doneSL = true;
	}
	else
	{
		value[0] = theplan.length();
	};

	if(violations.find("anonymous") != violations.end())
	{
		if(metric)
		{
			list<optimization>::const_iterator j = metric->opt.begin();
			for(unsigned int i = 0;i < value.size();++i,++j)
			{
				if(*j == E_MAXIMIZE)
				{
					value[i] += violations.find("anonymous")->second;
				}
				else
				{
					value[i] -= violations.find("anonymous")->second;
				}
			};
		}
		else
		{
			value[0] += violations.find("anonymous")->second;
		};
	};

	if(stepLengthDefault && !doneSL)
	{
		value.push_back(stepcount);
	}

	return value;
};

vector<const_symbol*> Validator::range(const var_symbol * v)
{
	return typeC.range(v);

};

void Validator::setMaxTime()
{
	maxTime = theplan.lastHappening()->getTime();



};

bool Validator::graphsToShow() const
{
	map<double,pair<double,double> >::const_iterator disconts;
	map<double,double>::const_iterator conts;


	for(map<const FuncExp *,FEGraph*>::const_iterator i = graphs.begin(); i != graphs.end(); ++i)
	{

		disconts = i->second->discons.begin();


		conts = i->second->points.begin();

		if( (disconts != i->second->discons.end()) || (conts != i->second->points.end()) ) return true;

	};


	return false;
};


void FEGraph::drawLaTeXAxis(double maxTime) const
{
	double ff = - 0.038; //fudge factor, qbezier and normal lines are slightly out- LaTeXs fault
	*report << "\\put("<<0 + ff<<","<<timeAxisV<<"){ \\vector(1,0){"<<graphMaxH*0.9<<"} }\n";
	*report << "\\put("<<graphMaxH*0.9 +ff<<","<<timeAxisV<<"){ Time }\n";

	*report << "\\put("<<0+ff<<","<<0<<"){ \\vector(0,1){"<<graphMaxV*1.1<<"} }\n";
	*report << "\\put("<<0+ff<<","<<graphMaxV*1.05<<"){ Value }\n";

	*report << "\\put("<<0.02+ff<<","<<timeAxisV - 0.1 <<"){ 0 }\n";
	*report << "\\put("<<graphMaxH*0.9+ff<<","<<timeAxisV<<"){ \\line(0,-1){0.05} }\n";
	*report << "\\put("<<graphMaxH*0.9+ff<<","<<timeAxisV - 0.1 <<"){ "<< maxTime <<" }\n";


	*report << "\\put("<<0+ff<<","<<timeAxisV<<"){ \\line(-1,0){0.05} }\n";
	*report << "\\put("<<-0.2+ff<<","<<timeAxisV<<"){ 0 }\n";



	if(maxValue > 0)
	{
		*report << "\\put("<<0+ff<<","<<graphMaxV<<"){ \\line(-1,0){0.05} }\n";
		*report << "\\put("<<-0.3+ff<<","<<graphMaxV<<"){ "<<maxValue<<" }\n";
	};


	if(minValue < 0)
	{
		*report << "\\put("<<0+ff<<","<<0<<"){ \\line(-1,0){0.05} }\n";
		*report << "\\put("<<-0.3+ff<<","<<0<<"){ "<<minValue<<" }\n";
	}
	else
	{
		*report << "\\put("<< 0+ff <<","<<timeAxisV<<"){ \\line(0,-1){0.05} }\n";
	};

	for(set<double>::const_iterator i = happenings.begin(); i != happenings.end();++i)
	{
		*report << "\\put("<< ( (*i)/maxTime)*graphMaxH*0.9 +ff <<","<<timeAxisV<<"){ \\line(0,-1){0.05} }\n";
	};
};

FEGraph * Validator::getGraph(const FuncExp * fe)
{
	FEGraph * g;
	//find graph or create it if it does not exist
	map<const FuncExp *,FEGraph*>::const_iterator j = graphs.find(fe);

	if(j != graphs.end())
	{
		g = j->second;
	}
	else
	{
		g =  new FEGraph(fe);

		graphs[fe] = g;

	};

	return g;
};

void dround(double & d)
{


	if((d < 0.0001) && (d > -0.0001)) d = 0;
};

void FEGraph::setMinMax()

{
  if(minValue == 0 && maxValue == 0)
  {
      	maxValue = 0;

      	minValue = 0; //change below!



      	for(map<double,double>::const_iterator i = points.begin(); i != points.end();++i)
      	{
      		if( i->second > maxValue) maxValue = i->second;
      		if( i->second < minValue) minValue = i->second;
      	};

      	for(map<double,pair<double,double> >::const_iterator i1 = discons.begin(); i1 != discons.end() ;++i1)


      	{
      		if( i1->second.first > maxValue) maxValue = i1->second.first;
      		if( i1->second.first < minValue) minValue = i1->second.first;
      		if( i1->second.second > maxValue) maxValue = i1->second.second;
      		if( i1->second.second < minValue) minValue = i1->second.second;
      	};

         //minValue = 38;   //change this value to the desired minimum value

      	if((minValue == 0) && (maxValue == 0)) maxValue = 1;
  };

   if(minValue >= 0 )
		timeAxisV = 0;

	else


	{
		timeAxisV = ((- minValue) / (maxValue - minValue) ) * graphMaxV;
		dround(timeAxisV);
	};

    //cout << minValue << " , "<<maxValue <<"\n";

};

void round(pair<double,double> & d)
{

	dround(d.first);
	dround(d.second);
};

void FEGraph::amendPoints(double maxTime)
{
   double tooSmall =  (0.0041*maxTime)/(graphMaxH*0.9);
   map<double,double> copyPoints = points;
   double prevPoint = 0;

   for(map<double,double>::const_iterator i = copyPoints.begin(); i != copyPoints.end(); )
   {
         prevPoint = i->first;
         ++i;     if( i == copyPoints.end()) break;

        if(i->first - prevPoint < tooSmall)
         {
           points.erase(prevPoint);
         };
   };

   copyPoints = points;

   for(map<double,double>::const_iterator j = copyPoints.begin(); j != copyPoints.end(); ++j)
   {
         //no need to have a point on top of a discontinuity

          map<double,pair<double,double> >::const_iterator k = discons.find(j->first);
          if(k != discons.end())
          {
            if((k->second.first - k->second.second > tooSmall) || (k->second.first - k->second.second < -tooSmall))
                points.erase(j->first);
          };
   };

};

void FEGraph::drawLaTeXLine(double t1,double y1,double t2,double y2,double maxTime) const
{
   double graphMin;
   if(minValue > 0 ) graphMin = minValue; else graphMin = 0;
    if((minValue > 0 && (y1 < minValue || y2 < minValue)) || y1 > maxValue || y2 > maxValue ) return;
	double lt1 = (t1/maxTime)*graphMaxH*0.9 ;
	double ly1 = ((y1-graphMin)/(maxValue-minValue))*graphMaxV + timeAxisV;
	double lt2 = (t2/maxTime)*graphMaxH*0.9 ;
	double ly2 = ((y2-graphMin)/(maxValue-minValue))*graphMaxV + timeAxisV;

	//latex dosent like numbers too small
	dround(lt1); dround(ly1);
	dround(lt2); dround(ly2);

	double midValue = (ly1+ly2)/2;
	dround(midValue);


	//latex doesnt like values being too close
	if( (lt2 - lt1 > 0.0041) )
	{


		*report << "\\qbezier("<<lt1<<","<<ly1<<")("<<(lt1+lt2)/2<<","<<midValue<<")("<<lt2<<","<<ly2<<")\n";
	};
};

void FEGraph::displayLaTeXGraph(double maxTime)
{

  amendPoints(maxTime);
	map<double,pair<double,double> >::const_iterator disconts = discons.begin();

	map<double,double>::const_iterator conts = points.begin();

	if( (disconts == discons.end()) && (conts == points.end()) ) return;

	setMinMax();

	if(title == "") *report << "%%--------- Graph of "<<*fe<<" --------------------------------------------------\n";
  else *report << "%%--------- "<<title<<" --------------------------------------------------\n";

	*report << "\\begin{figure}[!ht] \\begin{center} \\setlength{\\unitlength}{"<<pointSize<<"pt}\n";
	*report << "\\begin{picture}("<<graphMaxH*0.9<<","<<graphMaxV*1.16<<")(0,"<<0<<")\n";



	*report << "\\thinlines\n";

	drawLaTeXAxis(maxTime);
	*report << "\\thicklines\n";


	double time,value,tCirc,valCirc;
	double prevTime = 0;
	double nextTime = 0;
	double nextValue = 0,prevValue = 0;



	//setup initial values
//cout << initialValue << " is initialValue for " << *fe << "\n";
	prevTime = initialTime;
	prevValue = initialValue;
	bool ctsActivity;

	//draw it!

	//draw discontinuous segments of graph
	for( ; disconts != discons.end() ; ++disconts)
	{
		//draw cts bit before discontinuity

		if((conts != points.end()) && (conts->first < disconts->first))
		{
			//draw flat line before cts bit
			drawLaTeXLine(prevTime,prevValue,conts->first,conts->second,maxTime);

			prevTime = conts->first;
			prevValue = conts->second;

			ctsActivity = false;

			for( ; conts->first <= disconts->first ; )
			{

				time = conts->first;
				value = conts->second;
				++conts;

				if( (conts == points.end()) || (conts->first > disconts->first) )
				{
					if(!ctsActivity) drawLaTeXLine(prevTime,prevValue,disconts->first,disconts->second.first,maxTime);
					else drawLaTeXLine(nextTime,nextValue,disconts->first,disconts->second.first,maxTime);

					break;
				};


				ctsActivity = true;

				if( conts->first <= disconts->first)
				{
					nextTime = conts->first;

					nextValue = conts->second;
				}

				else

				{
					nextTime = disconts->first;
					nextValue = disconts->second.first;
				};


				drawLaTeXLine(time,value,nextTime,nextValue,maxTime);


			};

			prevTime = disconts->first;
			prevValue = disconts->second.second;



		}
		else



		{
			//no cts activity from last discont, so draw a flat line
			time = disconts->first;
			value = disconts->second.first;
			drawLaTeXLine(prevTime,prevValue,time,value,maxTime);
			prevTime = time;

			prevValue = disconts->second.second;
		};



		//draw little circles to show values at discontinuity


		tCirc = (disconts->first/maxTime)*graphMaxH*0.9;
		valCirc = (disconts->second.first/(maxValue-minValue))*graphMaxV + timeAxisV;

		//latex dosent like numbers too small

		dround(valCirc); dround(tCirc);

		*report << "\\put("<<tCirc<<","<<valCirc<<"){\\circle{0.04}}\n";

		tCirc = (disconts->first/maxTime)*graphMaxH*0.9;
		valCirc = (disconts->second.second/(maxValue-minValue))*graphMaxV + timeAxisV;
		//latex dosent like numbers too small
		dround(valCirc); dround(tCirc);

		*report << "\\put("<<tCirc<<","<<valCirc<<"){\\circle*{0.04}}\n";



	};

	//draw line before cts bit
	if( (conts != points.end()) && (conts->first - prevTime > 0.002) )
	{
		drawLaTeXLine(prevTime,prevValue,conts->first,conts->second,maxTime);
    nextTime = conts->first;
		nextValue = conts->second;
	};

	//draw last cts bit
	for( ; conts != points.end() ; )

	{
		time = conts->first;
		value = conts->second;

		++conts;
		if(conts == points.end()) break;

		nextTime = conts->first;
		nextValue = conts->second;

		drawLaTeXLine(time,value,nextTime,nextValue,maxTime);



	};

	//last drawn time value?

	if(nextTime > prevTime)
	{
		prevTime = nextTime;
		prevValue = nextValue;
	};

	//draw last bit
	if(maxTime - prevTime > 0.002)
	{
		drawLaTeXLine(prevTime,prevValue,maxTime,prevValue,maxTime);
	};


	if(title == "") *report << "\\end{picture} \\caption{Graph of "<<*fe<<".}\n";
  else *report << "\\end{picture} \\caption{"<<title<<".}\n";
	*report << "\\end{center} \\end{figure} \n";
	*report << "%%-----------------------------------------------------------\n";
};



void Validator::displayLaTeXGraphs() const
{
	for(map<const FuncExp *,FEGraph*>::const_iterator i = graphs.begin(); i != graphs.end(); ++i)
	{
		i->second->displayLaTeXGraph(maxTime);

	};


};

void Validator::drawLaTeXGantt(int noPages,int noPageRows)
{
	gantt.drawLaTeXGantt(theplan,noPages,noPageRows);
};

int Gantt::getNoPages(int np)

{
	if(np != 0) return np;

	int noPages = 1;
	int countNonDAEl = 0;

	double smallElement = 0.04; //smallest acceptable element, no is percentage of length of time line
								//if too many of these extra pages will appear
	double length;

	int totalEls = 0;
	for(map<int, map<int, GanttElement *> >::const_iterator r = chartRows.begin(); r != chartRows.end() ; ++r)
	{
		totalEls += r->second.size();
	};

	//too many small elements is 50% of the elements
	int tooMany = int(double(totalEls*0.50));

	map<int,int> smallElCount;


	for(map<int, map<int, GanttElement *> >::const_iterator r1 = chartRows.begin(); r1 != chartRows.end() ; ++r1)
	{
		for(map<int, GanttElement *>::const_iterator ge = r1->second.begin() ; ge != r1->second.end() ; ++ge)
		{
			if(ge->second->start != ge->second->end)
			{

				length = (ge->second->end - ge->second->start)/maxTime;

				for(int n = 2 ; ; ++n)
				{

					if(	length >= smallElement/n )
					{
						break;

					}
					else if(	(length < smallElement/n)
					   		&& (length >= smallElement/(n+1) ))
					{
						++smallElCount[n];
						break;

					};



				};
			}
			else
			{
				++countNonDAEl;
			};
		};
	};

	int totalSmallEls = 0;


	for(map<int,int>::reverse_iterator i = smallElCount.rbegin(); i != smallElCount.rend(); ++i)
	{
		totalSmallEls += i->second;
		if(totalSmallEls > tooMany)
		{
			noPages = i->first;

			break;
		};

	};

	return noPages;

};

int Gantt::getNoPageRows()
{


	int tooManyRows = 31;
	int noRows = chartRows.size();

	return int(double(noRows/tooManyRows)) + 1;






};

void Gantt::drawLaTeXGantt(const Plan & p,int noPages,int noPageRows)
{
	setMaxTime(p);
	//setSigObjs();
	buildRows(p);

	shuffleRows();

	displayKey();

	//split across more than one page?
	int noOfPages = getNoPages(noPages);
	int noOfPageRows;
	int numRowsPg1 = 0;


	if(noPageRows != 0)

		noOfPageRows = noPageRows;
	else
		noOfPageRows = getNoPageRows();

	int numRows = chartRows.size();
	int startRow,endRow;
	int rowsPerPage = int(double(numRows)/double(noOfPageRows) + 0.5);
	//draw the pages!
	for(int pg = 1; pg <= noOfPages; ++pg)
	{
		for(int pgr = 1; pgr <= noOfPageRows; ++pgr)

		{
			startRow = (pgr-1)*rowsPerPage + 1;

			endRow = pgr*rowsPerPage;
			if(pg == 1) numRowsPg1 = endRow - startRow + 1;

			if(endRow > numRows || pgr == noOfPageRows) endRow = numRows;


			drawLaTeXGantt( (maxTime/noOfPages)*(pg-1), (maxTime/noOfPages)*pg,startRow,endRow,numRowsPg1);
		};
	};

};


vector<string> Gantt::getSigObjs(const Action * a)
{

	string par;

	vector<string> so;
	//vector<string>::iterator i;

	for(var_symbol_list::const_iterator i = a->getAction()->parameters->begin() ; i != a->getAction()->parameters->end(); ++i)
	{
		par = a->getBindings().find(*i)->second->getName();
		//is parameter a sigificant object?
		vector<string>::iterator ii = std::find(sigObjs.begin(),sigObjs.end(),par);
		if(ii != sigObjs.end())
		{

			so.push_back(par);

			//add to list of used sig objs if nec
			vector<string>::iterator j = std::find(usedSigObjs.begin(),usedSigObjs.end(),par);
			if(j == usedSigObjs.end()) usedSigObjs.push_back(par);
		};





	};


	return so;
};


void Validator::setSigObjs(vector<string> & objects)
{
	gantt.setSigObjs(objects);
};



void Gantt::setSigObjs(vector<string> & objects)


{
	for(vector<string>::const_iterator o = objects.begin(); o != objects.end(); ++o)
	{

		sigObjs.push_back(*o);
	};
};

void Gantt::buildRows(const Plan & p)
{
	int row = 0;


	int pos = 1;


	double largestTime;

	double start,end;
	string label;
	vector<string> actSigObjs;


	vector<int> sigRows;

	vector<const Action*> actions;

	for(Plan::const_iterator i = p.begin() ; i != p.end() ; ++i)
	{
		if(i.isRegular())
		{
			actions = *((*i)->getActions());

			for(vector<const Action*>::const_iterator a = actions.begin();a != actions.end();++a)
			{
				if(!(dynamic_cast<EndAction*>(const_cast<Action*>(*a))))
				{



					//find sig Objs
					actSigObjs = getSigObjs(*a);


					//find rows where last element has a sig obj and finishes before this one
					sigRows.clear();


					for(map<int, map<int, GanttElement *> >::const_iterator r = chartRows.begin(); r != chartRows.end() ; ++r)
					{
						map<int, GanttElement *>::const_iterator ge = r->second.end();
						--ge;

						if(ge->second->end < (*i)->getTime())
						{
							for(vector<string>::const_iterator so = actSigObjs.begin(); so != actSigObjs.end(); ++so)
							{

								vector<string>::iterator fso = std::find(ge->second->sigObjs.begin(),ge->second->sigObjs.end(),*so);
								if(fso != ge->second->sigObjs.end()) sigRows.push_back(r->first);
							};
						};

					};

					if(sigRows.size() == 1)
					{
						row = *(sigRows.begin());

					}
					else if(sigRows.size() != 0)


					{
						//choose largest one
						largestTime = 0;

						for(vector<int>::const_iterator sr = sigRows.begin(); sr != sigRows.end(); ++sr)
						{
							map<int, GanttElement *>::const_iterator ge = chartRows[*sr].end();
							--ge;


							if( ge->second->end > largestTime )
							{
								largestTime = ge->second->end;
								row = *sr;
							};

						};


					}
					else
					{

						if(actSigObjs.size() != 0)
						{
							row = chartRows.size() + 1;
						}

						else


						{

							largestTime = -1;//maxTime + 1;
							//find the largest last time smaller than start time of this action from rows - dont choose row with sig objs
							for(map<int, map<int, GanttElement *> >::const_iterator r = chartRows.begin(); r != chartRows.end() ; ++r)
							{

								map<int, GanttElement *>::const_iterator ge = r->second.end();
								--ge;

								if( (ge->second->end > largestTime) && (ge->second->end < (*i)->getTime()) && (getSigObj(r->first) == "") )
								{
									largestTime = ge->second->end;
									row = r->first;
								};


							};


							if(largestTime == -1)
							{
								row = chartRows.size() + 1;

							};

						};
					};





					start = (*i)->getTime();


					if(StartAction * sa = dynamic_cast<StartAction*>(const_cast<Action*>(*a)))

					{
						end = (*i)->getTime() + (*sa).getDuration();
                  double dur = (*sa).getDuration();
						label = "\\action{"+(*a)->getName() + " [" + toString( dur ) + "]}";
					}
					else
					{
						end = (*i)->getTime();
						label = toString(*a);
					};


					chartRows[row][pos++] = new GanttElement( start, end, label, actSigObjs );
				};
			};

		};

	};




};

void Gantt::swapRows(int r1,int r2)

{
	map<int, GanttElement *> row1 = chartRows[r1];
	chartRows[r1] = chartRows[r2];
	chartRows[r2] = row1;
};

//move r2 into r1's place and move the rows inbetween down one row
void Gantt::insertRow(int r1,int r2)
{




	for(int i = r2; i >= r1 + 1 ; --i)
	{
		swapRows(i,i-1);
	};
};


string Gantt::getSigObj(int r)
{
	//determine sigificant object for this row
	map<string,int> countSigObjs;
	string	ans;
	int most;

	//count no of each sig obj
	for(map<int,GanttElement *>::const_iterator ge = chartRows[r].begin(); ge != chartRows[r].end(); ++ge)
	{
		for(vector<string>::const_iterator so = sigObjs.begin(); so != sigObjs.end(); ++so)
		{
			vector<string>::const_iterator fso = std::find(ge->second->sigObjs.begin(),ge->second->sigObjs.end(),*so);
			if(fso != ge->second->sigObjs.end()) countSigObjs[*so] = countSigObjs[*so] + 1;
		};

	};

	if(countSigObjs.size() == 0)


		return "";

	else
	{
		most = 0;

		for(map<string,int>::const_iterator i = countSigObjs.begin() ; i != countSigObjs.end() ; ++i)
		{
			if(i->second > most)
			{
				most = i->second;
				ans = i->first;

			};

		};

	};

	return ans;

};

void Gantt::shuffleRows()
{
	//put rows with the same sig objs together if possible

	if((chartRows.size() < 3) || (usedSigObjs.size() == 0)) return;

	string sigObj1,sigObj2,sigObj3;
	int rowToTake;
	bool alltheSameSigObj;

	for(unsigned int row = 1; row < chartRows.size() - 1 ; ++row)
	{
		sigObj1 = getSigObj(row);



		rowToTake = row + 1;



		for(unsigned int testRow = row + 2; testRow <= chartRows.size() ; ++testRow)
		{

			sigObj2 = getSigObj(testRow);



			if(sigObj1 == sigObj2)
			{
				//is everyrow between these two rows the same sigObj?
				alltheSameSigObj = true;
				for(unsigned int i = rowToTake; i <= testRow - 1 ; ++i)
				{

					sigObj3 = getSigObj(i);
					if(sigObj3 != sigObj2)
					{
						alltheSameSigObj = false;
						break;
					};
				};



				if(!alltheSameSigObj) insertRow(rowToTake,testRow);


				++rowToTake;
			};

		};

	};

};

void Gantt::displayKey()
{
	int pos = 1;
	int totalNum = 0;
	string sigObj,lastSigObj,colour;


	for(map<int, map<int, GanttElement *> >::const_iterator i = chartRows.begin(); i != chartRows.end() ; ++i)
	{
		totalNum += i->second.size();
	};

	*report << "{\\bf Gantt Chart Key}\\\\\n";


	for(map<int, map<int, GanttElement *> >::const_iterator r = chartRows.begin(); r != chartRows.end() ; ++r)
		{
			sigObj = getSigObj(r->first);

			if(sigObj != "")
			{
				*report << "Row "<< r->first <<" : ";
				colour = getColour(r->first);
				if(colour != "") *report << "\\color"<<colour;
				*report << "\\exprn{" << sigObj <<"}\n";
				if(colour != "") *report << "\\normalcolor\n";
			}
			else

				*report << "Row "<< r->first << "\n";




			 *report << "\\begin{tabbing}\n"
				  << "{\\bf No} \\qquad \\= {\\bf Time} \\qquad \\= {\\bf Action} \\\\\n";

			for(map<int, GanttElement *>::const_iterator j = r->second.begin() ; j != r->second.end() ; ++j)
			{


				*report << j->first << " \\> " << j->second->start << " \\> \\listrowg{" << j->second->label << "} \\\\\n";
				++pos;


			};


			*report << "\\end{tabbing}\n";

		};

};

string Gantt::getColour(int row)
{


	string sigObj = getSigObj(row);
	if(sigObj == "") return "";

	string ans = "";

	double r = 0,b = 0,g = 0;
	double cyc = 0;

	int colourNum = 1;
	for(vector<string>::iterator so = usedSigObjs.begin(); so != usedSigObjs.end(); ++so)


	{
		if(sigObj == *so) break;

		++colourNum;
	};

	for(int j = 1; j <= colourNum; ++j)
	{

		if( (g == 1) && (b == 1) && (r == 1) ) {g = 0; r = 0; b = 0; cyc += 1;};

		b += 1;

		if( b == 2)
		{
			b = 0;

			r += 1;
		};

		if( r == 2)
		{
			r = 0;
			g += 1;
		};


		if( g == 2)
		{
			g = 0;
			b = 1;
		};

	};

	//yellow to orange
	if( (g == 1) && (b == 0) && (r == 1) ) {g = 0.5; r = 1; b = 0;};

	//white to grey

	if( (g == 1) && (b == 1) && (r == 1) ) {g = 0.5; r = 0.5; b = 0.5;};


	//swap ordering of green and magenta
	if( (g == 0) && (b == 1) && (r == 1) ) {g = 1; r = 0; b = 0;}
	else if( (g == 1) && (b == 0) && (r == 0) ) {g = 0; r = 1; b = 1;};


	if( r != 0 || b != 0 || g != 0)
		ans = "[rgb]{"+toString(r*pow(0.7,cyc))+","+toString(g*pow(0.75,cyc))+","+toString(b*pow(0.65,cyc))+"}";
	else
		return "";

	return ans;

};

void Gantt::drawLaTeXGantt(double startTime,double endTime,int startRow,int endRow,int numRows)
{
	double ff = - 0.038; //fudge factor, qbezier and normal lines are slightly out- LaTeXs fault
	double y;
	string colour;


	*report << "%%---------------------------------------------------------\n";
	*report << "\\begin{figure} \\begin{center} \\setlength{\\unitlength}{"<<80<<"pt}\n";
	*report << "\\begin{picture}("<<graphH<<","<<graphV<<")(0,0)\n";

	*report << "\\put("<<ff<<","<<0<<"){ \\vector(0,1){"<<graphV*0.9<<"} }\n";
	*report << "\\put("<<ff<<","<<graphV*0.9<<"){ \\begin{sideways} Time \\end{sideways} }\n";
	*report << "\\put("<<ff<<","<<0<<"){ \\line(-1,0){0.05} }\n";
	*report << "\\put("<<ff+ 0.02 - 0.12<<", "<<0<<"){\\begin{sideways} "<< startTime <<" \\end{sideways}}\n";
	*report << "\\put("<<ff<<","<<graphV*0.9<<"){ \\line(-1,0){0.05} }\n";
	*report << "\\put("<<ff -0.12<<","<<graphV*0.9+0.05<<"){\\begin{sideways} "<< endTime <<" \\end{sideways}}\n";

	drawRowNums(startRow,endRow,numRows);

	for(map<int, map<int, GanttElement *> >::const_iterator i = chartRows.begin(); i != chartRows.end() ; ++i)
	{

		if(i->first >= startRow && i->first <= endRow)
		{
			//draw little lines on time axis
			*report << "\\normalcolor\n";
			for(map<int, GanttElement *>::const_iterator j = i->second.begin() ; j != i->second.end() ; ++j)
			{
				if((j->second->start < endTime) && (j->second->start >= startTime))
				{
					y =  (((j->second->start - startTime)/(endTime-startTime)))*graphV*0.9;
					dround(y);
					*report << "\\put("<<ff<<","<<y<<"){ \\line(-1,0){0.05} }\n";

					if( j->second->start != j->second->end )

					{

						if((j->second->end <= endTime) && (j->second->end > startTime))
						{

							y =  (((j->second->end - startTime)/(endTime-startTime)))*graphV*0.9;
							dround(y);

							*report << "\\put("<<ff<<","<<y<<"){ \\line(-1,0){0.05} }\n";
						};
					};
				};
			};


			//draw elements
			colour = getColour(i->first);
			if(colour != "") *report << "\\color"<<colour<<"\n";
			else *report << "\\normalcolor\n";

			for(map<int, GanttElement *>::const_iterator j1 = i->second.begin() ; j1 != i->second.end() ; ++j1)
			{

				if(! (  ((j1->second->start < startTime) && (j1->second->end < startTime))

					  ||((j1->second->start > endTime) && (j1->second->end > endTime)) )
				  )
				{
					if( (j1->second->start != j1->second->end) )
					{
						drawLaTeXDAElement(j1->second,i->first - startRow + 1,j1->first,startTime,endTime,numRows);
					}

					else
					{
						drawLaTeXElement(j1->second,i->first - startRow + 1,j1->first,startTime,endTime,numRows);
					};
				};
			};


		};
	};

	*report << "\\normalcolor\n";
	*report << "\\put("<<graphH + 0.1 <<","<<graphV/2 - 1<<"){\\rotcaption{Gantt Chart}} \\end{picture} \n";
	*report << "\\end{center} \\end{figure} \n";
	*report << "%%-----------------------------------------------------------\n";


};


void Gantt::setMaxTime(const Plan & p)
{
	double max = 0;

	for(Plan::const_iterator i = p.begin() ; i != p.end() ; ++i)
	{
		if( (*i)->getTime() > max ) max = (*i)->getTime();
	};

	maxTime = max;



};

void Gantt::drawRowNums(int startRow,int endRow,int numRows)
{

	double spacing = graphH/numRows;

	for(int r = endRow; r >= startRow ; --r)
	{
		*report << "\\put("<< spacing*(r - startRow + 0.5) <<","<<0 - 0.15<<"){\\begin{sideways}"<<r<<" \\end{sideways}}\n";

	};

};

void Gantt::drawLaTeXDAElement(const GanttElement * ge,int row,int pos,double startTime,double endTime,int numRows) const
{

	double ff = - 0.0515;
	double ff2 = 0.005;



	double geStart,geEnd;


	if(ge->start < startTime)
		geStart = startTime;
	else
		geStart = ge->start;

	if(ge->end > endTime)
		geEnd = endTime;
	else
		geEnd = ge->end;

	if(geEnd > endTime) geEnd = endTime;


	double posX = ((geStart - startTime)/(endTime-startTime))*graphV*0.9;

	double posY = (graphH/numRows)*(numRows - row + 0.9);

	pair<double,double> p1 = transPoint(posX,posY);

	round(p1);


	double length = ((geEnd - geStart)/(endTime-startTime))*graphV*0.9;
	double height = (graphH/numRows)*0.9;
	dround(height); dround(length);


	if( (ge->start >= startTime) && (ge->end <= endTime) )
	{
		*report << "\\put("<<p1.first<<","<<p1.second<<"){\\framebox("<<height<<","<<length<<"){ \\begin{sideways} "<<pos<<"  \\end{sideways}}}\n";
	}
	else if(ge->start < startTime)

	{
		height += ff2;

		*report << "\\put("<<ff+p1.first<<","<<p1.second<<"){ \\line(0,1){"<<length<<"} }\n"
			 << "\\put("<<ff+p1.first<<","<<p1.second+length<<"){ \\line(1,0){"<<height<<"} }\n"
		     << "\\put("<<ff+p1.first + height<<","<<p1.second<<"){ \\line(0,1){"<<length<<"} }\n"
		     << "\\put("<<ff+p1.first + height/2<<","<<p1.second + length/2<<"){ \\begin{sideways} "<<pos<<"  \\end{sideways} }\n";



	}

	else if(ge->end > endTime)

	{
		height += ff2;

		*report << "\\put("<<ff+p1.first<<","<<p1.second<<"){ \\line(0,1){"<<length<<"} }\n"
			 << "\\put("<<ff+p1.first<<","<<p1.second<<"){ \\line(1,0){"<<height<<"} }\n"
		     << "\\put("<<ff+p1.first+height<<","<<p1.second<<"){ \\line(0,1){"<<length<<"} }\n"
		     << "\\put("<<ff+p1.first + height/2<<","<<p1.second + length/2<<"){ \\begin{sideways} "<<pos<<"  \\end{sideways} }\n";


	};


};

void Gantt::drawLaTeXElement(const GanttElement * ge,int row,int pos,double startTime,double endTime,int numRows) const
{



	int numBoxes;

	if(numRows > 24)
		numBoxes = 1;
	else
		numBoxes = 24 / numRows; //num of boxes per row



	int boxPos = pos % numBoxes;
	double boxSize = 0.1;
	double height = (graphH/numRows)*0.9;
	double posX = ((ge->start - startTime)/(endTime-startTime))*graphV*0.9;
	double posY = (graphH/numRows)*(numRows - row);
	double boxX = posX - boxSize/2;
	double boxY = posY + boxPos*(height/numBoxes) + boxSize/2;


	pair<double,double> p11 = transPoint(posX,posY);
	pair<double,double> p13 = transPoint(posX,boxY - boxSize/2);


	pair<double,double> p21 = transPoint(posX,boxY + boxSize/2);
	pair<double,double> p23 = transPoint(posX,posY + height);

	pair<double,double> p4 = transPoint(boxX,boxY);

	round(p11); round(p13);
	round(p21); round(p23);

	*report << "\\put("<<p11.first<<","<<p11.second<<"){ \\line(-1,0){"<<p11.first - p13.first<<"} }\n";

	*report << "\\put("<<p4.first<<","<< p4.second <<"){\\framebox("<<boxSize<<","<<boxSize<<")[l]{ \\begin{sideways} {\\tiny "<<pos<<"  } \\end{sideways}}}\n";

	*report << "\\put("<<p21.first<<","<<p21.second<<"){ \\line(-1,0){"<<p21.first - p23.first<<"} }\n";

};

pair<double,double> Gantt::transPoint(double x,double y) const
{

	return make_pair(graphH-y,x);

};


void PlanRepair::advice(ErrorLog & el)
{
    if(el.getConditions().size() == 0) return;
    if(LaTeX) *report << "\\subsection{Plan Repair Advice}\n";
    else cout << "\nPlan Repair Advice:\n";

    if(LaTeX) *report << "\\begin{enumerate}\n";
    for(vector<const UnsatCondition *>::const_iterator i = el.getConditions().begin(); i != el.getConditions().end(); ++i)
    {


       (*i)->advice();

    };

     if(LaTeX) *report << "\\end{enumerate}\n";

};



void PlanRepair::firstPlanAdvice()
{

    advice(v.getErrorLog());

    //repair plans when reporting an error report!
    if(ErrorReport && getUnSatConditions().size() != 0) repairPlan();
};

string getName(const plan_step* ps)
{
  string actionName = ps->op_sym->getName();

      for(typed_symbol_list<const_symbol>::const_iterator j = ps->params->begin();
      			j != ps->params->end(); ++j)

      		{

      			actionName += (*j)->getName();
      		};


  return actionName;

};



double Validator::timeOf(const Action * a) const
{
	return theplan.timeOf(a);
};

pair<double,double> getSlideLimits(set<double> & actionTimes,double & actionTime,double & deadLine)
{
  pair<double,double> slideLimits = make_pair(actionTime,actionTime);
  double lastTime = 0;


  for(set<double>::const_iterator i = actionTimes.begin(); i != actionTimes.end(); ++i)
  {
        if(*i == actionTime)
        {
           slideLimits.first = lastTime;
           ++i;
           if(i != actionTimes.end()) slideLimits.second = *i;
           else slideLimits.second = deadLine;

           break;
        };
        lastTime = *i;
  };

   return slideLimits;
};

pair<double,double> getSlideLimits(const plan * aPlan, string actionName,double currentTime,double deadLine)
{
  set<double> actionTimes;
  pair<double,double> slideLimits;
  double startTime = 0;
  double endTime = -1;
  double duration;

  for(pc_list<plan_step*>::const_iterator i = aPlan->begin(); i != aPlan->end(); ++i)
  {
        actionTimes.insert((*i)->start_time);
        if((*i)->duration_given) actionTimes.insert((*i)->start_time + (*i)->duration);

        if(getName(*i) == actionName && (*i)->start_time == currentTime)
        {
             startTime = (*i)->start_time;
             if((*i)->duration_given)
             {
               endTime = (*i)->start_time + (*i)->duration;
               duration = (*i)->duration;
             };
        };
  };

  slideLimits = getSlideLimits(actionTimes,startTime,deadLine);

  if(endTime != -1)
  {
     pair<double,double> endSlideLimits = getSlideLimits(actionTimes,endTime,deadLine);
     if(slideLimits.first < (endSlideLimits.first - duration) && (endSlideLimits.first != startTime)) slideLimits.first = endSlideLimits.first - duration;
     if(!(slideLimits.second < (endSlideLimits.second - duration) && (slideLimits.second != endTime))) slideLimits.second = endSlideLimits.second - duration;
  };

  return slideLimits;
};

double getMaxTime(const plan * aPlan)
{
  double maxTime = 0;

  for(pc_list<plan_step*>::const_iterator i = aPlan->begin(); i != aPlan->end(); ++i)
  {
        if((*i)->start_time > maxTime) maxTime = (*i)->start_time;
        if((*i)->duration_given && (((*i)->start_time + (*i)->duration) > maxTime)) maxTime = (*i)->start_time + (*i)->duration;
  };

  return maxTime;
};

plan * newTestPlan(const plan * p)
{
 plan * newPlan = new plan();
 if(p != 0)
 for(pc_list<plan_step *>::const_iterator ps = p->begin(); ps != p->end(); ++ps)
 {
   if(*ps==0)
   {
   	cout << "Got a bad step here\n";
   }
   newPlan->push_back(new plan_step(**ps));
 };

 return newPlan;
};

void deleteTestPlan(plan * p)
{

 for(pc_list<plan_step *>::iterator ps = p->begin(); ps != p->end(); ++ps)
 {
   (*ps)->params = 0;
   (*ps)->op_sym = 0;
 };

  delete p;
};

void changeActionTime(const plan * aPlan, string actionName, double currentTime, double newTime)//,include double current time
{

  for(pc_list<plan_step*>::const_iterator i = aPlan->begin(); i != aPlan->end(); ++i)
    {            //  cout <<  getName(*i) << " == " << actionName <<" && "<< (*i)->start_time <<" == "<<currentTime<<"\n";
        if(getName(*i) == actionName && (*i)->start_time == currentTime)
        {


             (*i)->start_time = newTime;  (*i)->start_time_given = true;
        };

    };

};

//returns 1/2, 1/4, 3/4, 1/8, 3/8, 5/8, 7/8, 1/16, 3/16, 5/16, 7/16, 9/16, 11/16, 13/16, 15/16....
double aNumber(int i)
{
  double denom;
  int denomPower;
  for(int j = 1; ; ++j)
  {
    if(i < pow(2,j)) {denomPower = j; break;};
  };
  denom = pow(2,denomPower);

  return (2*(i - pow(2,denomPower-1) + 1) - 1)/denom;

};

void PlanRepair::repairPlan()
{
 v.setMaxTime();
 bool latex = LaTeX;
 bool verbose = Verbose;
 LaTeX = false; Verbose = false;
 ErrorReport = true;
 ContinueAnyway = false;
 bool planRepaired = false;

 if(!latex) cout << "\nRepairing plan...\n";
 if(!p) return;
 plan * repairingPlan = new plan(*p);

 //idenitify faults with the current plan and decide how the plan should be fixed
 set<const Action *> flawedActions = getUniqueFlawedActions(&v);
 const plan_step * firstPlanStep = 0;

 if(!flawedActions.empty()) firstPlanStep = (*(flawedActions.begin()))->getPlanStep();

 //call functions to fix the plan
 if(firstPlanStep != 0)
 {
   planRepaired = slideEndOfPlan(repairingPlan,firstPlanStep);
   if(!planRepaired) planRepaired = repairPlanOneActionAtATime(repairingPlan,firstPlanStep);
   if(!planRepaired) planRepaired = shakePlan(repairingPlan,firstPlanStep,int(deadLine/2)); //need to set the variation somehow, half the deadLine time seems to work pretty good
 };

 //report on success of plan repair
 LaTeX = latex; Verbose = verbose;

 //add in actions in timed initial literals that are not really timed initial literals
 for(vector<plan_step *>::const_iterator i = timedInitialLiteralActions.begin(); i != timedInitialLiteralActions.end(); ++i)
 {
    if(!((*i)->op_sym->getName().length() > 28 && (*i)->op_sym->getName().substr(0,28) == "Timed Initial Literal Action"))
    {
      repairingPlan->push_back(*i);
    };
 };

 if(planRepaired)
 {
    if(LaTeX) *report << "\\subsubsection{Similar Valid Plan}\n";
    else cout << "\nA valid similar plan to the plan given is:\n";

    if(LaTeX) v.displayInitPlanLaTeX(repairingPlan);
    else v.displayInitPlan(repairingPlan);

 }
 else
 {
     if(LaTeX) *report << "\\subsubsection{Similar Valid Plan}\n Failed to find a valid similar plan {\\begin{rotate}{270}:-( \\end{rotate}}";
     else cout << "\nFailed to find a valid similar plan :-(\n";

 };

 repairingPlan->clear(); delete repairingPlan;

};

//for every action after and including firstAction shift the start time by shiftTime, taking into account that this plan is already shifted
//starttime is the original time of the first action
void moveActionTimes(const plan * aPlan,const plan_step * firstAction,double startTime, double shiftTime)//,include double current time
{
  bool shift = false;
  double currentShift;

  for(pc_list<plan_step*>::const_iterator i = aPlan->begin(); i != aPlan->end(); ++i)
    {
        if(*i == firstAction)
        {
          shift = true;
          currentShift = (*i)->start_time - startTime;

        };

        if(shift)
        {
             (*i)->start_time += shiftTime - currentShift;
        };

    };

};


bool PlanRepair::slideEndOfPlan(const plan * repairingPlan,const plan_step * firstAction)
{
   double endOfPlanTimeLength = getMaxTime(repairingPlan) - firstAction->start_time;
   double originalStartTime = firstAction->start_time;
   double shiftMax = deadLine - originalStartTime - endOfPlanTimeLength;
   if(shiftMax <= 0) return false;

   bool searching = true;
   double shiftTime;
   double lastShiftTime = 0;
   int counter = 1;
   bool planRepaired = false;
   Validator * planRepairValidator = 0;
   bool anError;

   while(searching)
   {
         shiftTime = shiftMax*aNumber(counter++); //start time w.r.t. the first action in the end section of actions
         moveActionTimes(repairingPlan,firstAction,originalStartTime,shiftTime);

         plan * testPlan = newTestPlan(repairingPlan);
         plan * testPlan2 = new plan(*testPlan);

          //add timed initial literals to the plan from the problem spec, these time are fixed
          for(vector<plan_step *>::iterator ps = timedInitialLiteralActions.begin(); ps != timedInitialLiteralActions.end(); ++ps)
          {
             testPlan->push_back(*ps);
          };

         planRepairValidator = new Validator(v.getDerivRules(),v.getTolerance(),typeC,operators,initialState,testPlan,
                          metric,stepLength,durative,current_analysis->the_domain->constraints,current_analysis->the_problem->constraints);

         anError = false;
         try{
           planRepairValidator->execute();
         }
         catch(exception & e)
         {
             cout << e.what() << "\n"; anError = true;
         };


         if(!anError && (planRepairValidator->getErrorLog().getConditions().size() == 0) && (planRepairValidator->checkGoal(theGoal)))
         {
             planRepaired = true; searching = false;
         };

        deleteTestPlan(testPlan2); testPlan->clear(); delete testPlan;
        delete planRepairValidator;
        if(counter >= 8191 || ((shiftTime - lastShiftTime) < v.getTolerance() && (shiftTime - lastShiftTime) > 0 ) ) searching = false;
        lastShiftTime = shiftTime;
   };


   if(!planRepaired) moveActionTimes(repairingPlan,firstAction,originalStartTime,0);

   return planRepaired;
};

//return a random number with uniform prob over 0 to 1
/*double getRandomNumber()
{
     double randomNumber = double(rand()) / double(RAND_MAX);

     return randomNumber;
};
*/
map<const plan_step *,const plan_step *> varyPlanTimestampsUniform(plan * aplan,const plan * p,double & variation,double deadLine)
{
  map<const plan_step *,const plan_step *> planStepMap;
  pc_list<plan_step *>::const_iterator origPlanStep = p->begin();

  for(pc_list<plan_step *>::iterator ps = aplan->begin(); ps != aplan->end(); ++ps, ++origPlanStep)
  {

      (*ps)->start_time = (*ps)->start_time + (1 - 2*getRandomNumberUniform())*variation;
      if((*ps)->start_time < 0) (*ps)->start_time = 0;
      if(!(*ps)->duration_given && (*ps)->start_time > deadLine) (*ps)->start_time = deadLine;
      else if((*ps)->duration_given && (((*ps)->start_time + (*ps)->duration) > deadLine)) (*ps)->start_time = deadLine - (*ps)->duration;

     /* if((*ps)->duration_given)
      {
          //(*ps)->originalDuration = (*ps)->duration;

          (*ps)->duration = (*ps)->duration + (1 - 2*getRandomNumber())*variation;
          if((*ps)->duration < 0) (*ps)->duration = 0;
      }; */
          //cout << (*ps)->start_time << " ";
      planStepMap[*ps] = *origPlanStep;
  };
     // cout << "\n";
  return planStepMap;
};

//randomly vary the timestamps of actions to try and find a valid plan
bool PlanRepair::shakePlan(const plan * repairingPlan,const plan_step * firstAction,double variation)
{
   srand(time(0)); // Initialize random number generator.
   bool searching = true;
  // double shiftTime;
   int counter = 1;
   bool planRepaired = false;

   Validator * planRepairValidator = 0;
   bool anError;


   while(searching)
   {
        map<const plan_step *,const plan_step *> planStepMap;
        plan * testPlan = newTestPlan(repairingPlan);
        planStepMap = varyPlanTimestampsUniform(testPlan,repairingPlan,variation,deadLine);
        plan * testPlan2 = new plan(*testPlan);

        //add timed initial literals to the plan from the problem spec, these times are fixed
        for(vector<plan_step *>::iterator ps = timedInitialLiteralActions.begin(); ps != timedInitialLiteralActions.end(); ++ps)
        {
           testPlan->push_back(*ps);
        };


        planRepairValidator = new Validator(v.getDerivRules(),v.getTolerance(),typeC,operators,initialState,testPlan,
                          metric,stepLength,durative,current_analysis->the_domain->constraints,current_analysis->the_problem->constraints);

         anError = false;
         try{
           planRepairValidator->execute();
         }
         catch(exception & e)
         {
             cout << e.what() << "\n"; anError = true;
         };


         if(!anError && (planRepairValidator->getErrorLog().getConditions().size() == 0) && (planRepairValidator->checkGoal(theGoal)))
         {
             planRepaired = true; searching = false;

             for(map<const plan_step *,const plan_step *>::iterator i = planStepMap.begin(); i != planStepMap.end(); ++i)
             {
                 const_cast<plan_step *>(i->second)->start_time = i->first->start_time;
             };

         };

        deleteTestPlan(testPlan2); testPlan->clear(); delete testPlan;
        delete planRepairValidator;
        counter++;
        if(counter >= 32768) searching = false;
   };



   return planRepaired;
};

//change so an action is attempted more than once, until no more can be done.
bool PlanRepair::repairPlanOneActionAtATime(const plan * repairingPlan,const plan_step * firstAction)
{

  bool continueRepairing = true;
  pair<const plan_step *,pair<bool,bool> > fixOneAction;
  const plan_step * actionToRepair = firstAction;
  const plan_step * actionNotFixed = firstAction;
  //action that is not fixed after an action which is fixed; only try to repair an action twice if some action has been fixed in the meantime.

  while(continueRepairing)
  {

     fixOneAction = repairPlanOneAction(repairingPlan,actionToRepair);

     if(fixOneAction.first != 0)
     {
       if(fixOneAction.second.first) actionNotFixed = fixOneAction.first;

       actionToRepair = fixOneAction.first;
     }
     else
     {
       continueRepairing = false;
     };

     if(!fixOneAction.second.first && (actionNotFixed == actionToRepair)) continueRepairing = false;


  };

  //return if the plan is fixed or not
  return fixOneAction.second.second;
};

map<const plan_step *,const plan_step *> createPlanStepMap(const plan * plan1,const plan * plan2)
{
 map<const plan_step *,const plan_step *> planStepMap;

 pc_list<plan_step*>::const_iterator j = plan2->begin();
 for(pc_list<plan_step*>::const_iterator i = plan1->begin(); i != plan1->end(); ++i, ++j)

 {
    planStepMap[*i] = *j;
 };

 return planStepMap;
};

bool PlanRepair::isInTimeInitialLiteralList(const plan_step * ps)
{
  for(vector<plan_step *>::const_iterator i = timedInitialLiteralActions.begin(); i != timedInitialLiteralActions.end(); ++i)
  {
     if(ps == *i) return true;
  };

  return false;
};

set<const Action *> PlanRepair::getUniqueFlawedActions(Validator * vld)
{
   vector<const UnsatCondition *> unsatPlanRepairConditions = vld->getErrorLog().getConditions();
   set<const Action *> uniqueActions;


   for(vector<const UnsatCondition *>::const_iterator j = unsatPlanRepairConditions.begin(); j != unsatPlanRepairConditions.end(); ++j)
   {
       const Action * anAction = 0;

       if(const UnsatInvariant * usi = dynamic_cast<const UnsatInvariant*>(*j)) anAction = usi->action;
       else if(const UnsatPrecondition * usp = dynamic_cast<const UnsatPrecondition*>(*j)) anAction = usp->action;

       if(anAction != 0 && !isInTimeInitialLiteralList(anAction->getPlanStep())) uniqueActions.insert(anAction);
   };

  return uniqueActions;
};

//function reports back: (next flawed action, is the action fixed, is the whole plan fixed)
pair<const plan_step *,pair<bool,bool> > PlanRepair::repairPlanOneAction(const plan * repairingPlan,const plan_step * firstAction)
{

 const plan_step * nextFlawedAction = 0;
 bool planRepaired = false; //, goalSatisfied = false;
 bool actionFixed = false;
 string actionName = getName(firstAction);
 double actionTime = firstAction->start_time;
 bool con = ContinueAnyway;
 map<const plan_step *,const plan_step *> planStepMap;

 Validator * planRepairValidator = 0;

 ContinueAnyway = true;

 pair<double,double> slideLimits = getSlideLimits(repairingPlan,actionName,actionTime,deadLine);
 //cout << "Slide limits "<< actionName <<" = "<< slideLimits.first <<" , "<< slideLimits.second <<"\n";


  int counter = 1;
  double bNumber = aNumber(counter);
  double oldStartTime = actionTime;

  while(true)
  {

            actionTime = (slideLimits.second - slideLimits.first)*bNumber + slideLimits.first;
            changeActionTime(repairingPlan,actionName,oldStartTime,actionTime);
            oldStartTime = actionTime;

            plan * testPlan = newTestPlan(repairingPlan);
            planStepMap = createPlanStepMap(testPlan,repairingPlan);
            plan * testPlan2 = new plan(*testPlan); //we can delete the test plan now without deleting the timed initial literal actions

             //add timed initial literals to the plan from the problem spec, these time are fixed
             for(vector<plan_step *>::iterator ps = timedInitialLiteralActions.begin(); ps != timedInitialLiteralActions.end(); ++ps)
             {
               testPlan->push_back(*ps);
             };


            planRepairValidator = new Validator(v.getDerivRules(),v.getTolerance(),typeC,operators,initialState,testPlan,
                metric,stepLength,durative,current_analysis->the_domain->constraints,current_analysis->the_problem->constraints);


            try{
              planRepairValidator->execute();
            }
            catch(exception & e)
            {
                cout << e.what() << "\n";
            };


            if(planRepairValidator->getErrorLog().getConditions().size() == 0)
	      {  if(planRepairValidator->checkGoal(theGoal)) //goalSatisfied = true;
                    //cout << "Satisfied "<< actionName << " at time "<<actionTime<<"\n";
               actionFixed = true; planRepaired = true; break;
            };


            //get unique set of actions that are flawed
            set<const Action *> flawedActions = getUniqueFlawedActions(planRepairValidator);


            actionFixed = true;
            set<const Action *>::const_iterator j =  flawedActions.begin();
            for( ; j != flawedActions.end();++j)
            {
                   //still an error with the action
                   if(actionName == (*j)->getName0() && (*j)->getPlanStep()->start_time == actionTime)
                   {
                     actionFixed = false;
                     break;
                   };
              };

              //if(actionFixed) cout << "Satisfied "<< actionName << " at time "<<actionTime<<"\n";
              bNumber = aNumber(++counter);

              if(actionFixed || (counter >= 1023) || ((slideLimits.second - slideLimits.first)*bNumber < v.getTolerance()))
              {
                //return the next action to be fixed
                nextFlawedAction = 0;
                if(!actionFixed && !flawedActions.empty())
                {
                  ++j;

                  if(j != flawedActions.end())
                  {
                     nextFlawedAction = planStepMap[(*j)->getPlanStep()];
                  };
                }
                else if(!planRepaired)
                {
                  set<const Action *>::const_iterator k = flawedActions.begin();
                  if(k != flawedActions.end())
                  {
                    nextFlawedAction = planStepMap[(*k)->getPlanStep()];
                  };
                };

                deleteTestPlan(testPlan2); testPlan->clear(); delete testPlan;

                delete planRepairValidator;
                break;
              };

              deleteTestPlan(testPlan2); testPlan->clear(); delete testPlan;
              delete planRepairValidator;
  }; //loop thro' different points end, sliding action (while loop)


  ContinueAnyway = con;
  return make_pair(nextFlawedAction,make_pair(actionFixed,planRepaired));
};

void Validator::reportViolations() const
{
	if(violations.empty()) return;
	if(LaTeX) *report << "\\\\\n";
	*report << "Violations:\n";
	if(LaTeX) *report << "\\\\\n\\begin{tabular}{lc}\n";

	for(map<string,int>::const_iterator i = violations.begin();i != violations.end();++i)
	{
		if(i->second == 0) continue;
		if(LaTeX)
		{
			*report << i->first << " & " << i->second << "\\\\\n";
		}
		else
		{
			*report << "\t" << i->first << ": " << i->second << "\n";
		};
	};

	if(LaTeX) *report << "\\end{tabular}\n";
};

vector<plan_step *> getTimedInitialLiteralActions(analysis * current_analysis)
{

  vector<plan_step *> timedInitialLiteralActions;

    if(current_analysis->the_problem->initial_state->timed_effects.size() != 0)
      {
          int count = 1;
           for(pc_list<timed_effect*>::const_iterator e = current_analysis->the_problem->initial_state->timed_effects.begin(); e != current_analysis->the_problem->initial_state->timed_effects.end(); ++e)
           {
                  operator_symbol * timed_initial_lit = current_analysis->op_tab.symbol_put("Timed Initial Literal Action "+ toString(count++));

                  action  * timed_initial_lit_action = new action(timed_initial_lit,new var_symbol_list(),new conj_goal(new goal_list()),(*e)->effs,new var_symbol_table());

                  plan_step * a_plan_step =  new plan_step(timed_initial_lit,new const_symbol_list());
                  a_plan_step->start_time_given = true;
                  a_plan_step->start_time = dynamic_cast<const timed_initial_literal *>(*e)->time_stamp;

                  a_plan_step->duration_given = false;

                  timedInitialLiteralActions.push_back(a_plan_step);
                  current_analysis->the_domain->ops->push_back(timed_initial_lit_action);
           };
      };

  return timedInitialLiteralActions;
};

plan * addTimedInitialActions(plan * aPlan, vector<plan_step *> & timedInitialLiteralActions)
{
  //add timed initial literals to the plan from the problem spec
  for(vector<plan_step *>::iterator ps = timedInitialLiteralActions.begin(); ps != timedInitialLiteralActions.end(); ++ps)
  {
     aPlan->push_back(*ps);
  };

  return aPlan;
};

bool isLockedAction(plan_step * ps, set<plan_step *> & lockedActions)
{
  set<plan_step *>::const_iterator i = lockedActions.find(ps);
  return (i != lockedActions.end());
};

void PlanRepair::setPlanAndTimedInitLits(const plan * aPlan, set<plan_step *> lockedActions)
{
   plan * planWithoutTimedInits = new plan();

   for(pc_list<plan_step*>::const_iterator i = aPlan->begin(); i != aPlan->end(); ++i)
   {
      if(isLockedAction(*i,lockedActions)) timedInitialLiteralActions.push_back(*i);
      else planWithoutTimedInits->push_back(*i);
   };

  p = planWithoutTimedInits;
};


/*
void PlanRepair::repairPlanBeagle()
// Note that repairPlan and repairPlanBeagle have been swapped in this file
// compared with earlier versions (VAL-3.2 in particular)
{
 bool latex = LaTeX;
 bool verbose = Verbose;
 LaTeX = false; Verbose = false;
 ErrorReport = true;
 ContinueAnyway = true;

 if(!latex) cout << "\nRepairing plan...\n";
 const plan * repairingPlan = new plan(*p);
 bool planRepaired = false, goalSatisfied = false;
 string actionName; double actionTime;
 bool continueRepairingTemp = false, continueRepairing = true;
 Validator * toBeDeletedValidator = 0;
 Validator * continueValidator = 0;

 //if a duractive action invariant error log is split into two then only consider one of them, as only interested in action
 vector<const UnsatCondition *> unsatConditionsTemp = getUnSatConditions();
 vector<const UnsatCondition *> unsatConditions;
 bool addToConds;
 for(vector<const UnsatCondition *>::const_iterator r = unsatConditionsTemp.begin(); r != unsatConditionsTemp.end(); ++r)
 {
     addToConds = true;
     for(vector<const UnsatCondition *>::const_iterator s = unsatConditions.begin(); s != unsatConditions.end(); ++s)
     {
         if(const UnsatInvariant * usi = dynamic_cast<const UnsatInvariant*>(*r))
        {
             if(const UnsatInvariant * usi2 = dynamic_cast<const UnsatInvariant*>(*s))
             {
                   if(usi->action == usi2->action) addToConds = false;

             };
        };
     };

    if(addToConds) unsatConditions.push_back(*r);
    //else unsatConditions.erase(find(unsatConditions.begin(),unsatConditions.end(),*r));
 };
 unsatConditionsTemp.clear();

 vector<const UnsatCondition *> satConditions;


 while(continueRepairing){
 continueRepairingTemp = false;
 Validator * planRepairValidator = 0; continueValidator = 0;
 //unsatConditions = unsatConditionsTemp;
 //cout << unsatConditions.size() << "\n";
 //loop thro' unsatisfied conditions looking for unsatisfied invariant conditions that can be 'slided around' and try and repair plan
 for(vector<const UnsatCondition *>::const_iterator i = unsatConditions.begin(); i != unsatConditions.end(); )
 {
        if(const UnsatInvariant * usi = dynamic_cast<const UnsatInvariant*>(*i))
        {
           actionName = usi->action->getName0();
           pair<double,double> slideLimits = usi->getSlideLimits(repairingPlan,v.getTolerance());//times the start point can be slid without it or endpoint of durative action crossing another endpoint
            //cout << "Slide limits "<< actionName <<" = "<< slideLimits.first <<" , "<< slideLimits.second <<"\n";

           if(slideLimits.first != usi->startTime || slideLimits.second != usi->startTime)
           {
            //search for a new place for the durative action, use the repaired plan for the next unsatisfied condition

            int startingPoint = 1;
            int limit =  int((slideLimits.second - slideLimits.first) / (usi->endTime - usi->startTime) + 1.5 + 2.0);
            //cout << limit <<" =limit\n";
            double bNumber = aNumber(startingPoint++);
            //try middle point first...   loop thro' different points start
            while(true)
            {
                      actionTime = (slideLimits.second - slideLimits.first)*bNumber + slideLimits.first;
                      changeActionTime(repairingPlan,actionName, actionTime);

                      //Validator planRepairValidator(v.getDerivRules(),v.getTolerance(),typeC,operators,initialState,repairingPlan,
                       //   metric,stepLength,durative);
                      planRepairValidator = new Validator(v.getDerivRules(),v.getTolerance(),typeC,operators,initialState,repairingPlan,
                          metric,stepLength,durative);
					//cout << "New at " << planRepairValidator << "\n";
                      //cout << "Executing another plan...\n"; v.displayInitPlan(repairingPlan);
                      try{
                        planRepairValidator->execute();
                      }
                      catch(exception & e)
                      {
                          cout << e.what() << "\n";
                      };


                      if(planRepairValidator->getErrorLog().getConditions().size() == 0)
                           {  if(planRepairValidator->checkGoal(theGoal)) goalSatisfied = true;
                              //cout << "Satisfied "<< actionName << " at time "<<actionTime<<"\n";
                              planRepaired = true; break;
                            };


                      //find corresponding error log entry in planRepairValidator
                      vector<const UnsatCondition *> unsatPlanRepairConditions = planRepairValidator->getErrorLog().getConditions();
                      //continueRepairingTemp = false;
                      for(vector<const UnsatCondition *>::const_iterator j = unsatPlanRepairConditions.begin(); j != unsatPlanRepairConditions.end(); )
                      {
                             if(const UnsatInvariant * usi2 = dynamic_cast<const UnsatInvariant*>(*j))
                             {
                                  //since action is only sliding about between other actions the durative action is not split into > 1 errorlog entries
                                  if(actionName == usi2->action->getName0()) break;
                             };

                             ++j;

                             if(j == unsatPlanRepairConditions.end())
                             {
                                //cout << "Satisfied2 "<< actionName << " at time "<<actionTime<<"\n";
                                continueRepairingTemp = true; //ie this condition is repaired so may be possible to repair other errors
                                satConditions.push_back(*i);

                                  if(continueValidator != 0)
                                  {
                                  	//cout << "Deleting " << toBeDeletedValidator << "\n";
                                     delete toBeDeletedValidator;
                                     toBeDeletedValidator = 0;
                                      //cout << "delete con "<< continueValidator << "\n";
                                   };
                                 continueValidator = planRepairValidator;
                             };
                        };
                 bNumber = aNumber(startingPoint++);
                 if(((slideLimits.second - slideLimits.first)*bNumber < v.getTolerance()) || continueRepairingTemp || startingPoint > limit) break;
                 else {
                 	//cout << "Deleted here " << planRepairValidator <<"\n";
                 	delete planRepairValidator;
                 	planRepairValidator = 0;
                 };
            }; //loop thro' different points end, sliding action (while loop)


           };//end of sliding an action (if an unsatisfied invariant is slidable)


        }; //end of unsatisfied invariaint handling (if an unsatisfied invariant)


        //fix other actions and goal now?





   continueRepairing = false;
   ++i;
   if(continueRepairingTemp && i == unsatConditions.end())
   {
     continueRepairing = true;

     //toBeDeletedValidator = planRepairValidator;
   }
   else if(planRepairValidator != 0 && continueValidator != planRepairValidator)
   {
//   cout << "delete2 "<< planRepairValidator << "\n";
     delete planRepairValidator;
     planRepairValidator = 0;
     //cout << "delete2 "<< planRepairValidator << "\n";
   };


 };//end of loop for repairing unsatconditions


//delete Validator used for recursive loop for repairing
if(toBeDeletedValidator != 0)
{
   delete toBeDeletedValidator;
   // cout << "delete3 "<< toBeDeletedValidator << "\n";
};
toBeDeletedValidator = 0;

if(planRepaired) break;

 if(continueRepairing)
 {
           unsatConditions.clear();
           vector<const UnsatCondition *> unsatConditionsTemp = continueValidator->getErrorLog().getConditions();
           for(vector<const UnsatCondition *>::const_iterator r = unsatConditionsTemp.begin(); r != unsatConditionsTemp.end(); ++r)
           {
               addToConds = true;
               for(vector<const UnsatCondition *>::const_iterator s = unsatConditions.begin(); s != unsatConditions.end(); ++s)
               {
                   if(const UnsatInvariant * usi = dynamic_cast<const UnsatInvariant*>(*r))
                  {
                       if(const UnsatInvariant * usi2 = dynamic_cast<const UnsatInvariant*>(*s))
                       {
                             if(usi->action == usi2->action) addToConds = false;
                       };
                  };
               };

              if(addToConds) unsatConditions.push_back(*r);
           };
           unsatConditionsTemp.clear();

toBeDeletedValidator = continueValidator;
};


};//end of recursive loop for repairing




 LaTeX = latex; Verbose = verbose;

 if(planRepaired && goalSatisfied)
 {
    if(LaTeX) *report << "\\subsubsection{Similar Valid Plan}\n";
    else cout << "\nA valid similar plan to the plan given is:\n";

    if(LaTeX) v.displayInitPlanLaTeX(repairingPlan);
    else v.displayInitPlan(repairingPlan);
 }
 else
 {
     if(LaTeX) *report << "\\subsubsection{Similar Valid Plan}\n Failed to find a valid similar plan {\\begin{rotate}{270}:-( \\end{rotate}}";
     else cout << "\nFailed to find a valid similar plan :-(\n";

 };


};
*/


};

