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

#include "InstPropLinker.h"

#include "Proposition.h"
#include "instantiation.h"

#include "ptree.h"
#include "FastEnvironment.h"
#include "Environment.h"

using namespace VAL;


namespace Inst {

Literal * toLiteral(const VAL::SimpleProposition * sp)
{
	int id = -1;
	for(parameter_symbol_list::const_iterator i = sp->getProp()->args->begin();
			i != sp->getProp()->args->end();++i)
	{
		if(const var_symbol * vs = dynamic_cast<const var_symbol *>(*i))
		{
			id = std::max(id,static_cast<const IDsymbol<var_symbol>*>(vs)->getId());
		};
	};
	FastEnvironment * fe = new FastEnvironment(id+1);
	for(parameter_symbol_list::const_iterator i = sp->getProp()->args->begin();
			i != sp->getProp()->args->end();++i)
	{
		if(const var_symbol * vs = dynamic_cast<const var_symbol *>(*i))
		{
			(*fe)[vs] = const_cast<VAL::const_symbol *>(sp->getEnv()->find(vs)->second);
		};
	};
	
	CreatedLiteral * cl = new CreatedLiteral(sp->getProp(),fe);

	Literal * res = instantiatedOp::getLiteral(cl);
	if(res != cl)
	{
		delete cl;
	};
	return res;
};

Environment toEnv(instantiatedOp * op)
{
	Environment e;
	for(var_symbol_list::const_iterator i = op->forOp()->parameters->begin();
			i != op->forOp()->parameters->end();++i)
	{
		if(const var_symbol * vs = dynamic_cast<const var_symbol *>(*i))
		{
			e[vs] = (*(op->getEnv()))[vs];
		};
	};
	return e;
};
	
};
