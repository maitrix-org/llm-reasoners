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

/* ----------------------------------------------------------------------------
   Macros for diagnostic output.
   These generate code to output both name and value of a field,
   and to deal with pretty-printing indentation in a consistent way.

   Edit this file to change the output format of the syntax trees.

   $Date: 2009-02-05 10:50:26 $
   $Revision: 1.2 $

   stephen.cresswell@cis.strath.ac.uk
  
   Strathclyde Planning Group
   --------------------------------------------------------------------------*/

// #  expands arg into quoted string
// ## concatenates arg

// Output NAME - used for name of class
#define TITLE(NAME) indent(ind); cout << '(' << #NAME << ')';

// Display a data member that is a parse_category
#define FIELD(NAME) indent(ind); cout << #NAME << ": "; if (NAME != NULL) NAME->display(ind+1); else cout << "(NULL)";  

// Used for display of list element
#define ELT(NAME) { if ((NAME) != NULL) (NAME)->display(ind+1); else cout << "(NULL)"; }

// Display NAME only
#define LABEL(NAME) indent(ind); cout << #NAME << ':'; 

// Output a data member that is not a parse_category
#define LEAF(NAME) indent(ind); cout << #NAME << ": "; cout << NAME;


extern void indent(int ind);


