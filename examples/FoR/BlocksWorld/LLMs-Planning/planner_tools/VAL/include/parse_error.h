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
  Parse error representation

  $Date: 2009-02-05 10:50:26 $
  $Revision: 1.2 $

  stephen.cresswell@cis.strath.ac.uk
  July 2001.

  Strathclyde Planning Group
 ----------------------------------------------------------------------------
*/


#ifndef PARSE_ERROR_H
#define PARSE_ERROR_H

#include<string>
#include<cstdio>
#include <iostream>

#include "ptree.h" 

using std::list;
using std::string;
using std::cout;

extern int line_no;              // Line number global
extern char* current_filename;   // file global

namespace VAL {

enum error_severity {E_WARNING,E_FATAL};


class parse_error
{
private:
    error_severity severity;
    char* filename;
    int line;
    string description;
    
public:

    parse_error(error_severity s, const string& d) :
	severity(s),
	line(line_no),
	description(d)
	{
	    filename= current_filename;
	};

    // describe error
    void report()
	{
	    cout << filename
		 << ": line: " 
		 << line
		 << ": ";

	    if (severity==E_FATAL)
		cout << "Error: ";
	    else
		cout << "Warning: ";

	    cout << description 
		 << '\n';
	};
};


// It seems to be more sensible to keep errors and warnings together in the
// same list, as we want to output them in the same order that they were
// found.

class parse_error_list : public list<parse_error*>
{
public:
    int errors;
    int warnings;

    parse_error_list() : errors(0), warnings(0) {};

    ~parse_error_list()
	{
	    for (iterator i=begin(); i!=end(); ++i)
		delete (*i);
	};


    // parse_error_list is reponsible for creating and 
    // destroying parse_error objects,
    void add(error_severity sev, const string& description)
	{
	    //	Use yacc globals to retrieve file and line number;
	    push_back(new parse_error(sev,description));

	    if (sev==E_WARNING)
		++warnings;
	    else
		++errors;
	};

    void report()
	{
	    cout << "\nErrors: " << errors 
		 << ", warnings: " << warnings << '\n';

	    for (iterator i=begin(); i!=end(); ++i)
		(*i)->report();
	};
};

};

#endif /* PARSE_ERROR_H */
