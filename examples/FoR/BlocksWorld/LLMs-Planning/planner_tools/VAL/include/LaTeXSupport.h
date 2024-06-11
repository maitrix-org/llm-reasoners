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

  $Date: 2009-02-05 10:50:20 $
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

#ifndef __LATEXSUPPORT
#define __LATEXSUPPORT

#include <vector>
#include <string>
#include <iostream>
#include "Utils.h"
#include "main.h"
#include "Validator.h"


using std::ostream;
using std::string;
using std::vector;

namespace VAL {
  
struct showList {
	void operator()(const pair<double,vector<pair<string,vector<double> > > > & ps) const
	{
		if(LaTeX)
		{		
			string s;
			*report << ps.first<<" \\>";
			for(vector<pair<string,vector<double> > >::const_iterator i = ps.second.begin(); i != ps.second.end() ; ++i)
			{
				s = i->first; 
				replaceSubStrings(s,"/","/\\-");
            	latexString(s);
				*report << "\\begin{minipage}[t]{12cm} " << s << " ";
				for(vector<double>::const_iterator j = i->second.begin();
					j != i->second.end();++j)
				{
					*report << *j << " ";
				};
				
				*report << " \\end{minipage}\\\\\n \\>";
			};
			*report << "\\\\\n";
		}
		else
		{
			cout << "\nValue: " << ps.first << "\n ";
			for(vector<pair<string,vector<double> > >::const_iterator i = ps.second.begin();
					i != ps.second.end();++i)
			{
				cout << i->first << " ";
				copy(i->second.begin(),i->second.end(),ostream_iterator<double>(cout," "));
				cout << "\n";
			};
		};
	};
};

void displayFailedLaTeXList(vector<string> & vs);

class LaTeXSupport {
private:
	int NoGraphPoints;
	int noPoints;
	int noGCPages;
	int noGCPageRows;
	vector<string> ganttObjectsAndTypes;
	vector<string> ganttObjects;
public:
	LaTeXSupport() : NoGraphPoints(500), noGCPages(0), noGCPageRows(0) {};
	void LaTeXHeader();
	void LaTeXPlanReportPrepare(char *);
	void LaTeXPlanReport(Validator * v,plan *);
	void LaTeXEnd();
	void LaTeXGantt(Validator * v);
	void LaTeXGraphs(Validator * v);
	void LaTeXDomainAndProblem();
	void LaTeXBuildGraph(ActiveCtsEffects * ace,const State * s);
	void setnoGCPages(int g) {noGCPages = g;};
	void setnoGCPageRows(int g) {noGCPageRows = g;};
	void setnoPoints(int n)
	{
		noPoints = n;
		if(noPoints < 10) noPoints = 10;
		else if(noPoints > 878) noPoints = 878;
	    NoGraphPoints = noPoints;
	};
	void addGanttObject(char * c)
	{
		ganttObjectsAndTypes.push_back(c);
	};
};

extern LaTeXSupport latex;

};



#endif
