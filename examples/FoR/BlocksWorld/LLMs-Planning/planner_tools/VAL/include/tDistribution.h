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

  $Date: 2009-02-05 10:50:27 $
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
#ifndef __TDIST
#define __TDIST

namespace VAL {
  
//struct store the upper critial values of the student T Distribution
struct Table5Percent
{
    map<int,double> values;

    Table5Percent()
    {
       values[1] = 12.7061503;
       values[2] = 4.302655725;
       values[3] = 3.182449291;
       values[4] = 2.776450856;
       values[5] = 2.570577635;
       values[6] = 2.446913641;
       values[7] = 2.36462256;
       values[8] = 2.306005626;
       values[9] = 2.262158887;
       values[10] = 2.228139238;
       values[11] = 2.200986273;
       values[12] = 2.178812792;
       values[13] = 2.16036824;
       values[14] = 2.144788596;
       values[15] = 2.131450856;
       values[16] = 2.119904821;
       values[17] = 2.109818524;
       values[18] = 2.100923666;
       values[19] = 2.093024705;
       values[20] = 2.085962478;
       values[21] = 2.079614205;
       values[22] = 2.073875294;
       values[23] = 2.068654794;
       values[24] = 2.063898137;
       values[25] = 2.05953711;
       values[26] = 2.055530786;
       values[27] = 2.051829142;
       values[28] = 2.048409442;
       values[29] = 2.045230758;
       values[30] = 2.042270353;
       values[31] = 2.039514584;
       values[32] = 2.036931619;
       values[33] = 2.03451691;
       values[34] = 2.032243174;
       values[35] = 2.030110409;
       values[36] = 2.02809133;
       values[37] = 2.026190487;
       values[38] = 2.024394234;
       values[39] = 2.022688932;
       values[40] = 2.021074579;
       values[50] = 2.008559932;
       values[60] = 2.000297172;
       values[70] = 1.994435479;
       values[80] = 1.990065357;
       values[90] = 1.986672942;
       values[100] = 1.983971742;
       values[110] = 1.981766218;
       values[120] = 1.979929038;
       values[130] = 1.97837835;
       values[140] = 1.977055035;
       values[150] = 1.975904524;
       values[160] = 1.97490408;
       values[170] = 1.974017323;
       values[180] = 1.97323061;
       values[190] = 1.972530299;
       values[200] = 1.971893653;
       values[220] = 1.970806807;
       values[240] = 1.969897312;
       values[260] = 1.969128789;
       values[280] = 1.968473953;
       values[300] = 1.967900971;
       values[340] = 1.966964192;
       values[360] = 1.966573109;
       values[400] = 1.965913725;
       values[450] = 1.965249794;
       values[500] = 1.96471774;
       values[600] = 1.963926479;
       values[700] = 1.963358045;
       values[800] = 1.96293513;
       values[900] = 1.962603164;
       values[1000] = 1.962339411;
       values[1500] = 1.961548151;
       values[2000] = 1.96115252;
       values[3000] = 1.96075689;
       values[4000] = 1.960556801;
       values[5000] = 1.960438567;
       values[6000] = 1.96036126;
       values[7000] = 1.960302143;
       values[8000] = 1.960261216;
       values[9000] = 1.960229383;
       values[10000] = 1.960202098;
       values[20000] = 1.960083864;
       values[30000] = 1.960042937;
       values[50000] = 1.960011105;
       values[100000] = 1.959988367;
       values[200000] = 1.959974725;
       values[1000000] = 1.95996563;                                       
    };
    
  
};


double upperCritialValueTDistribution(double alpha,int degreesFreedom)
{    
   if(alpha == 0.05)
   {
     Table5Percent table5Percent;
     for(map<int,double>::reverse_iterator i = table5Percent.values.rbegin(); i != table5Percent.values.rend();++i)
     {
        if(degreesFreedom >= i->first) return i->second;             
     };    
   }
   else
   {

     BadAccessError bae;
		 throw bae;
   };

  return -1;
};

};

#endif
