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

  $Date: 2009-02-05 10:50:21 $
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
//#undef vector

#include <math.h>
#include <vector>
#include <algorithm>
#include "Polynomial.h"
#include "Exceptions.h"
#include "Proposition.h"
#include "main.h"

using std::cout;

//#define vector std::vector
//#define map std::map

namespace VAL {

ostream & operator <<(ostream & o,const Intervals & i)
{
	i.write(o);
	return o;
};

bool Intervals::operator==(const Intervals & ints) const
{
	return intervals == ints.intervals;

};

Intervals::Intervals(vector<pair<intervalEnd,intervalEnd> > ints) : intervals(ints)
{
	Intervals int0;

	for(vector< pair<intervalEnd,intervalEnd> >::iterator i = intervals.begin(); i != intervals.end();++i)
	{
		if(((i->first.first > i->second.first)	|| ( (i->first.first == i->second.first) && (!(i->first.second) || !(i->second.second)) ) )
			 )
		{
			*report << "The collection of intervals "<< *this << " is invalid\n";
			InvalidIntervalsError iie;
			throw iie;
		};

	};

	//Note: no check that intervals do not overlap

};

//dummy intervals for use in derived predicate method getIntervals
Intervals::Intervals(bool b)
{
  intervals.push_back(make_pair(make_pair(-2,false),make_pair(-1,false)));
};

Intervals::~Intervals()
{
	intervals.clear();
};

void Intervals::write(ostream & o) const
{
	if(intervals.size()==0)
	{
		o << "the empty set";
	}
	else
	{    if(LaTeX) o << "$";
		for(vector< pair<intervalEnd,intervalEnd> >::const_iterator i = intervals.begin(); i != intervals.end();)
		{

			if(i->first.second) o << "[ "; else o << "( ";

			o << i->first.first << " , " << i->second.first;

			if(i->second.second) o << " ]"; else o << " )";

			if(++i != intervals.end()) {if(LaTeX) o << "\\cup"; else o << " U ";};

		};
         if(LaTeX) o << "$";

	};

	//o << "\n";
};

void Intervals::writeOffset(double t) const
{
	if(intervals.size()==0)

	{
		*report << "the empty set";
	}
	else
	{
      if(LaTeX) *report << "$";
		for(vector< pair<intervalEnd,intervalEnd> >::const_iterator i = intervals.begin(); i != intervals.end();)
		{

			if(i->first.second) *report << "[ "; else *report << "( ";

			*report << t + i->first.first << " , " << t + i->second.first;

			if(i->second.second) *report << " ]"; else cout << " )";


			if(++i != intervals.end()) {if(LaTeX) *report << "\\cup"; else cout << " U ";};

		};
       if(LaTeX) *report << "$";
	};




};

ostream & operator << (ostream & o,const CtsFunction & cf)
{

	cf.write(o);

	return o;
};

//returns 1 / num!
CoScalar recipfact(unsigned int num)
{
	CoScalar ans = 1;

	for(CoScalar i = 1; i <= num; ++i)
	{
		ans *= 1 / i;
	};

	return ans;
};


unsigned int Exponential::getNoTerms(CoScalar endInt) const
{
	unsigned int N = 0;
   unsigned int N1 = 0;
	CoScalar aConstant;
	CoScalar a = 0;
	CoScalar rem;

    //find the maximum value of the polynomial
    vector<CoScalar> roots = poly->getRoots(endInt);
    a = poly->evaluate(0);
    CoScalar aValue = poly->evaluate(endInt);
    if(aValue > a) a = aValue;

    for(vector<CoScalar>::const_iterator i = roots.begin(); i != roots.end(); ++i)
    {
          aValue = *i;
          if(aValue > a) a = aValue;
    };
      //cout << " \\\\\nMax ="<<a<<"\\\\\n";

	if( a >= 0)
  {
               		aConstant = fabs(K*exp(a));
               	rem = aConstant;

               	while(rem >= Polynomial::tooSmall)
               	{
               		N++;
               		rem = aConstant * fabs(pow(a,N+1)) * recipfact(N);
               	};
   };
         //cout << " \\\\\nN = "<<N<<"\\\\\n";
    //find the minimum value of the polynomial
    a = poly->evaluate(0);
    aValue = poly->evaluate(endInt);
    if(aValue < a) a = aValue;

    for(vector<CoScalar>::const_iterator i = roots.begin(); i != roots.end(); ++i)
    {
          aValue = *i;
          if(aValue < a) a = aValue;
    };
       //cout << " \\\\\nMin = "<<a<<"\\\\\n";
	if( a < 0)
   {
                     aConstant = fabs(K*exp(a));
                     	rem = aConstant;

                     	while(rem >= Polynomial::tooSmall)
                     	{
                     		N1++;

                     		rem = aConstant * fabs(pow(a,N1+1)) * recipfact(N1);
                     	};
                      // cout << " \\\\\nN1 = "<<N1<<"\\\\\n";
                      if(N1 > N) N =N1;
   };

	//cout << "\\\\\\> No of terms for $"<<*this<<"$ = "<<N<<" aha!\\\\\n";
	return N;
};

Polynomial Exponential::getApproxPoly(CoScalar endInt) const
{
	Polynomial ans;

    if(poly->getDegree() == 1 && poly->getCoeff(0) == 0)
    {
         	ans.setCoeff(0,K);
            CoScalar a = poly->getCoeff(1);
         	unsigned int noTerms = getNoTerms(endInt);

         	for(unsigned int i = 1; i < noTerms ; ++i)
         	{

         		ans.setCoeff(i,K * recipfact(i) * pow(a,i));

         	};

         	if(c != 0)
         		ans.addToCoeff(0,c);

    }
    else if( poly->getDegree() == 0)
    {
            ans.setCoeff(0,K*exp(poly->getCoeff(0)));
         	if(c != 0)
         		ans.addToCoeff(0,c);

    }
    else
    {
            ans = *poly;
            ans.addToCoeff(0,1);

            Polynomial powPoly = *poly;
         	unsigned int noTerms = getNoTerms(endInt);

             if(noTerms > 150) //we are not prepared to wait to compute a lot of terms
             {
cout <<"Blown on terms...\n";
               ApproxPolyError ape;
               throw ape;
             };

         	for(unsigned int i = 2; i <= noTerms; ++i)
         	{
              powPoly = (powPoly*(*poly))/i; //powPoly = poly^i
              if(!powPoly.checkPolynomialCoeffs())
              {
          			ApproxPolyError ape;
          			throw ape;
          		};
              ans += powPoly;
              // cout << " \\\\\nApprox poly = "<<i<<":::$"<< ans <<"$\\\\\n";
         	};

            ans = K*ans;
         	if(c != 0)
         		ans.addToCoeff(0,c);

    };
		  // cout << " \\\\\nApprox poly = $"<< ans <<"$\\\\\n";
	return ans;
};

vector<CoScalar> Exponential::getRoots(CoScalar t) const
{
	return getApproxPoly(t).getRoots(t);
};

//check for infinity or NaN
void checkNum(CoScalar num)
{
	if( (2 * num == num && num != 0) || (!( num == num)) )
	{
		NumError ne;
		throw ne;
	};

};

bool Exponential::isLinear() const
{
	if(K == 0 || poly->getDegree() == 0) return true;
	return false;
};

bool Polynomial::isLinear() const
{
	if(getDegree() < 2) return true;
	return false;
};

pair<vector<pair<intervalEnd,intervalEnd> >,vector<CoScalar> > Exponential::isolateRoots(CoScalar t,CoScalar accuracy = 0) const
{
    return getApproxPoly(t).isolateRoots(t,accuracy);
};

CoScalar Exponential::evaluate(CoScalar t) const
{
	CoScalar ans = K * exp(poly->evaluate(t)) + c;
	checkNum(ans);
	return ans;
};

void Exponential::write(ostream & o) const
{
	if(K == 0)
		o << "0";
    o << K << "e^{"<<*poly <<"}";
	if(c-offSet != 0)
	{
		if(c-offSet > 0)
			o << " + "<<c-offSet;
		else
			o << " - "<<-c-offSet;
	};


};

Polynomial::~Polynomial()
{
	coeffs.clear();
};

CoScalar Polynomial::accuracy;
const CoScalar Polynomial::tooSmall = 1e-12;

CoScalar Polynomial::getCoeff(unsigned int pow) const
{
	map<unsigned int,CoScalar>::const_iterator i = coeffs.find(pow);

	if(i != coeffs.end())
		return i->second;
	else
		return 0;

};

void Polynomial::setCoeff(unsigned int pow,CoScalar value)
{
	CoScalar value0 = value;

	map<unsigned int,CoScalar>::iterator i = coeffs.find(pow);

	if(i != coeffs.end())
	{
		if(value0 != 0)
			i->second = value0;
		else
			coeffs.erase(i);
	}
	else
	{
    	if(value0 != 0) coeffs[pow] = value0;
    };


};

void Polynomial::addToCoeff(unsigned int pow,CoScalar value)
{

	map<unsigned int,CoScalar>::iterator i = coeffs.find(pow);

	if(i != coeffs.end())
	{
		i->second += value;
		if(i->second == 0) coeffs.erase(i);
	}
	else
	{
		if(value != 0) coeffs[pow] = value;
	};

};

unsigned int Polynomial::getDegree() const
{
	if(coeffs.size()==0) return 0;

	return (--coeffs.end())->first;
};

Polynomial& Polynomial::operator+=(const Polynomial & p)
{
	for(map<unsigned int,CoScalar>::const_iterator i = p.coeffs.begin(); i != p.coeffs.end();++i)

	{

		addToCoeff(i->first,i->second);
	};

	return *this;
};

Polynomial& Polynomial::operator+=(CoScalar num)
{
	addToCoeff(0,num);

	return *this;
};

Polynomial& Polynomial::operator-=(const Polynomial & p)
{
	for(map<unsigned int,CoScalar>::const_iterator i = p.coeffs.begin(); i != p.coeffs.end();++i)
	{
		addToCoeff(i->first,-(i->second));

	};

	return *this;
};

Polynomial& Polynomial::operator-=(CoScalar num)
{
	addToCoeff(0,-num);

	return *this;
};


Polynomial& Polynomial::operator*=(const Polynomial & p)
{
	Polynomial ans;

	for(map<unsigned int,CoScalar>::const_iterator i = p.coeffs.begin(); i != p.coeffs.end();++i)
	{
		for(map<unsigned int,CoScalar>::const_iterator j = coeffs.begin(); j != coeffs.end();++j)
		{
			ans.addToCoeff((i->first)+(j->first),(i->second)*(j->second));
		};
	};

	*this = ans;

	return *this;
};

Polynomial& Polynomial::operator*=(CoScalar num)
{
	for(map<unsigned int,CoScalar>::const_iterator i = coeffs.begin(); i != coeffs.end();++i)
	{
		setCoeff(i->first,(i->second)*num);
	};

	return *this;
};

Polynomial operator+(const Polynomial & p,const Polynomial & q)
{
	Polynomial ans = p;
	return ans += q;
};

Polynomial operator+(CoScalar num,const Polynomial & p)
{
	Polynomial ans = p;
	return ans += num;
};

Polynomial operator+(const Polynomial & p,CoScalar num)
{
	Polynomial ans = p;
	return ans += num;
};


Polynomial operator-(const Polynomial & p,const Polynomial & q)
{
	Polynomial ans = p;
	return ans -= q;
};

Polynomial operator-(CoScalar num,const Polynomial & p)
{
	Polynomial ans = p;
	return ans -= num;
};

Polynomial operator-(const Polynomial & p,CoScalar num)
{
	Polynomial ans = p;
	return ans -= num;
};

Polynomial operator*(const Polynomial & p,const Polynomial & q)
{
	Polynomial ans = p;
	return ans *= q;
};

Polynomial operator*(CoScalar num,const Polynomial & p)
{
	Polynomial ans = p;
	return ans *= num;
};

Polynomial operator*(const Polynomial & p,CoScalar num)
{
	Polynomial ans = p;
	return ans *= num;
};

Polynomial operator-(const Polynomial & p)
{
	Polynomial ans = p;
	return ans *= -1;
};

Polynomial operator/(const Polynomial & p,CoScalar num)
{
	Polynomial ans = p;
	return ans *= (1/num);
};

bool Polynomial::operator==(const Polynomial & p) const
{
	if(getDegree() != p.getDegree()) return false;

	for(unsigned int i = getDegree(); ; --i)
	{
		if(getCoeff(i) != p.getCoeff(i)) return false;

		if (i == 0) {
			break;
		}
	};

	return true;
};

CoScalar Polynomial::evaluate(CoScalar t) const
{
	CoScalar ans = 0;

	for(map<unsigned int,CoScalar>::const_iterator i = coeffs.begin(); i != coeffs.end();++i)
	{
		ans += CoScalar((i->second)*pow(t,i->first));
	};

	return ans;
};


Polynomial Polynomial::diff() const
{
	Polynomial ans;

	for(map<unsigned int,CoScalar>::const_iterator i = coeffs.begin(); i != coeffs.end();++i)
	{
		if( i->first != 0) ans.setCoeff((i->first) - 1,(i->second)*(i->first));
	};

	return ans;
};

Polynomial Polynomial::integrate() const
{
	Polynomial ans;

	for(map<unsigned int,CoScalar>::const_iterator i = coeffs.begin(); i != coeffs.end();++i)
	{
		ans.setCoeff((i->first) + 1,(i->second)/( (i->first) + 1 ));
	};

	return ans;
};

CoScalar newtonsMethod(const Polynomial & p,CoScalar startPt,CoScalar accuracy)
{
   //return -1 to show the method has failed on this ocassion, we will always be finding roots on (0,1)
	Polynomial pDiff = p.diff();
	CoScalar approxRoot,diffVal;
	CoScalar lastApproxRoot = startPt;
	int limit = 50 + int(1/accuracy); //should normally be within accurrcy < 10 steps, so limit is more than enough
	int i = 0;

	for(; i < limit; ++i)
	{
		diffVal = pDiff.evaluate(lastApproxRoot);

		if(diffVal == 0)
		{
			return -1;
		};

		approxRoot = lastApproxRoot - ((p.evaluate(lastApproxRoot) / diffVal));
      //cout << "Approx root = "<<approxRoot<<"\n";

		if((approxRoot - lastApproxRoot < accuracy) && (approxRoot - lastApproxRoot > -accuracy) ) break;


		lastApproxRoot = approxRoot;

	};

	if(i == limit)
	{
		return -1;
	};

	//cout << "Root of "<<p<<" is "<<approxRoot<<"\n";
	return approxRoot;
};

//on interval(0,t) within accuracy, repeated roots not returned
vector<CoScalar> Polynomial::getRoots(CoScalar t) const
{
	vector<CoScalar> roots;
	//remove repeated roots
	Polynomial testPoly = removeRepeatedRoots();

	if(testPoly.getDegree() == 0)
	{

		if(testPoly.getCoeff(0) == 0)
		{
			InfiniteRootsError ire;
			throw ire;
		};
	}
	else if(testPoly.getDegree() == 1)
	{
		CoScalar root = - (testPoly.getCoeff(0) / testPoly.getCoeff(1));

		if( root > 0 && root < t) roots.push_back( root );
	}
	else if(testPoly.getDegree() == 2)
	{
		CoScalar a = testPoly.getCoeff(2);
		CoScalar b = testPoly.getCoeff(1);
		CoScalar c = testPoly.getCoeff(0);
		CoScalar root1,root2;

		if( b*b - 4*a*c >= 0)
		{
			root1 = (-b - sqrt(b*b - 4*a*c) ) / (2*a);
			root2 = ( -b + sqrt(b*b - 4*a*c) ) / (2*a);
			if( root1 > 0 && root1 < t) roots.push_back( root1 );
			if( root2 > 0 && root2 < t && root1 != root2) roots.push_back( root2 );
		};

	}
	else
	{
		pair<vector<pair<intervalEnd,intervalEnd> >,vector<CoScalar> > isolExact = testPoly.isolateRoots(t,Polynomial::accuracy);

		for(vector<pair<intervalEnd,intervalEnd> >::iterator i = isolExact.first.begin(); i != isolExact.first.end(); ++i)
		{
			roots.push_back((i->first.first + i->second.first)/2);
		};

		for(vector<CoScalar>::iterator j = isolExact.second.begin(); j != isolExact.second.end(); ++j)
		{
			roots.push_back(*j);
		};

	};

   vector<CoScalar> copyRoots = roots;
   roots.clear();
   //remove small roots
   for(vector<CoScalar>::iterator i = copyRoots.begin(); i != copyRoots.end(); ++i)
   {
        if(*i > 0.01) roots.push_back(*i); //tolerance value
   };
   copyRoots.clear();

	return roots;
};

void Polynomial::write(ostream & o) const
{
	if(coeffs.size()==0)
	{
		o << "0";
	}
	else
	{
		map<unsigned int,CoScalar>::const_iterator i = coeffs.end();
		for(unsigned int count=0; count != coeffs.size(); ++count)
		{
			--i;

			if( count != 0)
			{
				if(i->second >= 0)
					o << " + ";
				else
					o << " - ";
			}
			else if(i->second < 0)
			{
					o << " - ";
			};

			if(i->second >= 0)
			{
				if( (i->second != 1) || (i->first == 0) ) o << i->second;
			}
			else
 			{
 				if( (i->second != -1) || (i->first == 0) ) o << -(i->second);
 			};

			if(i->first == 1)
				o << "t";

 			else if(i->first != 0)
 				o << "t^" << i->first;

			if( i == coeffs.begin()) break;


		};

	};

	//o << "\n";
};

ostream & operator << (ostream & o,const Polynomial & p)
{

	p.write(o);

	return o;
};

void Polynomial::removeSmallCoeffs()
{
	if(coeffs.size()==0) return;

	unsigned int deg = getDegree();
	for(unsigned int pow = 0; pow <= deg; ++pow)
	{
		map<unsigned int,CoScalar>::iterator i = coeffs.find(pow);
		if(i != coeffs.end() && i->second < tooSmall && i->second > -tooSmall) coeffs.erase(i);
	};

};

//divides this by d and returns the quotient and remainder
pair<Polynomial,Polynomial> Polynomial::divide(const Polynomial & d) const
{
	Polynomial quot;

	unsigned int numDegree = getDegree();
	unsigned int denomDegree = d.getDegree();

	if(numDegree < denomDegree)	return make_pair(quot,*this);

	CoScalar quotCo;
	CoScalar leadingDenomCo = d.getCoeff(denomDegree);

	Polynomial aPoly = *this;
	Polynomial bPoly;


	unsigned int i = numDegree + 1;
	unsigned int stepBound = 2*numDegree;

	for(unsigned int c = 0; c < stepBound ; ++c)
	{

		--i;
		quotCo = aPoly.getCoeff(i) / leadingDenomCo;


		bPoly.setCoeff(i - denomDegree, quotCo);
		quot += bPoly;

		aPoly = aPoly - d * bPoly;
		aPoly.removeSmallCoeffs();
		if(i - denomDegree == 0) break;
		bPoly.setCoeff(i - denomDegree,0);

	};

	return make_pair(quot,aPoly);
};

Polynomial Polynomial::getGCD(const Polynomial & p) const
{
	Polynomial gcd,num,denom;

	if(getDegree() >= p.getDegree())
	{
		num = *this;
		denom = p;
	}
	else
	{
		num = p;
		denom = *this;
	};


	Polynomial rem = denom;

	unsigned int stepBound = 3*num.getDegree();


	for(unsigned int c = 0; c < stepBound ; ++c)
	{


		gcd = rem;
		rem = num.divide(denom).second;

		if(!((rem.getDegree() == 0) && (rem.getCoeff(0) == 0)))
		{
			num = denom;
			denom = rem;
		}
		else
			break;

	};

	if((gcd.getDegree() == 0) && (gcd.getCoeff(0) == 0)) gcd.setCoeff(0,1);

	return gcd;
};


Polynomial Polynomial::removeRepeatedRoots() const
{

	Polynomial d = diff();
	Polynomial gcd = getGCD(d);
	Polynomial ans;

	if(gcd.getDegree() > 0)
	{
		ans = divide(gcd).first;
	}

	else
	{
		ans = *this;
	};


	return ans;
};

//R(p(t)) = t^(deg(p))*p(1/t)
Polynomial mapR(const Polynomial & p)

{
	Polynomial ans;
	unsigned int deg = p.getDegree();

	for(unsigned int i = 0; i < deg + 1; ++i)
	{
		ans.setCoeff(i,p.getCoeff(deg - i));
	};
	//cout << "R( "<<p<<" ) = "<<ans<<"\n";
	return ans;
};

//H_c(p(t)) = p(ct) for c in Reals
Polynomial mapHc(const Polynomial & p,CoScalar c)
{
	Polynomial ans;
	unsigned int deg = p.getDegree();
   CoScalar cPowi = 1; //c to the power of i

	for(unsigned int i = 0; i < deg + 1; ++i)
	{
		//ans.setCoeff(i,p.getCoeff(i)*pow(c,i));
    ans.setCoeff(i,p.getCoeff(i)*cPowi);
    cPowi = cPowi*c;
	};

	return ans;
};

//no. of ways to choose r things from n things, i.e. n!/[(n-r)!r!]
CoScalar choose(unsigned int n,unsigned int r)
{
	CoScalar ans = 1;

	for(unsigned int i = n; i > n - r; --i)
	{
		ans = (ans*i)/(n-i+1);

	};

	return ans;
};

//T_c(p(t)) = p(t+c) for c in Reals
Polynomial mapTc(const Polynomial & p,CoScalar c)
{
	Polynomial ans;

	unsigned int deg = p.getDegree();


	for(unsigned int i = 0; i < deg + 1; ++i)
	{
		Polynomial aPoly;

		for(unsigned int j = 0; j < i + 1; ++j)
		{
			aPoly.setCoeff(j,p.getCoeff(i)*pow(c,i-j)*choose(i,j));
		};

		ans += aPoly;

	};
	//cout << "T( "<<p<<","<<c<<" ) = "<<ans<<"\n";
	return ans;

};

//P_k_c (p(t)) = 2^{kn} p( (t+c)/(2^k) ) for c in Reals, k integer, n= degree(p)
Polynomial mapPkc(const Polynomial & p,CoScalar k,CoScalar c)
{
	Polynomial ans;

	unsigned int deg = p.getDegree();

	for(unsigned int i = 0; i < deg + 1; ++i)
	{
		Polynomial aPoly;
      CoScalar cPowij = 1;
      /*
		for(unsigned int j = 0; j < i + 1; ++j)
		{
      aPoly.setCoeff(j, pow(2,(deg-i)*k) * p.getCoeff(i) * pow(c,i-j)*choose(i,j));

		};*/


       	for(unsigned int j = i; ; --j)
		{
       aPoly.setCoeff(j, pow(2,(deg-i)*k) * p.getCoeff(i) * cPowij*choose(i,j));
       cPowij = cPowij * c;

        if(j==0) break;
		};
		ans += aPoly;

	};
	//cout << "P( "<<p<<","<<k<<" , "<<c<<" ) = "<<ans<<"\n";
	return ans;

};


int sign(CoScalar a)
{
	if(a > 0) return 1;

	if(a < 0) return -1;

	return 0;

};

unsigned int descartesBound(const Polynomial & p)
{
	Polynomial testPoly = mapTc(mapR(p),1);

	//testPoly.removeSmallCoeffs(); //may be dodgey- remove?
	if(!testPoly.checkPolynomialCoeffs())
	{
		PolyRootError pre;
		throw pre;
	};

	unsigned int noSignChanges = 0;
	CoScalar num;
	int lastSign = 0;
	int thisSign;
	unsigned int seqNo = 1;
	unsigned int extraVal = 0;

	for(unsigned int i = 0; i <= testPoly.getDegree() ; ++i)
	{
		num = testPoly.getCoeff(i);

		if(num != 0)
		{
			//doubles are accurate to 14 d.p. s so check that sign(num) is well defined
			if(num < Polynomial::tooSmall && num > -Polynomial::tooSmall)
			{
				if( !((i+1) < testPoly.getDegree() && sign(testPoly.getCoeff(i+1)) * lastSign == -1) )
				{
					if((seqNo == 1) || (i == testPoly.getDegree()))
						extraVal++;
					else
					{
						PolyRootError pre;
						throw pre;
					};

				};
			};

			thisSign = sign(num);
			if(thisSign*lastSign == -1) ++noSignChanges;
			lastSign = thisSign;
			++seqNo;
		};

	};

	//check that no of sign changes is well defined, i.e. either 0, 1 or > 1
	//cout << "sign changes = "<<noSignChanges<<" ||Extra = "<<extraVal<<"\n\n";
	if(noSignChanges == 0 && extraVal == 1)
	{

		CoScalar val0 = p.evaluate(0);
		CoScalar val1 = p.evaluate(1);
		if((val0 < Polynomial::tooSmall && val0 > -Polynomial::tooSmall)||
		   (val1 < Polynomial::tooSmall && val1 > -Polynomial::tooSmall))
		{
			PolyRootError pre;
			throw pre;
		};

		if(p.evaluate(0) * p.evaluate(1) < 0) noSignChanges = 1;

	}
	else if(extraVal != 0 && noSignChanges <= 1)
	{
		PolyRootError pre;
		throw pre;
	};



	return noSignChanges;
};

//test for infinity and NaN, returns true if there are no dodgey numbers ie infinity and NaN, (note coeffs value 0 not stored)
bool Polynomial::checkPolynomialCoeffs() const
{
	if(coeffs.size()==0) return true;

	for(map<unsigned int,CoScalar>::const_iterator i = coeffs.begin(); i != coeffs.end();++i)
	{
		if( (2 * (i->second) == i->second) || (!( i->second == i->second)) )
		{
			return false;
		};

	};

	return true;
};

CoScalar findRootUsingNewtonsMethod(const Polynomial & p,CoScalar intStart, CoScalar intEnd, CoScalar accuracy)
{
   CoScalar theRoot = -1;
   int step = 1;

   //use newtons method by symatically choosing different starting points, bisecting the interval(s) each time
   //the starting points will be dense on the interval at infinity but stop once within degree of accuracy
   //should normally find root at first point anyway - using only 10 steps of newtons method
   while(theRoot == -1 && pow(0.5,step-1) > accuracy)
   {

       for(int j = 0; j < pow(2,step-1); ++j)
       {
         theRoot = newtonsMethod(p,(intEnd-intStart)*(pow(0.5,step) + j*pow(0.5,step+1))+intStart,accuracy);
       };

       if(theRoot <= intStart || theRoot >= intEnd) theRoot = -1;
       step++;
    };

   if(theRoot == -1)
	{
		PolyRootError pre;
		throw pre;
	};

    return theRoot;
};

//isolate roots of polynomial p on interval (0,1) within accuracy
pair<vector<pair<intervalEnd,intervalEnd> >,vector<CoScalar> > descartesAlg(const Polynomial & p,CoScalar accuracy = 0)
{
	vector<pair<intervalEnd,intervalEnd> > isol;
	vector<CoScalar> exact;
	map<unsigned int,pair<CoScalar,CoScalar> > searchTree;
	unsigned int treeIter = 1;
	unsigned int treeIterAdd = 1;
	unsigned int desBnd;
	Polynomial testPoly;
	CoScalar intStart,intEnd,exactRoot,eval;
	bool addToTree;

	//initialize tree
	searchTree[1] = make_pair(0,0);

	do
	{
		//get new node, (front of list)
		map<unsigned int,pair<CoScalar,CoScalar> >::iterator i = searchTree.find(treeIter);
		if(i == searchTree.end()) break;

		testPoly = mapPkc(p,i->second.first,i->second.second);

		desBnd = descartesBound(testPoly);


		intStart = ((i->second.second) / (pow(2,i->second.first)));
		intEnd = ((i->second.second + 1) / (pow(2,i->second.first)));

		addToTree = false;
		if(desBnd == 1)
		{
			if( (accuracy == 0) || ( (intEnd - intStart) < accuracy) )
			{
				isol.push_back(make_pair(make_pair(intStart,false),make_pair(intEnd,false)));
			}
			else
			{
				CoScalar theRoot = findRootUsingNewtonsMethod(testPoly,intStart,intEnd,accuracy);
            exact.push_back(theRoot);
            //addToTree = true; //use this to use descartesAlg to find roots
			};

		}
		else if(desBnd > 1)
		{
			if( (intEnd - intStart) < accuracy ) //maybe repeated root or curve that is very close to zero
			{
				isol.push_back(make_pair(make_pair(intStart,false),make_pair(intEnd,false)));
			}
			else
			{
				addToTree = true;
			};
		};

		//add smaller intervals to investigate

		if(addToTree)
		{
			exactRoot = -1;
			eval = testPoly.evaluate(0.5);

			if( (eval < Polynomial::tooSmall) && (eval > -(Polynomial::tooSmall)) )
			{
				exactRoot = (2*(i->second.second) + 1) / (pow(2,(i->second.first)+1));
				exact.push_back(exactRoot);
			};

			if((exactRoot == -1) || (desBnd > 1))
			{
				searchTree[++treeIterAdd] = make_pair(i->second.first + 1, 2 * i->second.second);
				searchTree[++treeIterAdd] = make_pair(i->second.first + 1, 2 * i->second.second + 1);
			};

		};

		searchTree.erase(i);
		++treeIter;

	}while(true);

	return make_pair(isol,exact);
};

//for interval (0,t), each interval returned contains exactly one root, no. intervals == no. roots
//make intervals smaller than 'accuracy', default is to just isolate roots from one another
pair<vector<pair<intervalEnd,intervalEnd> >,vector<CoScalar> > Polynomial::isolateRoots(CoScalar t,CoScalar accuracy) const
{
	pair<vector<pair<intervalEnd,intervalEnd> >,vector<CoScalar> > isolExact;

	//map poly on (0,t) to a poly on (0,1)
	Polynomial testPoly = mapHc(*this,t);
          // cout << *this << "\n"<<"testpoly ="<<testPoly<<"\n";
	isolExact = descartesAlg(testPoly,accuracy/t);

	//map roots back to original interval

	for(vector<pair<intervalEnd,intervalEnd> >::iterator i = isolExact.first.begin(); i != isolExact.first.end(); ++i)
	{
		i->first.first = i->first.first * t;
		i->second.first = i->second.first * t;
	};

	for(vector<CoScalar>::iterator j = isolExact.second.begin(); j != isolExact.second.end(); ++j)
	{
		*j = (*j) * t;
	};



	return isolExact;
};

//do roots exist on the open interval (0,t)?
bool Polynomial::rootsExist(CoScalar t) const
{
   if(getDegree() < 3) {if(getRoots(t).size() > 0) return true; else return false;};

	pair<vector<pair<intervalEnd,intervalEnd> >,vector<CoScalar> > isolExact;
	bool ans = false;



	//remove repeated roots
	Polynomial testPoly = removeRepeatedRoots();

	isolExact = testPoly.isolateRoots(t);

	if( (!(isolExact.first.empty())) || (!(isolExact.second.empty())) ) ans = true;


	return ans;
};

Intervals getIntervalsFromRoots(vector<CoScalar> roots, const CtsFunction * ctsFtn, CoScalar t,bool strict)
{
    Intervals theAns;
	std::sort(roots.begin(),roots.end());

	pair<CoScalar,bool> startPt,endPt;

	if(roots.size() == 0)
	{

		if( ctsFtn->evaluate(t/2.0) > 0 )
		{

			theAns.intervals.push_back(make_pair( make_pair(0,true) , make_pair(t, true)));
		};

		return theAns;
	};

	//determine if starting satisfied or not
	Intervals aInt;
	vector< CoScalar >::const_iterator i = roots.begin();
   bool intStat = false;
   if(ctsFtn->evaluate(0) > 0) intStat = true;
   else if(ctsFtn->evaluate(0) == 0)
   {
     intStat = ctsFtn->evaluate((*i)/2.0) > 0;
   };

	if(intStat)
	{
		startPt = make_pair(0,true);
		endPt = make_pair(*i,!(strict));
		aInt.intervals.push_back(make_pair(startPt,endPt));
		theAns = setUnion(theAns,aInt);
	};


	for(; i != roots.end();++i)
	{
		if(*i != t)
		{
			aInt.intervals.clear();
			startPt = make_pair(*i,!(strict));

			if((i+1) != roots.end())
			{
				endPt = make_pair(*(i+1),!(strict));
			}
			else
			{
				endPt = make_pair(t,true);
			};

         CoScalar eval = ctsFtn->evaluate((startPt.first + endPt.first)/2.0);
			//in case of a repeated root, if a dodgey number let it go
			if(eval > 0 || (2 * eval == eval) || (!( eval == eval)))
			{
				aInt.intervals.push_back(make_pair(startPt,endPt));
				theAns = setUnion(theAns,aInt);
			}
			else if(!(strict))
			{
				aInt.intervals.push_back(make_pair(startPt,startPt));
				theAns = setUnion(theAns,aInt);
			};

		};
    };

	//cout << " \\\\ \\> The poly $"<<*this<<"$ is satisfied on "<<theAns<<"\\\\\n";
	return theAns;

};

Intervals Polynomial::getIntervals(const Comparison * comp, const State* s,CoScalar t) const
{
   Intervals theAns;

	if( getDegree() == 0 )
	{
		if(comp->evaluate(s))
		{
			theAns.intervals.push_back(make_pair( make_pair(0,true) , make_pair(t, true)));
		};

		return theAns;
	}
	else if(comp->getComparison()->getOp() == E_EQUALS)
	{
		if( (getDegree() == 0) && (getCoeff(0) == 0))
		{
			theAns.intervals.push_back(make_pair( make_pair(0,true) , make_pair(t, true)));
		};

		return theAns;
	};

	bool strict = (comp->getComparison()->getOp() == E_GREATER) || (comp->getComparison()->getOp() == E_LESS);


	vector<CoScalar> roots = comp->getRootsForIntervals(s,t);
   theAns = getIntervalsFromRoots(roots,this,t,strict);
  return theAns;
};

Intervals Exponential::getIntervals(const Comparison * comp, const State* s,CoScalar t) const
{    //f(t) = K e^(poly) + c   for    f # offSet, where # is >,  <=, > or >=
    //return getApproxPoly(t).getIntervals(comp,s,t);
   Intervals theAns;
  // pair<CoScalar,bool> startPt,endPt;

   bool rootsExist;
   bool strict = (comp->getComparison()->getOp() == E_GREATER) || (comp->getComparison()->getOp() == E_LESS);
   //bool greater = (comp->getComparison()->getOp() == E_GREATER) || (comp->getComparison()->getOp() == E_GREATEQ);

   if(comp->getComparison()->getOp() == E_EQUALS)
   {
     if(K == 0 && offSet - c == 0) theAns.intervals.push_back(make_pair( make_pair(0,true) , make_pair(t, true)));
     return theAns;
   };

   vector<CoScalar> roots;
   if( ((offSet - c)/K) <= 0 ) rootsExist = false;

   else
   {
     Polynomial aPoly = *poly - log((offSet -c)/K);

     try{roots = aPoly.getRoots(t);}
     catch(PolyRootError & pre){throw pre;};

     rootsExist = (roots.size() > 0);
   };

   if(!rootsExist)
   {
        if(comp->evaluateAtPoint(s))  //no roots so is either satisfied or not over whole interval - so just evaluate one pt
        {
             theAns.intervals.push_back(make_pair( make_pair(0,true) , make_pair(t, true)));
        };
        return theAns;
   };

   //if(greater)
   //{
     const Exponential * aExp = new Exponential(K,new Polynomial(*poly),c-offSet);
     theAns = getIntervalsFromRoots(roots,aExp,t,strict);
     delete aExp;
 /*  }
   else
   {
     const Exponential * aExp = new Exponential(-K,new Polynomial(*poly),-c+offSet);
     theAns = getIntervalsFromRoots(roots,aExp,t,strict);
     delete aExp;
   };
   */
   return theAns;
};

Intervals NumericalSolution::getIntervals(const Comparison * comp, const State* s,CoScalar t) const
{
   Intervals theAns;

   bool strict = (comp->getComparison()->getOp() == E_GREATER) || (comp->getComparison()->getOp() == E_LESS);
   bool greater = (comp->getComparison()->getOp() == E_GREATER) || (comp->getComparison()->getOp() == E_GREATEQ);

   pair<CoScalar,bool> startPt,endPt;
   //determine if starting satisfied or not
	Intervals aInt;
	map<double,CoScalar>::const_iterator j = points.begin();

  //initial interval comparison is satisfied on
	if((j->second > offSet &&  greater) || (j->second < offSet && !greater))
	{
		startPt = make_pair(0,true);

      CoScalar lastPoint = j->second;
      double lastTime = j->first;
      bool positive = j->second > offSet;   //record sign of last non-zero point
      j++;

      while( j != points.end())
      {
          if((positive && j->second < offSet) || (!positive && j->second > offSet) || (j->second == offSet && strict) ) break;

          if(j->second < offSet) positive = false;
          else if(j->second > offSet) positive = true;

          lastPoint = j->second;
          lastTime = j->first;
          j++;
      };

       double endPoint;
       if(j != points.end()) endPoint = lastTime+(j->first - lastTime)*(fabs(lastPoint)/(fabs(lastPoint) + fabs(j->second)));
       else {endPoint = lastTime; strict = false;};

		endPt = make_pair(endPoint,!strict);
		aInt.intervals.push_back(make_pair(startPt,endPt));
		theAns = setUnion(theAns,aInt);
	}
	else if(j->second == offSet && !strict)
	{

      startPt = make_pair(0,true);
      //CoScalar lastPoint = j->second; // set but never used
      double lastTime = j->first;
      j++;
//      lastPoint = lastPoint; // why was this line of code here?
      while( j != points.end())
      {
          if( j->second != offSet) break;
          //lastPoint = j->second;
          lastTime = j->first;
          j++;
      };

		endPt = make_pair(lastTime,!strict);
		aInt.intervals.push_back(make_pair(startPt,endPt));
		theAns = setUnion(theAns,aInt);
	};


   //define other intervals now, when the points change sign the interval end points are formed,
   //care is needed for values that are zero depending on if comparison is strict
   bool startPointDefined = false;
	while(j != points.end())
	{                       //point at j is not satisfied to begin with
     startPointDefined = false;
     Intervals aInt;
     //firstly find starting point
     if(j->second == offSet && strict)
     {
       double aTime = j->first;
       j++;
       if(j == points.end()) break;
       if((j->second > offSet &&  greater) || (j->second < offSet && !greater))
       {
         startPt = make_pair(aTime,!strict);
         startPointDefined = true;
       };
     };



     if(!startPointDefined)
     {
           CoScalar lastPoint = j->second;
           double lastTime = j->first;
           bool positive = j->second > offSet;   //record sign of last non-zero point

          while( j != points.end())
          {
              if((positive && j->second < offSet) || (!positive && j->second > offSet) || (j->second == offSet && strict) )
              {
                startPointDefined = true;
                break;
                };

              if(j->second < offSet) positive = false;
              else if(j->second > offSet) positive = true;

              lastPoint = j->second;
              lastTime = j->first;
              j++;
          };

          if(startPointDefined) startPt = make_pair(lastTime + (j->first - lastTime)*(fabs(lastPoint)/(fabs(lastPoint) + fabs(j->second))),!(strict));


      };

       //find the end point now
       if(startPointDefined)
       {
             CoScalar lastPoint = j->second;
             double lastTime = j->first;
             double theEndPoint;
             bool positive = j->second > offSet;   //record sign of last non-zero point
             j++;

             while(true)
            {
                if( j == points.end() ) {theEndPoint = lastTime; strict = false; break;};
                if((positive && j->second < offSet) || (!positive && j->second > offSet) || (j->second == offSet && strict) )
                {
                    theEndPoint = lastTime + (j->first - lastTime)*(fabs(lastPoint)/(fabs(lastPoint) + fabs(j->second)));
                    break;
                };

                if(j->second < offSet) positive = false;
                else if(j->second > offSet) positive = true;

                lastPoint = j->second;
                lastTime = j->first;
                j++;
            };

            endPt = make_pair(theEndPoint,!(strict));
            aInt.intervals.push_back(make_pair(startPt,endPt));
            theAns = setUnion(theAns,aInt);

       };

	};


   return theAns;
};

bool Polynomial::checkInvariant(const Comparison * comp, const State* s,CoScalar t,bool rhsIntervalOpen) const
{
   	if(getDegree() == 0)
      	{
              return comp->evaluateAtPoint(s);
      	}
      	else
      	{

      		int degree = getDegree();

      		if( (comp->getComparison()->getOp() == E_EQUALS) )
      		{
      			if( (degree == 0) && (getCoeff(0) == 0))
      				return true;
      			else
      				return false;
      		};



      		bool strict = (comp->getComparison()->getOp() == E_GREATER) || (comp->getComparison()->getOp() == E_LESS);

      		//check rhs side of interval so we are checking on ( , ] if req'd
      		if(strict && !(rhsIntervalOpen) && (evaluate(t) <= 0 ) )
      		{
      			return false;
      		};

      		if(degree == 1)
      		{
//      		cout << "WE HAVE " << evaluate(0) << " AND " << evaluate(t) << "\n";
      			if( (evaluate(0) > - accuracy) && (evaluate(t) > - accuracy))
      			{
      				return true;
      			};
      			return false;
      		}
      		else if( degree > 1 )
      		{
      			//check end points first, otherwise look for roots in interval
      			//being weary of repeated roots for non strict inequalities
      			if( (evaluate(0) < 0) || (evaluate(t) < 0))
      			{
      				return false;
      			}
      			else
      			{
      				//if(strict) return !(rootsExist(t));

                 if(strict)
                 {
                    bool rtsExist = rootsExist(t);
                    if(!rtsExist) return evaluate(t/2.0) > 0;
                    return !rtsExist;
                 };

      				pair<vector<pair<intervalEnd,intervalEnd> >,vector<CoScalar> > isolExact = removeRepeatedRoots().isolateRoots(t);

      				vector<pair<intervalEnd,intervalEnd> >::iterator ints = isolExact.first.begin();

      				vector<CoScalar>::iterator exact = isolExact.second.begin();


      				CoScalar pt1,pt2,nextpt1;
      				pt1 = 0;

      				while(ints != isolExact.first.end() && exact != isolExact.second.end() )
      				{
      					if(exact == isolExact.second.end())
      					{
      						pt2 = ints->first.first;
      						nextpt1 = ints->second.first;
      						++ints;
      					}
      					else if(ints == isolExact.first.end())
      					{
      						pt2 = *exact;
      						nextpt1 = pt2;
      						++exact;
      					}
      					else if(ints->first.first < *exact)
      					{
      						pt2 = ints->first.first;
      						nextpt1 = ints->second.first;
      						++ints;

      					}
      					else
      					{
      						pt2 = *exact;
      						nextpt1 = pt2;
      						++exact;
      					};

      					if(evaluate((pt1+pt2)/2) < 0) return false;
      					pt1 = nextpt1;
      				};

      				return true;
      			};

      		}
      		else
      		{
      			InvariantError ie;
      			throw ie;

      		};

      	};

      return false;


};

bool Exponential::rootsExist(CoScalar t) const //on open interval (0,t)
{
   vector<CoScalar> roots;
   if( ((offSet - c)/K) <= 0 ) return false;
   else
   {
     Polynomial aPoly = *poly - log((offSet -c)/K);

     try{roots = aPoly.getRoots(t);}
     catch(PolyRootError & pre){throw pre;};

     return (roots.size() > 0);
   };

};

bool Exponential::checkInvariant(const Comparison * comp, const State* s,CoScalar t,bool rhsIntervalOpen) const
{

      if(comp->getComparison()->getOp() == E_EQUALS)
      {
          if(K == 0 && offSet - c == 0) return true;
          return false;
      };

      bool strict = (comp->getComparison()->getOp() == E_GREATER) || (comp->getComparison()->getOp() == E_LESS);

		//check rhs side of interval so we are checking on ( , ] if req'd
		if(strict && !(rhsIntervalOpen) && (evaluate(t)-offSet <= 0 ) )
		{
			return false;
		};

			//check end points first, otherwise look for roots in interval
			//being weary of repeated roots for non strict inequalities
			if( (evaluate(0)-offSet < 0) || (evaluate(t)-offSet < 0))
			{
				return false;
			}
			else
			{
				if(strict)
            {
               bool rtsExist = rootsExist(t);
               if(!rtsExist) return evaluate(t/2.0)-offSet > 0;
               return !rtsExist;
            };

            vector<CoScalar> roots;
            if( ((offSet - c)/K) <= 0 ) return true;
            else
            {
              Polynomial aPoly = *poly - log((offSet -c)/K);

              try{roots = aPoly.getRoots(t);}
              catch(PolyRootError & pre){throw pre;};
            };

            //if non strict and there may be repeated roots!
            CoScalar previousPt = 0;
            for(vector<CoScalar>::const_iterator i = roots.begin(); i != roots.end(); ++i)
            {
                if(evaluate((previousPt+(*i))/2) -offSet < 0) return false;
                previousPt = *i;
            };

				return true;
			};


return false;

};


bool NumericalSolution::checkInvariant(const Comparison * comp, const State* s,CoScalar t,bool rhsIntervalOpen) const
{

bool strict = (comp->getComparison()->getOp() == E_GREATER) || (comp->getComparison()->getOp() == E_LESS);
bool greater = (comp->getComparison()->getOp() == E_GREATER) || (comp->getComparison()->getOp() == E_GREATEQ);

//check initial point
map<double,CoScalar>::const_iterator j = points.begin();
if( (greater && (j->second - offSet < 0)) ||
      (!greater && (j->second - offSet > 0)) ) return false;


for( ; j != points.end(); )
{
  ++j; if( j->first == t) break;
  if( ((greater && (j->second - offSet < 0)) || (j->second - offSet <= 0 && !strict)) ||
      ((!greater && (j->second - offSet > 0)) || (j->second - offSet >= 0 && !strict))) return false;


};
//check end point
if(j != points.end())
{
if( ((greater && (j->second - offSet < 0)) || (j->second - offSet <= 0 && !strict && !rhsIntervalOpen)) ||
      ((!greater && (j->second - offSet > 0)) || (j->second - offSet >= 0 && !strict && !rhsIntervalOpen))) return false;
};

return true;

};

//build points using Runge Kutta Fehlberg method
void NumericalSolution::buildPoints(CoScalar a0,CoScalar b0,CoScalar y,CoScalar accuracy)
{
    CoScalar a2 = 1.0/4.0;
    CoScalar b2 = 1.0/4.0;
    CoScalar a3 = 3.0/8.0;
    CoScalar b3 = 3.0/32.0;
    CoScalar c3 = 9.0/32.0;
    CoScalar a4 = 12.0/13.0;
    CoScalar b4 = 1932.0/2197.0;
    CoScalar c4 = -7200.0/2197.0;
    CoScalar d4 = 7296.0/2197.0;
    CoScalar a5 = 1.0;
    CoScalar b5 = 439.0/216.0;

    CoScalar c5 = -8.0;
    CoScalar d5 = 3680.0/513.0;
    CoScalar e5 = -845.0/4104.0;
    CoScalar a6 = 1.0/2.0;
    CoScalar b6 = -8.0/27.0;
    CoScalar c6 = 2.0;
    CoScalar d6 = 3544.0/2565.0;
    CoScalar e6 = 1859.0/4104.0;
    CoScalar f6 = -11.0/40.0;
    CoScalar r1 = 1.0/360.0;
    CoScalar r3 = -128.0/4275.0;
    CoScalar r4 = -2197.0/75240.0;
    CoScalar r5 = 1.0/50.0;
    CoScalar r6 = 2.0/55.0;
    CoScalar n1 = 25.0/216.0;
    CoScalar n3 = 1408.0/2565.0;
    CoScalar n4 = 2197.0/4104.0;
    CoScalar n5 = -1.0/5.0;
    CoScalar k1,k2,k3,k4,k5,k6;
    CoScalar hmin = 0.001;
    CoScalar hmax = 1.0; //orignially 0.25
    //CoScalar maxPoints = 200;
    CoScalar h = 1.0;     //orignially 0.25
    CoScalar t = a0;
    CoScalar br = b0 - 0.00001*fabs(b0);
    CoScalar err, ynew,s;

    //points.push_back(make_pair(double(a0),double(y))); //initial point
    points[double(a0)] = y;

    while(t < b0)
    {

      if(t + h > br) h = b0 - t;
      k1 = h*evaluateDiff(t, y);
      k2 = h*evaluateDiff(t + a2*h, y + b2*k1);
      k3 = h*evaluateDiff(t + a3*h, y + b3*k1 + c3*k2);
      k4 = h*evaluateDiff(t + a4*h, y + b4*k1 + c4*k2 + d4*k3);
      k5 = h*evaluateDiff(t + a5*h, y + b5*k1 + c5*k2 + d5*k3 + e5*k4);
      k6 = h*evaluateDiff(t + a6*h, y + b6*k1 + c6*k2 + d6*k3 + e6*k4 + f6*k5);

      err = fabs(r1*k1 + r3*k3 + r4*k4 + r5*k5 + r6*k6);
      ynew = y + n1*k1 + n3*k3 + n4*k4 + n5*k5;

      if( err < accuracy || h < 2* hmin)
      {
          if(t + h > br)
          {
            //points.push_back(make_pair(b0,ynew));
            points[double(b0)] = ynew;
             t = b0;
             }
          else
          {
            //points.push_back(make_pair(double(t+h),double(ynew)));
            points[double(t+h)] = ynew;
            t = t + h;
          };
          y = ynew;   //cout << t<< " , "<< ynew <<"\\\\\n";
      };

      if(err ==0) s = 0;
      else s = pow(((accuracy*h)/(2*err)),0.25);

      if(s < 0.1) s = 0.1;
      if(s > 4.0) s = 4.0;

      h = s*h;

      if(h>hmax) h = hmax;
      else if(h < hmin) h = hmin;

      //if(maxPoints == points.size()) break;
    };





};

CoScalar BatteryCharge::evaluateDiff(CoScalar t,CoScalar y)
{
    CoScalar dis = 0;
    for(vector<pair<const CtsFunction *,bool> >::const_iterator i = discharge.begin(); i != discharge.end(); ++i)
    {
        if(i->second) dis += (*i).first->evaluate(t);
        else dis -= (*i).first->evaluate(t);
    };

    return ((poly->evaluate(t))* (m - y)) + dis;
};

void BatteryCharge::write(ostream & o) const
{
	if(LaTeX) o << " $Numerical solution of:$ \\frac{dy}{dt} = ";
  else o << "Numerical solution of: dy/dt = ";
   o << "(" << *poly << ") ("<< m << " - "<< "y)";
     for(vector<pair<const CtsFunction *,bool> >::const_iterator i = discharge.begin(); i != discharge.end(); ++i)
    {
        if(i->second) o << " + " << *(i->first);
        else o << " - " << *(i->first);
    };


};

ostream & operator << (ostream & o,const BatteryCharge & p)
{

	p.write(o);

	return o;
};

vector<CoScalar> NumericalSolution::getRoots(CoScalar t) const
{
    vector<CoScalar> roots;
    NumError ne;
    throw ne;
    return roots;
};

Polynomial NumericalSolution::getApproxPoly(CoScalar endInt) const
{
    Polynomial poly;

    ApproxPolyError ape;
    throw ape;

      poly.setCoeff(0,1);
    return poly;
};

pair<vector<pair<intervalEnd,intervalEnd> >,vector<CoScalar> > NumericalSolution::isolateRoots(CoScalar t,CoScalar accuracy) const
{
    pair<vector<pair<intervalEnd,intervalEnd> >,vector<CoScalar> > isol;

    NumError ne;
    throw ne;

    return isol;
};

CoScalar NumericalSolution::evaluate(CoScalar t) const
{
    CoScalar ans =0;
    map<double,CoScalar>::const_iterator j = points.find(t);
    if(j != points.end()) return j->second;
    //else interpolate

    map<double,CoScalar>::const_iterator i = points.begin();
    double lastTime = i->first;
    CoScalar lastPoint = i->second;
    ++i;
      //should only ever evaluate the end point which will be found above so the stuff below is OK
    for( ; i != points.end(); ++i)
    {
      if( t <= i->first  && t >= lastTime)
      {

        ans = lastPoint + ((i->second - lastPoint)/(i->first - lastTime))*(t - lastTime);

        return ans;
      };

       lastTime = i->first;
       lastPoint = i->second;
    };

    return ans;

};

bool NumericalSolution::isLinear() const
{
    if(points.size() > 2) return false;

    return true;
};

};
