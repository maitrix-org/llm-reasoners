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


/******************************************************************************
 *
 *	random.h
 *	
 *	Header file for random number generator class declarations and definitions.
 *
 *	This header file does not depend on any of the other VecMat files, and it
 *  can be used independently of the rest of the software.
 *
 ******************************************************************************
 *
 *	VecMat Software, version 1.05
 *	By Kevin Dolan (2002)
 *	kdolan@mailaps.org
 *
 *****************************************************************************/

#include <cmath>
#include <ctime>

#ifndef __RANDOM_H
#define __RANDOM_H

namespace VAL {
  
// These classes must use 32 bit integers. If int is not 32 bit for your
// compiler, then replace the following typedefs with a type that is.
typedef unsigned int UINT32;
typedef int          INT32;

static const UINT32 CM_   = 69069;
static const UINT32 CA_   = 1234567;
static const UINT32 MASK_ = 65535;
static const UINT32 MWC1_ = 36939;
static const UINT32 MWC2_ = 18000;

class BaseGen
// Generate uniformly distributed unsigned 32 bit integers on the closed
// interval (0, 2^32 - 1).
{
protected:
	mutable UINT32 z_, w_, jsr_, jc_;

public:
	BaseGen(UINT32 seed = 0)
	{
		if (!seed) seed = static_cast<UINT32>(time(0));
		jc_  = seed;
		z_   = (jc_ = CM_ * jc_ + CA_);
		w_   = (jc_ = CM_ * jc_ + CA_);
		jsr_ = (jc_ = CM_ * jc_ + CA_);
		jc_  = CM_ * jc_ + CA_;
	}

	BaseGen(const BaseGen& gen)
	{
		jc_  = gen();
		z_   = (jc_ = CM_ * jc_ + CA_);
		w_   = (jc_ = CM_ * jc_ + CA_);
		jsr_ = (jc_ = CM_ * jc_ + CA_);
		jc_  = CM_ * jc_ + CA_;
	}

	~BaseGen() {;}

	BaseGen& operator=(const BaseGen& gen)
	{
		jc_  = gen();
		z_   = (jc_ = CM_ * jc_ + CA_);
		w_   = (jc_ = CM_ * jc_ + CA_);
		jsr_ = (jc_ = CM_ * jc_ + CA_);
		jc_  = CM_ * jc_ + CA_;
		return *this;
	}
	
	UINT32 operator()() const
	{
		z_    = MWC1_ * (z_ & MASK_) + (z_ >> 16);
		w_    = MWC2_ * (w_ & MASK_) + (w_ >> 16);
		jsr_ ^= (jsr_ << 17); jsr_ ^= (jsr_ >> 13); jsr_ ^= (jsr_ << 5);
		jc_   = CM_ * jc_ + CA_;
		return (((z_ << 16) + (w_ & MASK_)) ^ jc_) + jsr_;
	}
};


class IntGen
// Generate uniformly distributed integers from a closed interval.
{
protected:
	BaseGen gen_;
	INT32   low_, delta_, shift_;
	mutable UINT32 bit_, word_;
	UINT32  mask_;

public:
	explicit IntGen(UINT32 seed = 0, int a = 0, int b = 0) : gen_(seed)
	{
		if (a == b)
		{
			low_  = 0;
			delta_ = 1;
		}
		else
		{
			low_  = (a < b) ? a : b;
			delta_ = (a > b) ? a : b;
		}
		delta_ -= low_;
		shift_  = 0;
		mask_   = 1;
		while (INT32(mask_) <= delta_)
		{
			++shift_;
			mask_ <<= 1;
		}
		bit_  = 0;
		word_ = gen_();
		--mask_;
	}

	IntGen(const IntGen& uni) : gen_(uni.gen_), low_(uni.low_), delta_(uni.delta_),
		shift_(uni.shift_), mask_(uni.mask_)
	{
		bit_  = 0;
		word_ = gen_();
	}

	~IntGen() {;}

	int operator()() const
	{
		UINT32 val;
		do {
			val = word_ & mask_;
			word_ >>= shift_;


			bit_   += shift_;
			if (bit_ > 31)
			{
				bit_  = 0;
				word_ = gen_();
			}
		} while (INT32(val) > delta_);
		return low_ + val;
	}

	IntGen& operator=(const IntGen& uni)
	{
		low_   = uni.low_;
		delta_ = uni.delta_;
		gen_   = uni.gen_;
		shift_ = uni.shift_;
		bit_   = 0;
		word_  = gen_();
		mask_  = uni.mask_;
		return *this;
	}

	void set_range(int a = 0, int b = 0)
	{
		if (a == b)
		{
			low_   = 0;
			delta_ = 1;
		}
		else
		{
			low_   = (a < b) ? a : b;
			delta_ = (a > b) ? a : b;
		}
		delta_ -= low_;
		shift_  = 0;
		mask_   = 1;
		while (INT32(mask_) <= delta_)
		{
			++shift_;
			mask_ <<= 1;
		}
		bit_  = 0;
		word_ = gen_();
		--mask_;
		return;
	}

	int low() const
	{
		return low_;
	}

	int high() const
	{
		return low_ + delta_;
	}
};


class UniformGen
// Generate uniformly distributed doubles from an open interval.
{
protected:
	BaseGen gen_;
	double  mean_, delta_;
  
public:
  
	explicit UniformGen(UINT32 seed = 0, double a = 0, double b = 0) : gen_(seed)
	{
		if (a == b)
		{
			mean_  = 0.5;
			delta_ = 0.5;
		}
		else
		{
			mean_  = (b + a) / 2.0;
			delta_ = fabs(b - a) / 2.0;
		}
		delta_ *= 4.656613e-10;
	}

	UniformGen(const UniformGen& uni) : gen_(uni.gen_), mean_(uni.mean_), delta_(uni.delta_) {;}
	
	~UniformGen() {;}

	double operator()() const
	{
		return mean_ + static_cast<INT32>(gen_()) * delta_;
	}

	UniformGen& operator=(const UniformGen& uni)
	{
		mean_  = uni.mean_;
		delta_ = uni.delta_;
		gen_   = uni.gen_;
		return *this;
	}

	void set_range(double a = 0, double b = 0)
	{
		if (a == b)
		{
			mean_  = 0.5;
			delta_ = 0.5;
		}
		else
		{
			mean_  = (b + a) / 2.0;
			delta_ = fabs(b - a) / 2.0;
		}
		delta_ *= 4.656613e-10;
	}

	double low() const
	{
		return mean_ - delta_;
	}

	double high() const
	{
		return mean_ + delta_;
	}
};


class NormalGen
// Generate normally distributed random numbers, using the Monty Python method.
{
protected:
	BaseGen gen_;
	double  mean_, stdv_;
  

public:
  
  
	explicit NormalGen(UINT32 seed = 0, double m = 0, double v = 1) : gen_(seed),
		mean_(m), stdv_(v)
	{
		if (stdv_ < 0) stdv_ = - stdv_;
	}

	NormalGen(const NormalGen& nrm) : gen_(nrm.gen_), mean_(nrm.mean_),
		stdv_(nrm.stdv_) {;}

	~NormalGen() {;}

	double operator()() const
	{
		double x, y, v;
		x = static_cast<INT32>(gen_()) * 1.167239e-9;
		if (fabs(x) < 1.17741) return (mean_ + stdv_ * x);
		y = gen_() * 2.328306e-10;
		if (log(y) < (0.6931472 - 0.5 * x * x)) return (mean_ + stdv_ * x);
		x = (x > 0) ? (0.8857913 * (2.506628 - x)) :
			(-0.8857913 * (2.506628 + x));
		if (log(1.8857913 - y) < (0.5718733 - 0.5 * x * x))
			return (mean_ + stdv_ * x);
		do
		{
			v = static_cast<INT32>(gen_()) * 4.656613e-10;
			x = -log(fabs(v)) * 0.3989423;
			y = -log(gen_() * 2.328306e-10);
		}
		while (y + y < x * x);
		return ((v > 0) ? (mean_ + stdv_ * (2.506628 + x)) :
			(mean_ + stdv_ * (-2.506628 - x)));
	}

	NormalGen& operator=(const NormalGen& nrm)
	{
		mean_ = nrm.mean_;
		stdv_ = nrm.stdv_;
		gen_  = nrm.gen_;
		return *this;
	}

	void set_range(double m = 0, double v = 1.0)
	{
		mean_ = m;
		stdv_ = v;
		if (stdv_ < 0) stdv_ = - stdv_;
	}

	double mean() const
	{
		return mean_;
	}

	double stdv() const
	{
		return stdv_;
	}

 
};

struct Generators
{
  static UniformGen randomNumberUniGenerator;
  static NormalGen randomNumberNormGenerator; 
};


double getRandomNumberNormal();
double getRandomNumberUniform();
double getRandomNumberPsuedoNormal();

};

#endif

  
