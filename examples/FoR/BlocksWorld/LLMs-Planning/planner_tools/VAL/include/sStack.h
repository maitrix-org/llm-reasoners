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


/* 
  sStack.h

  Simple Stack.  

  $Date: 2009-02-05 10:50:27 $
  $Revision: 1.2 $

  This is an STL deque with a stack-like interface added.  This is an
  insecure stack with all the deque interface deliberately left
  available.  

*/

#ifndef SSTACK_H
#define SSTACK_H

#include <deque>

using std::deque;

namespace VAL {

template <class T>
class sStack : public deque<T>
{
private:
    typedef deque<T> _Base;
public:

    // push elem onto stack
    void push(const T& elem) 
	{
	    _Base::push_front(elem);
	};

     // pop elem from stack and return it
    T pop() 
	{
	    T elem(_Base::front());
	    _Base::pop_front();
	    return elem;
	};

    // return top element, leaving it on the stack
    T& top() 
	{
	    return _Base::front();
	};
};

};

#endif /* SSTACK_H */
