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
  Class definitions for PDDL2.1 parse trees

  $Date: 2009-02-11 17:20:38 $
  $Revision: 1.4 $

  stephen.cresswell@cis.strath.ac.uk

  Strathclyde Planning Group
  http://planning.cis.strath.ac.uk/
 ----------------------------------------------------------------------------


  In general, data members are pointers to objects allocated using
  new.  Yacc (bison) is not C++-tolerant enough to allow object
  instances to be returned as semantic values, so it is necessary in
  general to return pointers instead.

  Deleting any parse_category class should automatically delete all
  contained structures.  Symbols are an exception to this, as a
  symbol is always owned by a symbol table.
 ----------------------------------------------------------------------------*/

#ifndef PTREE_H
#define PTREE_H

#include <list>
#include <memory>
#include <map>
#include <string>
#include "sStack.h"
#include "macros.h"
#include "parse_error.h"
#include <iostream>


/*-----------------------------------------------------------------------------
  Forward declaration of classes,
  (because in some cases we have mutually referring structures).
 ----------------------------------------------------------------------------*/

using std::list;
using std::map;
using std::cout;
using std::string;
using std::auto_ptr;
using std::ostream;

namespace VAL {

class parse_category;

 class symbol;
  class pred_symbol;
  class func_symbol;
  class pddl_typed_symbol;
   class parameter_symbol;
   class var_symbol;
   class const_symbol;
   class pddl_type;
 class operator_symbol;

 class proposition;
 class proposition_list;
 class pred_decl;
 class func_decl;
 class pred_decl_list;
 class func_decl_list;

 class expression;
  class binary_expression;
   class plus_expression;
   class minus_expression;
   class mul_expression;
   class div_expression;
  class uminus_expression;
  class num_expression;
  class int_expression;
  class float_expression;
  class func_term;
  class class_func_term;
  class special_val_expr;

 class goal_list;
 class goal;
  class simple_goal;
  class qfied_goal;
  class conj_goal;
  class disj_goal;
  class imply_goal;
  class neg_goal;
  class timed_goal;
  class comparison;

 class effect;
  class simple_effect;
  class forall_effect;
  class cond_effect;
  class timed_effect;
   class timed_initial_literal;
  class assignment;

 class structure_store;
 class operator_list;
 class derivations_list;
 class structure_def;
  class operator_;
   class action;
   class event;
   class process;
   class durative_action;
  class derivation_rule;

 class domain;

 class metric_spec;
 class length_spec;
 class problem;

 class plan_step;
 class plan;

class var_symbol_table_stack;
class analysis;

struct WriteController;
struct VisitController;
 class class_def;
 class class_symbol;

enum quantifier { E_FORALL, E_EXISTS };
enum polarity { E_NEG, E_POS };
enum assign_op { E_ASSIGN, E_INCREASE, E_DECREASE, E_SCALE_UP, E_SCALE_DOWN, E_ASSIGN_CTS};
enum comparison_op { E_GREATER, E_GREATEQ, E_LESS, E_LESSEQ, E_EQUALS };
enum optimization { E_MINIMIZE, E_MAXIMIZE };
enum time_spec { E_AT_START, E_AT_END, E_OVER_ALL, E_CONTINUOUS, E_AT };
enum special_val { E_HASHT, E_DURATION_VAR, E_TOTAL_TIME };
enum length_mode { E_SERIAL, E_PARALLEL , E_BOTH};
enum constraint_sort {E_ATEND,E_ALWAYS,E_SOMETIME,E_WITHIN,E_ATMOSTONCE,E_SOMETIMEAFTER,
						E_SOMETIMEBEFORE,E_ALWAYSWITHIN,E_HOLDDURING,E_HOLDAFTER};
template <class symbol_class> class typed_symbol_list;

/*---------------------------------------------------------------------------
  PDDL requirements flags
  -------------------------------------------------------------------------*/

typedef unsigned long pddl_req_flag;


// When changing these, also look at the function pddl_req_attribute_name()
enum pddl_req_attr { E_EQUALITY              =    1,
		     E_STRIPS                =    2,
		     E_TYPING                =    4,
		     E_DISJUNCTIVE_PRECONDS  =    8,
		     E_EXT_PRECS             =   16,
		     E_UNIV_PRECS            =   32,
		     E_COND_EFFS             =   64,
		     E_NFLUENTS               =  128,
		     E_DURATIVE_ACTIONS      =  256,
		     E_TIME                  =  512,    // Obsolete?
		     E_DURATION_INEQUALITIES = 1024,
		     E_CONTINUOUS_EFFECTS    = 2048,
		     E_NEGATIVE_PRECONDITIONS= 4096,
		     E_DERIVED_PREDICATES    = 8192,
		     E_TIMED_INITIAL_LITERALS= 16384,
		     E_PREFERENCES           = 32768,
		     E_CONSTRAINTS           = 65536,
		     E_OFLUENTS              = 131072,
		     E_ACTIONCOSTS           = 262144,
		     E_MODULES               = 524288
// Attributes which are defined as combinations of others
// are expanded by parser, and don't need to be included here.
};

/*---------------------------------------------------------------------------
  Functions relating to error handling
  ---------------------------------------------------------------------------*/

void requires(pddl_req_flag);
extern bool types_defined;
extern bool types_used;
//string pddl_req_flags_string(pddl_req_flag flags);
string pddl_req_flags_string(pddl_req_flag flags);
void log_error(error_severity sev, const string& description);



/*----------------------------------------------------------------------------
  --------------------------------------------------------------------------*/

extern parse_category* top_thing;
extern analysis* current_analysis;

/*---------------------------------------------------------------------------*
  ---------------------------------------------------------------------------*/

class parse_category
{
protected:
	static auto_ptr<WriteController> wcntr;
public:
    parse_category() {};
    virtual ~parse_category() {};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const {};
    virtual void visit(VisitController * v) const {};
    static void setWriteController(auto_ptr<WriteController> w);
    static WriteController * recoverWriteController();
};

ostream & operator <<(ostream  & o,const parse_category & p);


/*---------------------------------------------------------------------------*
  Specialisation of list template.
  This is used as a list of pointers to parse category entities.
  ---------------------------------------------------------------------------*/

template<class pc>
class pc_list : public list<pc>, public parse_category
{
private:
	typedef list<pc> _Base;
public:
    virtual ~pc_list();
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
};

/*-----------------------------------------------------------------------------
  Lists of parse category things
  ---------------------------------------------------------------------------*/



template<class pc>
void pc_list<pc>::display(int ind) const
{
    for (typename pc_list<pc >::const_iterator i=_Base::begin(); i!=_Base::end(); ++i)
	ELT(*i);
};

template<class pc>
void pc_list<pc>::write(ostream & o) const
{
	for (typename pc_list<pc >::const_iterator i=_Base::begin(); i!=_Base::end(); ++i)
	{
		(*i)->write(o);
	};
};

template<class pc>
void pc_list<pc>::visit(VisitController * v) const
{
	for (typename pc_list<pc >::const_iterator i=_Base::begin(); i!=_Base::end(); ++i)
	{
		(*i)->visit(v);
	};
};

/*---------------------------------------------------------------------------*
  Symbol tables
  We have various ways of looking up/adding symbols, depending on whether
  we expect symbol to be already present in the table.
 *---------------------------------------------------------------------------*/


template<class T>
struct SymbolFactory {
	virtual T * build(const string & name) {return new T(name);};
	virtual ~SymbolFactory() {};
};

template<class T,class U>
struct SpecialistSymbolFactory : public SymbolFactory<T> {
	T * build(const string & name) {return new U(name);};
};

template<class symbol_class>
class symbol_table : public map<string,symbol_class*>
{
private:
	typedef map<string,symbol_class*> _Base;
	auto_ptr<SymbolFactory<symbol_class> > factory;

public :

	symbol_table() : factory(new SymbolFactory<symbol_class>()) {};

	void setFactory(SymbolFactory<symbol_class> * sf)
	{
		auto_ptr<SymbolFactory<symbol_class> > x(sf);
		factory = x;
	};

	template<class T>
	void replaceFactory()
	{
		auto_ptr<SymbolFactory<symbol_class> > x(new SpecialistSymbolFactory<symbol_class,T>());
		factory = x;
	};

    typedef typename _Base::iterator iterator;

 	typedef typename _Base::const_iterator const_iterator;

    // symbol_ref(string)
    // Don't care whether symbol is already present
    symbol_class* symbol_ref(const string& name)
	{
	    iterator i= _Base::find(name);
	    //symbol_table::iterator i= find(name);

	    // If name is already in symbol table
	    if (i != _Base::end())
	    {
		// Return existing symbol entry
		return i->second;
	    }
	    else
	    {
		// Create new symbol for name and add to table
		symbol_class* sym= factory->build(name);

		this->insert(std::make_pair(name,sym));
		return sym;
	    }
	};

    // Look up symbol, returning NULL pointer if not already present.
    //  (so callers must check the result!).
    symbol_class* symbol_probe(const string& name)
 	{
	    iterator i= _Base::find(name);
 	    //symbol_table::iterator i= find(name);

	    // If name is already in symbol table
	    if (i != _Base::end())
	    {
		// Return existing symbol entry
 		return i->second;
	    }
	    else
 	    {
		// Otherwise return null pointer
 		return NULL;
 	    }
	};

    // Look up symbol, requiring that symbol is already present
    symbol_class* symbol_get(const string& name)
 	{
	    iterator i= _Base::find(name);
 	    //symbol_table::iterator i= find(name);

 	    // If name is already in symbol table
 	    if (i != _Base::end())
	    {
		// Return found symbol
		return i->second;
	    }
	    else
 	    {
		// Log an error, then add symbol to table anyway.
		log_error( E_WARNING,
			   "Undeclared symbol: " + name );
		symbol_class* sym= factory->build(name);
		this->insert(std::make_pair(name,sym));

		return(sym);
 	    }
	};

    // Add symbol to table, requiring that symbol is not already present
    symbol_class* symbol_put(const string& name)
 	{
	    iterator i= _Base::find(name);
 	    //symbol_table::iterator i= find(name);

 	    // If name is already in symbol table
 	    if (i != _Base::end())
	    {
 		// Log an error
		log_error( E_WARNING,
			   "Re-declaration of symbol in same scope: " + name);
		return i->second;
	    }
	    else
 	    {
		// add new symbol
		symbol_class* sym= factory->build(name);
		this->insert(std::make_pair(name,sym));

		return(sym);
 	    }
	};

    symbol_class* new_symbol_put(const string& name)
 	{
	    iterator i= _Base::find(name);
 	    //symbol_table::iterator i= find(name);

 	    // If name is already in symbol table
 	    if (i != _Base::end())
	    {
 		// Log an error
		log_error( E_FATAL,
			   "Re-declaration of symbol in same scope: " + name);
		return i->second;
	    }
	    else
 	    {
		// add new symbol
		symbol_class* sym= factory->build(name);
		this->insert(std::make_pair(name,sym));

		return(sym);
 	    }
	};

    virtual void display(int ind) const
	{
	    TITLE(symbol_table);
	    //for (symbol_table::iterator i=begin(); i!=end(); ++i)
	    for (const_iterator i=_Base::begin(); i!=_Base::end(); ++i)
	    {
		LEAF(i->first);
		FIELD(i->second);
	    }
	};

    virtual ~symbol_table()
	{
//	    for(symbol_table::iterator i= begin(); i!=end(); ++i)
	    for(iterator i= _Base::begin(); i!=_Base::end(); ++i)
		delete i->second;
	};
};


// Refinements of symbol tables
typedef symbol_table<var_symbol>   var_symbol_table;
typedef symbol_table<const_symbol> const_symbol_table;
typedef symbol_table<pddl_type>    pddl_type_symbol_table;
typedef symbol_table<pred_symbol>  pred_symbol_table;
typedef symbol_table<func_symbol>  func_symbol_table;
typedef symbol_table<operator_symbol>  operator_symbol_table;
typedef symbol_table<class_symbol> class_symbol_table;

/*-----------------------------------------------------------------------------
  Lists of symbols
  ---------------------------------------------------------------------------*/

// No destructor for symbol lists
// - destroying the symbols themselves is responsibility of a symbol_table
// - hence we don't make it a pc_list.
template <class symbol_class>
class typed_symbol_list : public list<symbol_class*>, public parse_category
{
private:
	typedef list<symbol_class*> _Base;
public:
    typedef typename _Base::iterator iterator;
	typedef typename _Base::const_iterator const_iterator;

    void set_types(pddl_type* t)
	{
	    //for (typed_symbol_list::iterator i= begin(); i!=end(); ++i)
	    for (iterator i= _Base::begin(); i!=_Base::end(); ++i)
	    {
	    	if((*i)->type)
	    	{
	    		(*i)->either_types = new typed_symbol_list<pddl_type>;
	    		(*i)->either_types->push_back((*i)->type);
	    		(*i)->either_types->push_back(t);
	    		(*i)->type = 0;
	    		continue;
	    	};
	    	if((*i)->either_types)
	    	{
	    		(*i)->either_types->push_back(t);
	    		continue;
	    	};
	    	(*i)->type = t;
	    };
	};

    void set_either_types(typed_symbol_list<pddl_type>* tl)
 	{
 	    //for (typed_symbol_list::iterator i= begin(); i!=end(); ++i)
 	    iterator i= _Base::begin();
 	    if(i == _Base::end())
 	    {
 	    	return;
 	    };
 	    (*i)->either_types = tl;
 	    ++i;
 	    for (; i!=_Base::end(); ++i)
 	    {
 			(*i)->either_types= new typed_symbol_list<pddl_type>(*tl);
 		}
 	};

    virtual void display(int ind) const
	{
	    TITLE(typed_symbol_list<>);
	    //for (typed_symbol_list::iterator i= begin(); i!=end(); ++i)
	    for (const_iterator i= _Base::begin(); i!=_Base::end(); ++i)
		ELT(*i);
	};

	virtual void write(ostream & o) const
	{
		for (typename list<symbol_class*>::const_iterator i=_Base::begin(); i!=_Base::end(); ++i)
		{
			o << " ";
			(*i)->symbol_class::write(o);
		};
	};

	virtual void visit(VisitController * v) const
	{
		for (typename list<symbol_class*>::const_iterator i=_Base::begin(); i!=_Base::end(); ++i)
		{
			(*i)->visit(v);
		};
	};

    virtual ~typed_symbol_list() {};
};


//class var_symbol_list : public typed_symbol_list<var_symbol> {};
typedef typed_symbol_list<var_symbol> var_symbol_list;
typedef typed_symbol_list<parameter_symbol> parameter_symbol_list;
typedef typed_symbol_list<const_symbol> const_symbol_list;
typedef typed_symbol_list<pddl_type> pddl_type_list;



/*----------------------------------------------------------------------------
  Symbols
   used for constants, variables, types, and predicate names.
   Generally, a pointer to a symbol will be used as a unique identifier.
  --------------------------------------------------------------------------*/

class symbol : public parse_category
{
protected:
    string name;
public:
    symbol() {};
    symbol(const string& s) : name(s) {};

    virtual ~symbol() {};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;

    const string getName() const {return name;};
};

class pred_symbol : public symbol
{
public:
    pred_symbol(const string& s) : symbol(s) {};
    virtual ~pred_symbol() {};
    virtual void visit(VisitController * v) const;
};

class func_symbol : public symbol
{
public:
    func_symbol(const string& s) : symbol(s) {};
    virtual ~func_symbol() {};
    virtual void visit(VisitController * v) const;
};

// Variables, constants or types.
class pddl_typed_symbol : public symbol
{
public:
    pddl_type* type;               // parent type
    pddl_type_list* either_types;  // types declared with 'either'

    pddl_typed_symbol() : symbol(""), type(NULL), either_types(NULL) {};
    pddl_typed_symbol(const string& s) : symbol(s), type(NULL), either_types(NULL) {};

    virtual ~pddl_typed_symbol()
	{
	    delete either_types;
	};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
};

// Parameters can be variables or constant symbols
class parameter_symbol : public pddl_typed_symbol
{
public:
    parameter_symbol(const string& s) : pddl_typed_symbol(s) {};
    virtual ~parameter_symbol() {};
};


class var_symbol   : public parameter_symbol
{
public:
    var_symbol(const string& s) : parameter_symbol(s) {};
    virtual ~var_symbol() {};

    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
};


class const_symbol : public parameter_symbol
{
public:
    const_symbol(const string& s) : parameter_symbol(s) {};
    virtual ~const_symbol() {};

    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
};

class class_symbol : public symbol
{
public:
    class_symbol(const string& s) : symbol(s) {};
    virtual ~class_symbol() {};

    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
};

// PDDL types

class pddl_type : public pddl_typed_symbol
{
public:
    pddl_type(const string& s) : pddl_typed_symbol(s) {};
    virtual ~pddl_type() {};
};

class operator_symbol : public symbol
{
public:
    operator_symbol(const string& s) : symbol(s) {};
    // probably need to also refer to operator itself to enable
    // lookup of operator by name
    virtual ~operator_symbol() {};
};

/*---------------------------------------------------------------------------*
  Proposition
 *---------------------------------------------------------------------------*/

class proposition : public parse_category
{
public:
    pred_symbol* head;
    parameter_symbol_list* args;

    proposition(pred_symbol* h, parameter_symbol_list* a) :
	head(h), args(a) {};

	proposition(pred_symbol* h, var_symbol_list* a) :
	head(h), args(new parameter_symbol_list)
	{
		for(var_symbol_list::iterator i = a->begin();i != a->end();++i)
		{
			args->push_back(*i);
		};
	};

    virtual ~proposition()
	{
	    // don't delete head - it belongs to a symbol table
	    delete args;
	};

    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
};



class proposition_list : public pc_list<proposition*> {};

// Nearly the same as a proposition, but:
//    The arguments must be variables.
//    These variables are local to the declaration,
//     so the pred_decl class has its own symbol table.

class pred_decl : public parse_category
{
protected:
    pred_symbol* head;
    var_symbol_list* args;
    var_symbol_table* var_tab;

public:
    pred_decl(pred_symbol* h,
//	      typed_symbol_list<var_symbol>* a,
	      var_symbol_list* a,
	      var_symbol_table* vt) :
	head(h), args(a), var_tab(vt) {};

	const pred_symbol * getPred() const {return head;};
    const var_symbol_list * getArgs() const {return args;};

    void setTypes(proposition * p) const
    {
    	var_symbol_list::iterator j = args->begin();
		for(parameter_symbol_list::iterator i = p->args->begin();i != p->args->end();++i,++j)
		{
			(*i)->type = (*j)->type;
			(*i)->either_types = (*j)->either_types;
		};
    };

    virtual ~pred_decl() { delete args; delete var_tab; };
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
};



class func_decl : public parse_category
{
private:
    func_symbol* head;
    var_symbol_list* args;
    var_symbol_table* var_tab;


public:
    func_decl(func_symbol* h,
//	      typed_symbol_list<var_symbol>* a,
	      var_symbol_list* a,
	      var_symbol_table* vt) :
      head(h), args(a), var_tab(vt) {};

	const func_symbol * getFunction() const {return head;};
    const var_symbol_list * getArgs() const {return args;};

    virtual ~func_decl() { delete args; delete var_tab; };
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};


class pred_decl_list : public pc_list<pred_decl*>
{
public:
    virtual ~pred_decl_list() {};
    virtual void write(ostream & o) const;
};

class func_decl_list : public pc_list<func_decl*>
{
public:
    virtual ~func_decl_list() {};
    virtual void write(ostream & o) const;
};

 class classes_list : public pc_list<class_def*>
 {
 public:
   virtual ~classes_list() {};
   virtual void write(ostream & o) const;
 };



/*----------------------------------------------------------------------------
  Expressions
  --------------------------------------------------------------------------*/

class expression : public parse_category
{
};

class binary_expression : public expression {
protected:
	expression * arg1;
	expression * arg2;
public:
	binary_expression(expression * a1,expression * a2) :
		arg1(a1), arg2(a2)
	{};
	virtual ~binary_expression()
	{
		delete arg1;
		delete arg2;
	};
	const expression * getLHS() const {return arg1;};
	const expression * getRHS() const {return arg2;};
};

class plus_expression : public binary_expression
{
public:
    plus_expression(expression *a1, expression *a2) :
	binary_expression(a1,a2) {};

    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
};

class minus_expression : public binary_expression
{
public:
    minus_expression(expression *a1, expression *a2) :
	binary_expression(a1,a2) {};

    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
};

class mul_expression : public binary_expression
{
public:
    mul_expression(expression *a1, expression *a2) :
	binary_expression(a1,a2) {};

    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
};

class div_expression : public binary_expression
{
public:
    div_expression(expression *a1, expression *a2) :
	binary_expression(a1,a2) {};

    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
};

class uminus_expression : public expression
{
private:
    expression *arg1;
public:
    uminus_expression(expression *a1) :
	arg1(a1) {};
    virtual ~uminus_expression()
	{ delete arg1;};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;

    const expression * getExpr() const {return arg1;};
};

typedef long double NumScalar;

class num_expression : public expression {
public:
	virtual ~num_expression() {};
	virtual const NumScalar double_value() const = 0;
	};

class int_expression : public num_expression
{
private:
    int val;
public:
    int_expression(int v) : val(v) {};
    virtual ~int_expression() {};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;

    const NumScalar double_value() const {return static_cast<const NumScalar>(val);};
};

class float_expression : public num_expression
{
private:
    NumScalar val;
public:
    float_expression(NumScalar v) : val(v) {};
    virtual ~float_expression() {};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
    const NumScalar double_value() const {return static_cast<const NumScalar>(val);};
};

class func_term : public expression
{
protected:
    func_symbol *func_sym;
    parameter_symbol_list *param_list;
public:
    func_term(func_symbol *fs, parameter_symbol_list *psl) :
	func_sym(fs), param_list(psl) {};
    virtual ~func_term() {delete param_list;};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;

    const func_symbol * getFunction() const {return func_sym;};
    const parameter_symbol_list * getArgs() const {return param_list;};
};

class class_func_term : public func_term
{
private:
    class_symbol *csym;

public:
 class_func_term(class_symbol * cs,func_symbol *fs, parameter_symbol_list *psl) :
    func_term(fs,psl), csym(cs) {};
    virtual ~class_func_term() {};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;

    const class_symbol * getClass() const {return csym;};
};

// FIX: this is the duration var
// This class for special values hasht and ?duration
// Not sure what should be done with these.
class special_val_expr : public expression
{
private:
    const special_val var;

public:
    special_val_expr(special_val v) : var(v) {};
    virtual ~special_val_expr() {};
    const special_val getKind() const {return var;};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
};

class violation_term : public expression
{
private:
	const string name;
public:
	violation_term(const char * n) : name(n) {};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;
    const string getName() const {return name;};
};

// [ end of expression classes ]

/*---------------------------------------------------------------------------
  Goals
  ---------------------------------------------------------------------------*/


class goal_list: public pc_list<goal*>
{
public:
    virtual ~goal_list() {};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
};

class goal : public parse_category {
public:
	virtual void display(int ind) const;
	virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};

class con_goal : public goal {};

class constraint_goal : public con_goal
{
private:
	constraint_sort cons;
	goal * requirement;
	goal * trigger;
	double deadline;
	double from;
public:
	constraint_goal(constraint_sort c,goal * g) : cons(c), requirement(g),
		trigger(0),deadline(0),from(0)
	{};
	constraint_goal(constraint_sort c,goal * req,goal * tri) : cons(c),
		requirement(req), trigger(tri), deadline(0), from(0)
	{};
	constraint_goal(constraint_sort c,goal * req,goal * tri,double d,double f) :
		cons(c), requirement(req), trigger(tri), deadline(d), from(f)
	{};
	virtual void display(int ind) const;
	virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;

	constraint_sort getCons() const {return cons;};
	goal * getTrigger() const {return trigger;};
	goal * getRequirement() const {return requirement;};
	double getDeadline() const {return deadline;};
	double getFrom() const {return from;};
};

class preference : public con_goal
{
private:
	string name;
	goal * gl;
public:
	preference(const char * nm,goal * g) : name(nm), gl(g) {};
	preference(goal * g) : name("anonymous"), gl(g) {};

	const string & getName() const {return name;};
	goal * getGoal() const {return gl;};
	virtual void display(int ind) const;
	virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};

class simple_goal : public goal
{
private:
    polarity plrty;    // +ve or -ve goals
    proposition* prop;

public:
    simple_goal(proposition* prp, polarity pol) : plrty(pol), prop(prp) {};
    virtual ~simple_goal()
	{ delete prop; };
	virtual void display(int ind) const;
	virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
    const polarity getPolarity() const {return plrty;};
    const proposition * getProp() const {return prop;};
};

class qfied_goal : public con_goal
{
private:
    const quantifier qfier;
    var_symbol_list* vars;
    var_symbol_table* sym_tab;
    goal* gl;

public:
    qfied_goal(quantifier q, var_symbol_list* vl, goal* g, var_symbol_table* s) :
	qfier(q),
	vars(vl),
	sym_tab(s),
	gl(g)
	{};
    virtual ~qfied_goal()  { delete vars; delete sym_tab; delete gl; };
    const quantifier getQuantifier() const {return qfier;};
    const var_symbol_list* getVars() const {return vars;};
    const var_symbol_table* getSymTab() const {return sym_tab;};
    const goal * getGoal() const {return gl;};
	virtual void display(int ind) const;
	virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};


class conj_goal : public con_goal
{
private:
    goal_list* goals;
public:
    conj_goal(goal_list* gs): goals(gs) {};
    virtual ~conj_goal() { delete goals; };
    const goal_list * getGoals() const {return goals;};
	virtual void display(int ind) const;
	virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};

class disj_goal : public goal
{
private:
    goal_list* goals;
public:
    disj_goal(goal_list* gs): goals(gs) {};
    virtual ~disj_goal() { delete goals; };
    const goal_list * getGoals() const {return goals;};
    virtual void display(int ind) const;
	virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};

class imply_goal : public goal
{
private:
    goal* lhs;
    goal* rhs;

public:
    imply_goal(goal* lhs, goal* rhs) :
	lhs(lhs), rhs(rhs)
	{};
    virtual ~imply_goal()  { delete lhs; delete rhs; };
	virtual void display(int ind) const;
	virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
    const goal * getAntecedent() const {return lhs;};
    const goal * getConsequent() const {return rhs;};
};

class neg_goal : public goal
{
private:
    goal* gl;

public:
    neg_goal(goal* g) :
	gl(g) {};

    virtual ~neg_goal() { delete gl; };
	virtual void display(int ind) const;
	virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
    virtual void destroy() {gl = 0; delete this;};//do not delete gl
    const goal * getGoal() const {return gl;};
};

class timed_goal : public goal
{
private:
    goal* gl;
    time_spec ts;

public:
    timed_goal (goal* g, time_spec t) : gl(g), ts(t) {};
    virtual ~timed_goal() { delete gl; };
    goal * clearGoal()
    {
    	goal * gl1 = gl;
    	gl = 0;
    	return gl1;
    };
	virtual void display(int ind) const;
	virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
    const goal * getGoal() const {return gl;};
    time_spec getTime() const {return ts;};
};

class comparison : public goal, public binary_expression // proposition?
{
private:
    comparison_op op;

public:
    comparison(comparison_op c_op, expression* e1, expression* e2) :
	 binary_expression(e1,e2), op(c_op) {};
	virtual void display(int ind) const;
	virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
    const comparison_op getOp() const {return op;};
};


/*---------------------------------------------------------------------------*
  Effect lists
  - a single class containing a separate list of effects of each type
 *---------------------------------------------------------------------------*/

class effect_lists : public parse_category
{
public:
    pc_list<simple_effect*> add_effects;
    pc_list<simple_effect*> del_effects;
    pc_list<forall_effect*> forall_effects;
    pc_list<cond_effect*>   cond_effects;
    pc_list<cond_effect*>   cond_assign_effects;
    pc_list<assignment*>    assign_effects;
    pc_list<timed_effect*>  timed_effects;

    effect_lists() {};

    virtual ~effect_lists() {};
    void append_effects(effect_lists* from);
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;

};

/*-----------------------------------------------------------------------------
  effect classes
  ---------------------------------------------------------------------------*/

class effect : public parse_category
{
public:
    effect() {};

    virtual ~effect() {};
    virtual void display(int ind) const {};
};


class simple_effect : public effect
{
public:
    proposition* prop;

    simple_effect(proposition* eff) : effect(), prop(eff) {};
    virtual ~simple_effect() {delete prop;};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;

};

class forall_effect : public effect
{
private:
    effect_lists* operand;
    var_symbol_list * vars;
    var_symbol_table* var_tab;

public:
    forall_effect(effect_lists* eff, var_symbol_list* vs,var_symbol_table* vt) :
	effect(), operand(eff), vars(vs), var_tab(vt)
	{};

    virtual ~forall_effect()
	{
	    delete operand;
	    delete vars;
	    delete var_tab;
	};

    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;

	const var_symbol_list * getVarsList() const {return vars;};
	const var_symbol_table * getVars() const {return var_tab;};
    const effect_lists * getEffects() const {return operand;};
};


class cond_effect : public effect
{
private:
    goal* cond;
    effect_lists* effects;

public:
    // Construct from a list
    cond_effect(goal* g, effect_lists* e) :
	effect(),
	cond(g),
	effects(e)

	{};

    virtual ~cond_effect()
	{
	    delete cond;
	    delete effects;
	};

    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;

    const goal * getCondition() const {return cond;};
    const effect_lists* getEffects() const {return effects;};
};


class timed_effect : public effect
{
public:
    time_spec ts;
    effect_lists* effs;

    timed_effect(effect_lists* e, time_spec t ) : ts(t), effs(e) {};
    virtual ~timed_effect() {delete effs;};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};

class timed_initial_literal : public timed_effect
{
public:
	long double time_stamp;
  ~timed_initial_literal() {effs = 0;};
  //effs->add_effects.clear();effs->del_effects.clear();effs->assign_effects.clear();effs->timed_effects.clear();};
	timed_initial_literal(effect_lists* e,long double t) : timed_effect(e,E_AT), time_stamp(t) {};
	virtual void display(int ind) const;
	virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};

class assignment : public effect
{
private:
    func_term *f_term; // Thing to which value is assigned.
    assign_op op;      // Assignment operator, e.g.
    expression *expr;  // Value that gets assigned
public:
    assignment(func_term *ft, assign_op a_op, expression *e) :
	f_term(ft), op(a_op), expr(e) {};
    virtual ~assignment() { delete f_term; delete expr; };
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;

    const func_term * getFTerm() const {return f_term;};
    const expression* getExpr() const {return expr;};
    const assign_op getOp() const {return op;};
};


/*---------------------------------------------------------------------------
 * Structures
 * --------------------------------------------------------------------------*/

class structure_def : public parse_category
{
public:
	virtual ~structure_def() {};
	virtual void add_to(operator_list * ops,derivations_list * dvs,classes_list * cs)
	{};
};


/*---------------------------------------------------------------------------
 * Classes
 *---------------------------------------------------------------------------*/

 class class_def : public structure_def {
 public:
    const class_symbol * name;
    func_decl_list * funcs;

    class_def(const class_symbol * c,func_decl_list * fd) : name(c), funcs(fd) {};
    virtual ~class_def() {delete funcs;};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual void visit(VisitController * v) const;

    virtual void add_to(operator_list * ops,derivations_list * dvs,classes_list * cs)
    {
      cs->push_back(this);
    }
 };



/*----------------------------------------------------------------------------
  Operators
  --------------------------------------------------------------------------*/

class operator_list: public pc_list<operator_*>
{
public:
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual ~operator_list() {};
};

class operator_ : public structure_def
{
public:
    operator_symbol* name;
    var_symbol_table* symtab;
    var_symbol_list* parameters;
    goal* precondition;
    effect_lists* effects;

    operator_() : symtab(0), parameters(0), precondition(0), effects(0) {};
    operator_( operator_symbol* nm,
	       var_symbol_list* ps,
	       goal* pre,
	       effect_lists* effs,
	       var_symbol_table* st) :
	name(nm),
	symtab(st),
	parameters(ps),
	precondition(pre),
	effects(effs)

	{};
    virtual ~operator_()
	{
	    delete parameters;
	    delete precondition;
	    delete effects;
	    delete symtab;
	};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;

	virtual void add_to(operator_list * ops,derivations_list * dvs,classes_list * cs)
    {
		ops->push_back(this);
	};

};





/*-------------------------------------------------------------------------
 * Structure store
 *-------------------------------------------------------------------------*/

class derivations_list : public pc_list<derivation_rule *>
{
public:
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
    virtual ~derivations_list() {};
};

template<class pc>
pc_list<pc>::~pc_list()
{
    for (typename pc_list<pc>::iterator i=_Base::begin(); i!=_Base::end(); ++i)
	delete(*i);
};

class structure_store : public parse_category
{
private:
	operator_list * ops;
	derivations_list * dvs;
	classes_list * cs;
public:
 structure_store() : ops(new operator_list), dvs(new derivations_list), cs(new classes_list) {};

	void push_back(structure_def * s)
	{
		if(s)
		{
		  s->add_to(ops,dvs,cs);
		}
		else
		{
			log_error( E_FATAL,
			   "Unreadable structure" );
		};
	};
	operator_list * get_operators() {return ops;};
	derivations_list * get_derivations() {return dvs;};
	classes_list * get_classes() {return cs;};
};

class derivation_rule : public structure_def
{
private:
	var_symbol_table * vtab;
	proposition * head;
	goal * body;
    bool body_changed;
public:
	derivation_rule(proposition * p,goal * g,var_symbol_table * v) : vtab(v), head(p), body(g),body_changed(false) {};
	var_symbol_table* get_vars() const {return vtab;};
	proposition * get_head() const {return head;};
	goal * get_body() const {return body;};
    void set_body(goal * g) {body = g; body_changed = true; };

	virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
	virtual ~derivation_rule()
	{
		delete head;
		if(!body_changed) delete body;
	};
	virtual void display(int ind) const;
	virtual void add_to(operator_list* ops,derivations_list* drvs,classes_list * cs)
	{
		drvs->push_back(this);
	};
};

/*----------------------------------------------------------------------------
  Classes derived from operator:
    action
    event
    process
    durative_action
  --------------------------------------------------------------------------*/

class action : public operator_
{
public:
    action( operator_symbol* nm,
	    var_symbol_list* ps,
	    goal* pre,
	    effect_lists* effs,
	    var_symbol_table* st) :
	operator_(nm,ps,pre,effs,st)
	{};
    virtual ~action() {};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};

class event : public operator_
{
public:
    event( operator_symbol* nm,
	   var_symbol_list* ps,
	   goal* pre,
	   effect_lists* effs,
	   var_symbol_table* st) :
	operator_(nm,ps,pre,effs,st)
	{};
    virtual ~event() {};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};

class process : public operator_
{
public:
    process( operator_symbol* nm,
	     var_symbol_list* ps,
	     goal* pre,
	     effect_lists* effs,
	     var_symbol_table* st) :
	operator_(nm,ps,pre,effs,st)
	{};
    virtual ~process() {};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};


class durative_action : public operator_
{
public:
    goal* dur_constraint;
    durative_action() {};
    virtual ~durative_action()
	{
	    delete dur_constraint;
	};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};






/*---------------------------------------------------------------------------
  Domain.
  ---------------------------------------------------------------------------*/

class domain : public parse_category
{
public:
    operator_list* ops;
    derivations_list* drvs;
    string name;
    pddl_req_flag req;
    pddl_type_list* types;
    const_symbol_list* constants;
    var_symbol_table* pred_vars;  // Vars used in predicate declarations
    pred_decl_list* predicates;
    func_decl_list* functions;
    con_goal * constraints;
    classes_list * classes;

    domain( structure_store * ss) :
	ops(ss->get_operators()),
	drvs(ss->get_derivations()),
	req(0),
	types(NULL),
	constants(NULL),
	pred_vars(NULL),
	predicates(NULL),
	functions(NULL),
	  constraints(NULL),
	  classes(ss->get_classes())
	{
		delete ss;
	};

    virtual ~domain()
	{
		delete drvs;
	    delete ops;
	    delete types;
	    delete constants;
	    delete predicates;
	    delete functions;
	    delete pred_vars;
	    delete constraints;
	    delete classes;
	};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
    bool isDurative() const
    {
    	return req & (E_DURATIVE_ACTIONS | E_TIME);
    };
    bool isTyped() const
    {
    	return req & (E_TYPING);
    };
};

/*----------------------------------------------------------------------------
  Plan
 ----------------------------------------------------------------------------*/

class plan_step : public parse_category
{
public:
    operator_symbol* op_sym;
    const_symbol_list* params;

    bool start_time_given;
    bool duration_given;
    double start_time;
    double duration;
    double originalDuration; //for testing duration constraints when testing robustness

    plan_step(operator_symbol* o, const_symbol_list* p) :
	op_sym(o),
	params(p)
	{};

    virtual ~plan_step()
	{
	    delete params;
	};

    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};


class plan : public pc_list<plan_step*>
{private:
	double timeTaken;
public:
	plan() : timeTaken(-1) {};
    virtual ~plan() {};
    void insertTime(double t) {timeTaken = t;};
    double getTime() const {return timeTaken;};
};

/*----------------------------------------------------------------------------
  PDDL+ entities
  --------------------------------------------------------------------------*/


class metric_spec : public parse_category
{
public:
    list<optimization> opt;
    pc_list<expression*> * expr;

    metric_spec(optimization o, expression* e) : opt(),
    	expr(new pc_list<expression*>()) {
    	opt.push_back(o);
    	expr->push_back(e);
    };
    virtual ~metric_spec() {
    	delete expr;
    };
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;

	void add(metric_spec * m)
	{
		opt.push_back(m->opt.front());
		expr->push_back(m->expr->front());
		m->expr->clear();
		delete m;
	}
};


class length_spec : public parse_category
{
public:
    length_mode mode;
    int lengths;
    int lengthp;

    length_spec(length_mode m, int l) : mode(m), lengths(l), lengthp(l) {};
    length_spec(length_mode m,int ls,int lp) : mode(m), lengths(ls), lengthp(lp) {};
    virtual ~length_spec() {};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};


/*----------------------------------------------------------------------------
  Problem definition
  --------------------------------------------------------------------------*/

class problem : public parse_category
{

public:
	char * name;
	char * domain_name;
    pddl_req_flag req;
    pddl_type_list* types;
    const_symbol_list* objects;
    effect_lists* initial_state;
    goal* the_goal;
    con_goal *constraints;
    metric_spec* metric;
    length_spec* length;

    problem() :
    name(0),
    domain_name(0),
	req(0),
	types(NULL),
	objects(NULL),
	initial_state(NULL),
	the_goal(NULL),
	constraints(NULL),
	metric(NULL),
	length(NULL)
	{};

    virtual ~problem()
	{
		delete [] name;
		delete [] domain_name;
	    delete types;
	    delete objects;
	    delete initial_state;
	    delete the_goal;
	    delete constraints;
	    delete metric;
	    delete length;
	};
    virtual void display(int ind) const;
    virtual void write(ostream & o) const;
	virtual void visit(VisitController * v) const;
};



/*----------------------------------------------------------------------------
  We need to be able to search back through the tables in the stack to
  find a reference to a particular symbol.  The standard STL stack
  only allows access to top.

  The symbol_ref() member function does this, making use of its access
  to the iterator for the stack.
  --------------------------------------------------------------------------*/

class var_symbol_table_stack : public sStack<var_symbol_table*>
{
public:
    var_symbol* symbol_get(const string& name);
    var_symbol* symbol_put(const string& name);
    var_symbol* new_symbol_put(const string& name);
    ~var_symbol_table_stack()
    {
    	for(deque<var_symbol_table*>::const_iterator i = begin();
    			i != end();++i)
    		delete (*i);
    };
};

/*---------------------------------------------------------------------------*
  Analysis.
  Here we store various symbol tables for constants, types, and predicates.
  For variables, we have a stack of symbol tables which is used during
  parsing.
  Operators and quantified constructs have their own local scope,
  and their own symbol tables.
 *---------------------------------------------------------------------------*/


class VarTabFactory {
public:
	virtual ~VarTabFactory() {};
	virtual var_symbol_table * buildPredTab() {return new var_symbol_table;};
	virtual var_symbol_table * buildFuncTab() {return new var_symbol_table;};
	virtual var_symbol_table * buildForallTab() {return new var_symbol_table;};
	virtual var_symbol_table * buildExistsTab() {return new var_symbol_table;};
	virtual var_symbol_table * buildRuleTab() {return new var_symbol_table;};
	virtual var_symbol_table * buildOpTab() {return new var_symbol_table;};
};

class StructureFactory {
public:
	virtual ~StructureFactory() {};
	virtual action * buildAction(operator_symbol* nm,
	    var_symbol_list* ps,
	    goal* pre,
	    effect_lists* effs,
	    var_symbol_table* st) {return new action(nm,ps,pre,effs,st);};
	virtual durative_action * buildDurativeAction() {return new durative_action;};
	virtual event * buildEvent(operator_symbol* nm,
	   var_symbol_list* ps,
	   goal* pre,
	   effect_lists* effs,
	   var_symbol_table* st) {return new event(nm,ps,pre,effs,st);};
	virtual process * buildProcess(operator_symbol* nm,
	     var_symbol_list* ps,
	     goal* pre,
	     effect_lists* effs,
	     var_symbol_table* st) {return new process(nm,ps,pre,effs,st);};
};

class analysis
{
private:
	auto_ptr<VarTabFactory> varTabFactory;
	auto_ptr<StructureFactory> strucFactory;

public:
	var_symbol_table * buildPredTab() {return varTabFactory->buildPredTab();};
	var_symbol_table * buildFuncTab() {return varTabFactory->buildFuncTab();};
	var_symbol_table * buildForallTab() {return varTabFactory->buildForallTab();};
	var_symbol_table * buildExistsTab() {return varTabFactory->buildExistsTab();};
	var_symbol_table * buildRuleTab() {return varTabFactory->buildRuleTab();};
	var_symbol_table * buildOpTab() {return varTabFactory->buildOpTab();};

	durative_action * buildDurativeAction() {return strucFactory->buildDurativeAction();};
	action * buildAction(operator_symbol* nm,
	    var_symbol_list* ps,
	    goal* pre,
	    effect_lists* effs,
	    var_symbol_table* st) {return strucFactory->buildAction(nm,ps,pre,effs,st);};
	event * buildEvent(operator_symbol* nm,
	     var_symbol_list* ps,
	     goal* pre,
	     effect_lists* effs,
	     var_symbol_table* st) {return strucFactory->buildEvent(nm,ps,pre,effs,st);};
	process * buildProcess(operator_symbol* nm,
	     var_symbol_list* ps,
	     goal* pre,
	     effect_lists* effs,
	     var_symbol_table* st) {return strucFactory->buildProcess(nm,ps,pre,effs,st);};

	void setFactory(VarTabFactory * vf)
	{
		auto_ptr<VarTabFactory> x(vf);
		varTabFactory = x;
	};

	void setFactory(StructureFactory * sf)
	{
		auto_ptr<StructureFactory> x(sf);
		strucFactory = x;
	};

    var_symbol_table_stack var_tab_stack;
    const_symbol_table     const_tab;
    pddl_type_symbol_table pddl_type_tab;
    pred_symbol_table	   pred_tab;
    func_symbol_table      func_tab;
    operator_symbol_table  op_tab;
    class_symbol_table     classes_tab;
    pddl_req_flag          req;

    parse_error_list error_list;

    domain* the_domain;
    problem* the_problem;

    analysis() :
    varTabFactory(new VarTabFactory),
    strucFactory(new StructureFactory),
	the_domain(NULL),
	the_problem(NULL)
	{
	    // Push a symbol table on stack as a safety net
	    var_tab_stack.push(new var_symbol_table);
	}

    virtual ~analysis()
	{
	    delete the_domain;
	    delete the_problem;
	};
};

};

#endif /* PTREE_H */

