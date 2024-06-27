 /*
  PDDL2.1 grammar file for bison.

  $Date: 2009-02-11 17:20:39 $
  $Revision: 1.5 $

  s.n.cresswell@durham.ac.uk
  Derek Long

  Srathclyde Planning Group
  http://planning.cis.ac.uk
 */


%start mystartsymbol

%{
/*
Error reporting:
Intention is to provide error token on most bracket expressions,
so synchronisation can occur on next CLOSE_BRAC.
Hence error should be generated for innermost expression containing error.
Expressions which cause errors return a NULL values, and parser
always attempts to carry on.
This won't behave so well if CLOSE_BRAC is missing.

Naming conventions:
Generally, the names should be similar to the PDDL2.1 spec.
During development, they have also been based on older PDDL specs,
older PDDL+ and TIM parsers, and this shows in places.

All the names of fields in the semantic value type begin with t_
Corresponding categories in the grammar begin with c_
Corresponding classes have no prefix.

PDDL grammar       yacc grammar      type of corresponding semantic val.

thing+             c_things          thing_list
(thing+)           c_thing_list      thing_list

*/

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <ctype.h>

// This is now copied locally to avoid relying on installation
// of flex++.

//#include "FlexLexer.h"
//#include <FlexLexer.h>

#include "ptree.h"
#include "parse_error.h"

#define YYDEBUG 1

int yyerror(char *);

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", ((char *)msgid))
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) ((char *) msgid)
# endif
#endif

extern int yylex();

using namespace VAL;

%}

%union	{
    parse_category* t_parse_category;

    effect_lists* t_effect_lists;
    effect* t_effect;
    simple_effect* t_simple_effect;
    cond_effect*   t_cond_effect;
    forall_effect* t_forall_effect;
    timed_effect* t_timed_effect;

    quantifier t_quantifier;
    metric_spec*  t_metric;
    optimization t_optimization;

    symbol* t_symbol;
    var_symbol*   t_var_symbol;
    pddl_type*    t_type;
    pred_symbol*  t_pred_symbol;
    func_symbol*  t_func_symbol;
    const_symbol* t_const_symbol;
    class_symbol* t_class;

    parameter_symbol_list* t_parameter_symbol_list;
    var_symbol_list* t_var_symbol_list;
    const_symbol_list* t_const_symbol_list;
    pddl_type_list* t_type_list;

    proposition* t_proposition;
    pred_decl* t_pred_decl;
    pred_decl_list* t_pred_decl_list;
    func_decl* t_func_decl;
    func_decl_list* t_func_decl_list;

    goal* t_goal;
    con_goal * t_con_goal;
    goal_list* t_goal_list;

    func_term* t_func_term;
    assignment* t_assignment;
    expression* t_expression;
    num_expression* t_num_expression;
    assign_op t_assign_op;
    comparison_op t_comparison_op;

    structure_def* t_structure_def;
    structure_store* t_structure_store;

    action* t_action_def;
    event* t_event_def;
    process* t_process_def;
    durative_action* t_durative_action_def;
    derivation_rule* t_derivation_rule;

    problem* t_problem;
    length_spec* t_length_spec;

    domain* t_domain;

    pddl_req_flag t_pddl_req_flag;

    plan* t_plan;
    plan_step* t_step;

    int ival;
    double fval;

    char* cp;
    int t_dummy;

    var_symbol_table * vtab;

  class_def * t_class_def;
  //  classes_list* t_classes;

};


%type <t_effect_lists> c_effects c_conj_effect c_effect c_initial_state
      c_a_effect c_p_effects c_p_effect c_da_effect c_da_effects c_da_cts_only_effect c_da_cts_only_effects
      c_a_effect_da c_p_effect_da c_p_effects_da
      c_init_els c_proc_effect c_proc_effects
%type <t_simple_effect> c_pos_simple_effect c_neg_simple_effect
                        c_init_pos_simple_effect c_init_neg_simple_effect
%type <t_cond_effect> c_cond_effect
%type <t_forall_effect> c_forall_effect
%type <t_timed_effect> c_timed_effect c_cts_only_timed_effect c_timed_initial_literal

//%type <t_parameter_symbol> c_parameter_symbol
%type <t_type> /* c_type */ c_primitive_type c_new_primitive_type
%type <t_pred_symbol> c_pred_symbol c_new_pred_symbol c_init_pred_symbol
%type <t_func_symbol> /* c_func_symbol */ c_new_func_symbol
%type <t_const_symbol> c_const_symbol c_new_const_symbol
%type <t_var_symbol> c_var_symbol c_declaration_var_symbol

%type <t_proposition> c_proposition c_derived_proposition c_init_proposition
%type <t_pred_decl> c_pred_decl
%type <t_pred_decl_list> c_pred_decls c_predicates
%type <t_func_decl> c_func_decl
%type <t_func_decl_list> c_func_decls c_functions_def

%type <t_parameter_symbol_list> c_parameter_symbols
%type <t_var_symbol_list> c_var_symbol_list c_typed_var_list
%type <t_const_symbol_list> c_const_symbols c_new_const_symbols c_typed_consts c_domain_constants c_objects
%type <t_type_list> c_primitive_types c_new_primitive_types c_typed_types c_type_names c_either_type

%type <t_goal> c_goal_descriptor c_pre_goal_descriptor c_pref_goal_descriptor  c_goal_spec c_duration_constraint c_da_gd c_timed_gd /* c_f_comp */
%type <t_con_goal> c_constraints_def c_constraints_probdef c_constraint_goal c_pref_con_goal c_pref_goal
%type <t_goal_list> c_goal_list c_pre_goal_descriptor_list c_duration_constraints c_da_gds c_constraint_goal_list c_pref_con_goal_list
%type <t_quantifier> /*c_quantifier*/ c_forall c_exists

%type <t_func_term> c_f_head /* c_new_f_head */  c_ground_f_head
%type <t_assignment> c_assignment c_f_assign_da
%type <t_expression> c_f_exp c_ground_f_exp c_binary_ground_f_exp c_f_exp_da c_f_exp_t c_binary_expr_da c_d_value c_binary_ground_f_pexps c_binary_ground_f_mexps
%type <t_num_expression> c_number
%type <t_comparison_op> c_comparison_op c_d_op

%type <t_structure_def> c_structure_def
%type <t_class_def> c_class_def
%type <t_action_def> c_action_def
%type <t_event_def> c_event_def
%type <t_process_def> c_process_def
%type <t_durative_action_def> c_durative_action_def c_da_def_body
%type <t_derivation_rule> c_derivation_rule
%type <t_structure_store> c_structure_defs

%type <t_pddl_req_flag> c_domain_require_def c_require_key c_reqs

%type <t_problem> c_problem c_problem_body
%type <t_length_spec> c_length_spec

%type <t_class> c_class c_new_class

%type <t_domain> c_domain c_preamble
%type <t_dummy> /*c_action_kind*/ c_args_head c_rule_head c_ntype c_classes c_class_seq

%type <t_optimization> c_optimization
%type <t_metric> c_metric_spec

%type <t_plan> c_plan
%type <t_step> c_step c_step_t_d c_step_d

%type <cp> c_domain_name
%type <fval> c_float

%type <vtab> c_goals;

%token <punct> OPEN_BRAC CLOSE_BRAC MODULES
       OPEN_SQ CLOSE_SQ DOT CLASSES CLASS
       DEFINE PDDLDOMAIN REQS EQUALITY STRIPS ADL NEGATIVE_PRECONDITIONS
       TYPING DISJUNCTIVE_PRECONDS EXT_PRECS UNIV_PRECS QUANT_PRECS COND_EFFS
       FLUENTS OBJECTFLUENTS NUMERICFLUENTS ACTIONCOSTS
       TIME DURATIVE_ACTIONS DURATION_INEQUALITIES CONTINUOUS_EFFECTS
       DERIVED_PREDICATES TIMED_INITIAL_LITERALS PREFERENCES CONSTRAINTS
       ACTION PROCESS EVENT DURATIVE_ACTION DERIVED
       CONSTANTS PREDS FUNCTIONS TYPES ARGS PRE CONDITION PREFERENCE
       START_PRE END_PRE /* Redundant */
       EFFECTS
       INITIAL_EFFECT FINAL_EFFECT INVARIANT DURATION /* Redundant */
       AT_START AT_END OVER_ALL
       AND OR EXISTS FORALL IMPLY NOT WHEN WHENEVER EITHER
       PROBLEM FORDOMAIN INITIALLY
       OBJECTS GOALS EQ LENGTH SERIAL PARALLEL METRIC
       MINIMIZE MAXIMIZE
       HASHT DURATION_VAR TOTAL_TIME
       INCREASE DECREASE SCALE_UP SCALE_DOWN ASSIGN
       GREATER GREATEQ LESS LESSEQ /* EQUALS */ Q COLON NUMBER
       ALWAYS SOMETIME WITHIN ATMOSTONCE SOMETIMEAFTER SOMETIMEBEFORE
       ALWAYSWITHIN HOLDDURING HOLDAFTER ISVIOLATED
       BOGUS


%token <cp> NAME FUNCTION_SYMBOL
%token <ival> INTVAL
%token <fval> FLOATVAL AT_TIME

%left HYPHEN PLUS
%left MUL DIV
%left UMINUS

%%
mystartsymbol :
    c_domain  {top_thing= $1; current_analysis->the_domain= $1;}
|   c_problem {top_thing= $1; current_analysis->the_problem= $1;}
|   c_plan    {top_thing= $1; }
;

c_domain :
    OPEN_BRAC DEFINE c_domain_name c_preamble CLOSE_BRAC
       {$$= $4; $$->name= $3;delete [] $3;
	if (types_used && !types_defined) {
		yyerrok; log_error(E_FATAL,"Syntax error in domain - no :types section, but types used in definitions.");
	}
	}
|   OPEN_BRAC DEFINE c_domain_name error
    	{yyerrok; $$=static_cast<domain*>(NULL);
       	log_error(E_FATAL,"Syntax error in domain"); }  // Helpful?
;

// Assumes operators defns are last, and at least one of them present.
c_preamble :
     c_domain_require_def c_preamble  {$$= $2; $$->req= $1;}
   | c_type_names c_preamble          {types_defined = true; $$= $2; $$->types= $1;}
   | c_domain_constants c_preamble    {$$= $2; $$->constants= $1;}
   | c_predicates c_preamble          {$$= $2;
                                       $$->predicates= $1; }
   | c_functions_def c_preamble       {$$= $2;
                                       $$->functions= $1; }
   | c_constraints_def c_preamble     {$$= $2;
   				       $$->constraints = $1;}
   | c_classes c_preamble             {$$ = $2;}
   | c_structure_defs                 {$$= new domain($1); }
;

c_domain_name : OPEN_BRAC PDDLDOMAIN NAME CLOSE_BRAC {$$=$3;}

;

c_new_class : NAME { $$=current_analysis->classes_tab.new_symbol_put($1);
       delete [] $1; };

c_class : NAME { $$ = current_analysis->classes_tab.symbol_get($1); delete [] $1;};

c_classes : OPEN_BRAC CLASSES c_class_seq CLOSE_BRAC {$$ = 0;};

c_class_seq : c_new_class c_class_seq {$$ = 0;}|
/* empty */ {$$ = 0;}
       ;

c_domain_require_def :
   OPEN_BRAC REQS c_reqs CLOSE_BRAC
    {
	// Stash in analysis object --- we need to refer to it during parse
	//   but domain object is not created yet,
	current_analysis->req |= $3;
	$$=$3;
    }
|  OPEN_BRAC REQS error CLOSE_BRAC
      {yyerrok;
       log_error(E_FATAL,"Syntax error in requirements declaration.");
       $$= 0; }
;

c_reqs :
    c_reqs c_require_key { $$= $1 | $2; }
|   /* empty */          { $$= 0; }
;


c_pred_decls :
    c_pred_decl c_pred_decls
           {$$=$2; $$->push_front($1);}
|   c_pred_decl
        {  $$=new pred_decl_list;
           $$->push_front($1); };

c_pred_decl :
    OPEN_BRAC c_new_pred_symbol c_typed_var_list CLOSE_BRAC
       {$$= new pred_decl($2,$3,current_analysis->var_tab_stack.pop());}
|   OPEN_BRAC error CLOSE_BRAC
       {yyerrok;
        // hope someone makes this error someday
        log_error(E_FATAL,"Syntax error in predicate declaration.");
	$$= static_cast<pred_decl*>(NULL); }
;

c_new_pred_symbol :
     NAME
         { $$=current_analysis->pred_tab.new_symbol_put($1);
           current_analysis->var_tab_stack.push(
           				current_analysis->buildPredTab());
           delete [] $1; }
;

c_pred_symbol :
    EQ   { $$=current_analysis->pred_tab.symbol_ref("=");
	      requires(E_EQUALITY); }
|   NAME { $$=current_analysis->pred_tab.symbol_get($1); delete [] $1; }
;


c_init_pred_symbol :
	// We have a different pred_symbol rule for the initial state
        // so as to exclude EQ,
	// which must be parsed as assignment in initial state.
    NAME { $$=current_analysis->pred_tab.symbol_get($1); delete [] $1;}
;


c_func_decls :
    c_func_decls c_func_decl
           {$$=$1; $$->push_back($2);}
|   /* empty */  { $$=new func_decl_list; }
;

c_func_decl :
    OPEN_BRAC c_new_func_symbol c_typed_var_list CLOSE_BRAC c_ntype
       {$$= new func_decl($2,$3,current_analysis->var_tab_stack.pop());}
|   OPEN_BRAC error CLOSE_BRAC
	{yyerrok;
	 log_error(E_FATAL,"Syntax error in functor declaration.");
	 $$= (int) NULL; }
;

c_ntype :
    HYPHEN NUMBER {$$ = (int) NULL;}| /* empty */ {$$= (int) NULL;};

c_new_func_symbol :
     NAME
         { $$=current_analysis->func_tab.new_symbol_put($1);
           current_analysis->var_tab_stack.push(
           		current_analysis->buildFuncTab());
           delete [] $1; }
;

//c_func_symbol :
//     NAME { $$=current_analysis->func_tab.symbol_get($1); }
//;

// variables, possibly with types
c_typed_var_list :        /* Type specified */
   c_var_symbol_list HYPHEN c_primitive_type c_typed_var_list
   {
      $$= $1;
      $$->set_types($3);           /* Set types for variables */
      $$->splice($$->end(),*$4);   /* Join lists */
      delete $4;                   /* Delete (now empty) list */
      requires(E_TYPING);
      types_used = true;
   }
|  c_var_symbol_list HYPHEN c_either_type c_typed_var_list
   {
      $$= $1;
      $$->set_either_types($3);    /* Set types for variables */
      $$->splice($$->end(),*$4);   /* Join lists */
      delete $4;                   /* Delete (now empty) list */
      requires(E_TYPING);
      types_used = true;
   }
|  c_var_symbol_list                /* No type specified */
   {
       $$= $1;
   }
;



// a list of variables (excluding type declaration)
// Semantic value is a list of symbols

c_var_symbol_list :
    Q c_declaration_var_symbol c_var_symbol_list
     {$$=$3; $3->push_front($2); }
| /* Empty */ {$$= new var_symbol_list; }
 ;

// A list of constants (object names or types), possibly with parent types
c_typed_consts :
   /* Type specified */
   c_new_const_symbols HYPHEN c_primitive_type c_typed_consts
   {
      $$= $1;
      $1->set_types($3);           /* Set types for constants */
      $1->splice($1->end(),*$4); /* Join lists */
      delete $4;                   /* Delete (now empty) list */
      requires(E_TYPING);
      types_used = true;
   }
|  c_new_const_symbols HYPHEN c_either_type c_typed_consts
   {
      $$= $1;
      $1->set_either_types($3);
      $1->splice($1->end(),*$4);
      delete $4;
      requires(E_TYPING);
      types_used = true;
   }
| /* No type specified */
    c_new_const_symbols {$$= $1;}
;

// A list of object names without parent types
c_const_symbols :
   c_const_symbol c_const_symbols {$$=$2; $2->push_front($1);}
 | /* Empty */ {$$=new const_symbol_list;}
;

c_new_const_symbols :
   c_new_const_symbol c_new_const_symbols {$$=$2; $2->push_front($1);}
 | /* Empty */ {$$=new const_symbol_list;}
;


// As above, but for PDDL types
// possibly with parent types
c_typed_types :
   // Type specified
   c_new_primitive_types HYPHEN c_primitive_type c_typed_types
   {
       $$= $1;
       $$->set_types($3);           /* Set types for constants */
       $$->splice($$->end(),*$4); /* Join lists */
       delete $4;                   /* Delete (now empty) list */
   }
|  c_new_primitive_types HYPHEN c_either_type c_typed_types
   {
   // This parse needs to be excluded, we think (DPL&MF: 6/9/01)
       $$= $1;
       $$->set_either_types($3);
       $$->splice($1->end(),*$4);
       delete $4;
   }
|  // No parent type specified
   c_new_primitive_types
      { $$= $1; }
;

// constants or variables (excluding type declaration)
c_parameter_symbols :
    c_parameter_symbols c_const_symbol
         {$$=$1; $$->push_back($2); }
|   c_parameter_symbols Q c_var_symbol
         {$$=$1; $$->push_back($3); }
|   /* Empty */ {$$= new parameter_symbol_list;}
;


 // Used in declaration of variable
 //  - var symbol is added to var table at top of stack
c_declaration_var_symbol :
    NAME { $$= current_analysis->var_tab_stack.top()->symbol_put($1); delete [] $1; }
;

 // Used when variable is expected to have already been declared.
 // The lookup is an operation on the whole stack of variable tables.
c_var_symbol :
    NAME { $$= current_analysis->var_tab_stack.symbol_get($1); delete [] $1; }
;

c_const_symbol :
    NAME { $$= current_analysis->const_tab.symbol_get($1); delete [] $1; }
;

c_new_const_symbol :
    NAME { $$= current_analysis->const_tab.new_symbol_put($1); delete [] $1;}
;

c_either_type :
    OPEN_BRAC EITHER c_primitive_types CLOSE_BRAC
     { $$= $3; }
;

c_new_primitive_type :
    NAME
     { $$= current_analysis->pddl_type_tab.symbol_ref($1); delete [] $1;}
     // We use symbol ref here in order to support multiple declarations of
     // a type symbol - this is required for multiple inheritance.
;

c_primitive_type :
    NAME
     { $$= current_analysis->pddl_type_tab.symbol_ref($1); delete [] $1;}
;

c_new_primitive_types :
    c_new_primitive_types c_new_primitive_type
        {$$= $1; $$->push_back($2);}
|   /* empty */ {$$= new pddl_type_list;}
;

c_primitive_types :
    c_primitive_types c_primitive_type
        {$$= $1; $$->push_back($2);}
|   /* empty */ {$$= new pddl_type_list;}
;

c_init_els :
    c_init_els OPEN_BRAC EQ c_f_head c_number CLOSE_BRAC
        { $$=$1;
	  $$->assign_effects.push_back(new assignment($4,E_ASSIGN,$5));
          if($4->getFunction()->getName()=="total-cost")
          {
          	requires(E_ACTIONCOSTS);
          	// Should also check that $5 is 0...
		  }
          else
          {
          	requires(E_NFLUENTS);
          }
	}
|   c_init_els c_init_pos_simple_effect
        { $$=$1; $$->add_effects.push_back($2); }
|   c_init_els c_init_neg_simple_effect
        { $$=$1; $$->del_effects.push_back($2); }
|   c_init_els c_timed_initial_literal
		{ $$=$1; $$->timed_effects.push_back($2); }
|   /* empty */
        { $$= new effect_lists;}
;

c_timed_initial_literal :
   OPEN_BRAC AT_TIME c_init_els CLOSE_BRAC
   { requires(E_TIMED_INITIAL_LITERALS);
   		$$=new timed_initial_literal($3,$2);}
;

c_effects :
    c_a_effect    c_effects       {$$=$2; $$->append_effects($1); delete $1;}
|   c_cond_effect c_effects       {$$=$2; $$->cond_effects.push_front($1);
                                      requires(E_COND_EFFS);}
|   c_forall_effect c_effects     {$$=$2; $$->forall_effects.push_front($1);
                                      requires(E_COND_EFFS);}
|   /* nothing */                 {$$=new effect_lists(); }
;

// Parse a single effect as effect_lists
// Wasteful, but we get the benefit of categorising effect, and
// we will often need the lists when normalising the contained effects.
// e.g. conjunctive effects will immediately collapse into this structure.

c_effect :
    c_conj_effect       {$$= $1;}
|   c_pos_simple_effect {$$=new effect_lists; $$->add_effects.push_front($1);}
|   c_neg_simple_effect {$$=new effect_lists; $$->del_effects.push_front($1);}
|   c_cond_effect       {$$=new effect_lists; $$->cond_effects.push_front($1);}
|   c_forall_effect     {$$=new effect_lists; $$->forall_effects.push_front($1);}
;

c_a_effect :
    OPEN_BRAC AND c_p_effects CLOSE_BRAC {$$= $3;}
|   c_p_effect        {$$= $1;}
;

c_p_effect :
    c_neg_simple_effect
        {$$=new effect_lists; $$->del_effects.push_front($1);}
|   c_pos_simple_effect
        {$$=new effect_lists; $$->add_effects.push_front($1);}
|   c_assignment
        {$$=new effect_lists; $$->assign_effects.push_front($1);
         requires(E_NFLUENTS);}
;


c_p_effects :
    c_p_effects c_neg_simple_effect {$$= $1; $$->del_effects.push_back($2);}
|   c_p_effects c_pos_simple_effect {$$= $1; $$->add_effects.push_back($2);}
|   c_p_effects c_assignment        {$$= $1; $$->assign_effects.push_back($2);
                                     requires(E_NFLUENTS); }
|   /* empty */  { $$= new effect_lists; }
;

c_conj_effect :
    OPEN_BRAC AND c_effects CLOSE_BRAC
        { $$=$3; }
|   OPEN_BRAC AND error CLOSE_BRAC
	{yyerrok; $$=NULL;
	 log_error(E_FATAL,"Syntax error in (and ...)");
	}
;


c_da_effect :
    OPEN_BRAC AND c_da_effects CLOSE_BRAC
        { $$=$3; }
|   OPEN_BRAC c_forall
      OPEN_BRAC c_typed_var_list CLOSE_BRAC
       c_da_effect
    CLOSE_BRAC
        { $$= new effect_lists;
          $$->forall_effects.push_back(
	       new forall_effect($6, $4, current_analysis->var_tab_stack.pop()));
          requires(E_COND_EFFS);}
|   OPEN_BRAC WHEN c_da_gd c_da_effect CLOSE_BRAC
        { $$= new effect_lists;
	  $$->cond_effects.push_back(
	       new cond_effect($3,$4));
          requires(E_COND_EFFS); }
|   OPEN_BRAC WHENEVER c_goal_descriptor c_da_cts_only_effect CLOSE_BRAC
        { $$= new effect_lists;
	  $$->cond_assign_effects.push_back(
	       new cond_effect($3,$4));
          requires(E_COND_EFFS); }
|   c_timed_effect
        { $$=new effect_lists;
          $$->timed_effects.push_back($1); }
|   c_assignment
        { $$= new effect_lists;
	  $$->assign_effects.push_front($1);
          requires(E_NFLUENTS); }
;

c_da_effects :
    c_da_effects c_da_effect { $$=$1; $1->append_effects($2); delete $2; }
|   /* empty */ { $$= new effect_lists; }
;

c_timed_effect :
    OPEN_BRAC AT_START c_a_effect_da CLOSE_BRAC
        {$$=new timed_effect($3,E_AT_START);}
|   OPEN_BRAC AT_END c_a_effect_da CLOSE_BRAC
        {$$=new timed_effect($3,E_AT_END);}
|   OPEN_BRAC INCREASE c_f_head c_f_exp_t CLOSE_BRAC
        {$$=new timed_effect(new effect_lists,E_CONTINUOUS);
         $$->effs->assign_effects.push_front(
	     new assignment($3,E_INCREASE,$4)); }
|   OPEN_BRAC DECREASE c_f_head c_f_exp_t CLOSE_BRAC
        {$$=new timed_effect(new effect_lists,E_CONTINUOUS);
         $$->effs->assign_effects.push_front(
	     new assignment($3,E_DECREASE,$4)); }
|   OPEN_BRAC error CLOSE_BRAC
	{yyerrok; $$=NULL;
	log_error(E_FATAL,"Syntax error in timed effect"); }
;

c_cts_only_timed_effect :
    OPEN_BRAC INCREASE c_f_head c_f_exp_t CLOSE_BRAC
        {$$=new timed_effect(new effect_lists,E_CONTINUOUS);
         $$->effs->assign_effects.push_front(
	     new assignment($3,E_INCREASE,$4)); }
|   OPEN_BRAC DECREASE c_f_head c_f_exp_t CLOSE_BRAC
        {$$=new timed_effect(new effect_lists,E_CONTINUOUS);
         $$->effs->assign_effects.push_front(
	     new assignment($3,E_DECREASE,$4)); }
|   OPEN_BRAC error CLOSE_BRAC
	{yyerrok; $$=NULL;
	log_error(E_FATAL,"Syntax error in conditional continuous effect"); }
;

c_da_cts_only_effect :
    OPEN_BRAC AND c_da_cts_only_effects CLOSE_BRAC
        { $$=$3; }
|   OPEN_BRAC c_forall
      OPEN_BRAC c_typed_var_list CLOSE_BRAC
      c_da_cts_only_effect
    CLOSE_BRAC
        { $$= new effect_lists;
          $$->forall_effects.push_back(
	       new forall_effect($6, $4, current_analysis->var_tab_stack.pop()));
          requires(E_COND_EFFS);}
|   OPEN_BRAC WHENEVER c_goal_descriptor c_da_cts_only_effect CLOSE_BRAC
        { $$= new effect_lists;
	  $$->cond_assign_effects.push_back(
	       new cond_effect($3,$4));
          requires(E_COND_EFFS); }
|   c_cts_only_timed_effect
        { $$=new effect_lists;
          $$->timed_effects.push_back($1); }
;

c_da_cts_only_effects :
    c_da_cts_only_effects c_da_cts_only_effect { $$=$1; $1->append_effects($2); delete $2; }
|   /* empty */ { $$= new effect_lists; }
;

c_a_effect_da :
    OPEN_BRAC AND c_p_effects_da CLOSE_BRAC {$$= $3;}
|   c_p_effect_da        {$$= $1;}
;

c_p_effect_da :
    c_neg_simple_effect
        {$$=new effect_lists; $$->del_effects.push_front($1);}
|   c_pos_simple_effect
        {$$=new effect_lists; $$->add_effects.push_front($1);}
|   c_f_assign_da
        {$$=new effect_lists; $$->assign_effects.push_front($1);
         requires(E_NFLUENTS);}
;


c_p_effects_da :
    c_p_effects_da c_neg_simple_effect {$$= $1; $$->del_effects.push_back($2);}
|   c_p_effects_da c_pos_simple_effect {$$= $1; $$->add_effects.push_back($2);}
|   c_p_effects_da c_f_assign_da       {$$= $1; $$->assign_effects.push_back($2);
                                     requires(E_NFLUENTS); }
|   /* empty */  { $$= new effect_lists; }
;


c_f_assign_da :
   OPEN_BRAC ASSIGN c_f_head c_f_exp_da CLOSE_BRAC
     { $$= new assignment($3,E_ASSIGN,$4); }
|  OPEN_BRAC INCREASE c_f_head c_f_exp_da CLOSE_BRAC
     { $$= new assignment($3,E_INCREASE,$4); }
|  OPEN_BRAC DECREASE c_f_head c_f_exp_da CLOSE_BRAC
     { $$= new assignment($3,E_DECREASE,$4); }
|  OPEN_BRAC SCALE_UP c_f_head c_f_exp_da CLOSE_BRAC
     { $$= new assignment($3,E_SCALE_UP,$4); }
|  OPEN_BRAC SCALE_DOWN c_f_head c_f_exp_da CLOSE_BRAC
     { $$= new assignment($3,E_SCALE_DOWN,$4); }
;

c_proc_effect :
	OPEN_BRAC INCREASE c_f_head c_f_exp_t CLOSE_BRAC
        {$$=new effect_lists;
         timed_effect * te = new timed_effect(new effect_lists,E_CONTINUOUS);
         $$->timed_effects.push_front(te);
         te->effs->assign_effects.push_front(
	     new assignment($3,E_INCREASE,$4)); }
|   OPEN_BRAC DECREASE c_f_head c_f_exp_t CLOSE_BRAC
        {$$=new effect_lists;
         timed_effect * te = new timed_effect(new effect_lists,E_CONTINUOUS);
         $$->timed_effects.push_front(te);
         te->effs->assign_effects.push_front(
	     new assignment($3,E_DECREASE,$4)); }
| OPEN_BRAC AND c_proc_effects CLOSE_BRAC
		{$$ = $3;}
;

c_proc_effects :
  c_proc_effects c_proc_effect { $$=$1; $1->append_effects($2); delete $2; }
|   /* empty */ { $$= new effect_lists; }
;

c_f_exp_da :
    c_binary_expr_da {$$= $1;}
|   Q DURATION_VAR {$$= new special_val_expr(E_DURATION_VAR);
                    requires( E_DURATION_INEQUALITIES );}
|   c_number { $$=$1; }
|   c_f_head  { $$= $1; }
;

c_binary_expr_da :
    OPEN_BRAC PLUS c_f_exp_da c_f_exp_da CLOSE_BRAC
        { $$= new plus_expression($3,$4); }
|   OPEN_BRAC HYPHEN c_f_exp_da c_f_exp_da  CLOSE_BRAC
        { $$= new minus_expression($3,$4); }
|   OPEN_BRAC MUL c_f_exp_da c_f_exp_da CLOSE_BRAC
        { $$= new mul_expression($3,$4); }
|   OPEN_BRAC DIV c_f_exp_da c_f_exp_da	CLOSE_BRAC
        { $$= new div_expression($3,$4); }
;

c_duration_constraint :
    OPEN_BRAC AND c_duration_constraints CLOSE_BRAC
        { $$= new conj_goal($3); }
|   OPEN_BRAC c_d_op Q DURATION_VAR c_d_value CLOSE_BRAC
        { $$= new timed_goal(new comparison($2,
        			new special_val_expr(E_DURATION_VAR),$5),E_AT_START); }
|   OPEN_BRAC AT_START OPEN_BRAC c_d_op Q DURATION_VAR c_d_value CLOSE_BRAC CLOSE_BRAC
		{ $$ = new timed_goal(new comparison($4,
					new special_val_expr(E_DURATION_VAR),$7),E_AT_START);}
|   OPEN_BRAC AT_END OPEN_BRAC c_d_op Q DURATION_VAR c_d_value CLOSE_BRAC CLOSE_BRAC
		{ $$ = new timed_goal(new comparison($4,
					new special_val_expr(E_DURATION_VAR),$7),E_AT_END);}
;

c_d_op :
    LESSEQ   {$$= E_LESSEQ; requires(E_DURATION_INEQUALITIES);}
|   GREATEQ  {$$= E_GREATEQ; requires(E_DURATION_INEQUALITIES);}
|   EQ       {$$= E_EQUALS; }
;

c_d_value :
//    Fix: c_number doesn't apparently require E_FLUENTS
//         some needs to be included as separate item.
//    c_number {$$= $1;}
//|
    c_f_exp  {$$= $1; }
;

c_duration_constraints :
    c_duration_constraints c_duration_constraint
        { $$=$1; $$->push_back($2); }
|   /* empty */
        { $$= new goal_list; }
;

c_neg_simple_effect :
    OPEN_BRAC NOT c_proposition CLOSE_BRAC
     { $$= new simple_effect($3); }
;

c_pos_simple_effect :
     c_proposition
     { $$= new simple_effect($1); }
;

/* init versions disallow equality as a predicate */

c_init_neg_simple_effect :
    OPEN_BRAC NOT c_init_proposition CLOSE_BRAC
     { $$= new simple_effect($3); }
;

c_init_pos_simple_effect :
     c_init_proposition
     { $$= new simple_effect($1); }
;

c_forall_effect :
OPEN_BRAC c_forall OPEN_BRAC c_typed_var_list CLOSE_BRAC c_effect CLOSE_BRAC
     { $$= new forall_effect($6, $4, current_analysis->var_tab_stack.pop());}
;

c_cond_effect :
  OPEN_BRAC WHEN c_goal_descriptor c_effects CLOSE_BRAC
     { $$= new cond_effect($3,$4); }
;

c_assignment :
   OPEN_BRAC ASSIGN c_f_head c_f_exp CLOSE_BRAC
     { $$= new assignment($3,E_ASSIGN,$4); }
|  OPEN_BRAC INCREASE c_f_head c_f_exp CLOSE_BRAC
     { $$= new assignment($3,E_INCREASE,$4); }
|  OPEN_BRAC DECREASE c_f_head c_f_exp CLOSE_BRAC
     { $$= new assignment($3,E_DECREASE,$4); }
|  OPEN_BRAC SCALE_UP c_f_head c_f_exp CLOSE_BRAC
     { $$= new assignment($3,E_SCALE_UP,$4); }
|  OPEN_BRAC SCALE_DOWN c_f_head c_f_exp CLOSE_BRAC
     { $$= new assignment($3,E_SCALE_DOWN,$4); }
;

c_f_exp :
    OPEN_BRAC HYPHEN c_f_exp CLOSE_BRAC %prec UMINUS
        { $$= new uminus_expression($3); requires(E_NFLUENTS); }
|   OPEN_BRAC PLUS c_f_exp c_f_exp CLOSE_BRAC
        { $$= new plus_expression($3,$4); requires(E_NFLUENTS); }
|   OPEN_BRAC HYPHEN c_f_exp c_f_exp CLOSE_BRAC
        { $$= new minus_expression($3,$4); requires(E_NFLUENTS); }
|   OPEN_BRAC MUL c_f_exp c_f_exp CLOSE_BRAC
        { $$= new mul_expression($3,$4); requires(E_NFLUENTS); }
|   OPEN_BRAC DIV c_f_exp c_f_exp CLOSE_BRAC
        { $$= new div_expression($3,$4); requires(E_NFLUENTS); }
|   c_number { $$=$1; }
|   c_f_head  { $$= $1; requires(E_NFLUENTS); }
;

c_f_exp_t :
    OPEN_BRAC MUL HASHT c_f_exp CLOSE_BRAC
       { $$= new mul_expression(new special_val_expr(E_HASHT),$4); }
|   OPEN_BRAC MUL c_f_exp HASHT CLOSE_BRAC
       { $$= new mul_expression($3, new special_val_expr(E_HASHT)); }
|   HASHT
       { $$= new special_val_expr(E_HASHT); }
;


c_number :
     INTVAL   { $$=new int_expression($1);   }
|    FLOATVAL { $$=new float_expression($1); };

c_f_head :
    OPEN_BRAC FUNCTION_SYMBOL c_parameter_symbols CLOSE_BRAC
        { $$=new func_term( current_analysis->func_tab.symbol_get($2), $3); delete [] $2; }
    // "Undeclared function symbol" case
|   OPEN_BRAC NAME c_parameter_symbols CLOSE_BRAC
        { $$=new func_term( current_analysis->func_tab.symbol_get($2), $3); delete [] $2; }
|   FUNCTION_SYMBOL
        { $$=new func_term( current_analysis->func_tab.symbol_get($1),
                            new parameter_symbol_list); delete [] $1;}
| OPEN_BRAC c_class DOT FUNCTION_SYMBOL c_parameter_symbols CLOSE_BRAC
        { $$ = new class_func_term( $2, current_analysis->func_tab.symbol_get($4), $5); delete [] $4;}
;

// c_new_f_head :
//     OPEN_BRAC NAME c_parameter_symbol_list CLOSE_BRAC
//         { $$=new func_term( current_analysis->func_tab.symbol_put($2), $3); }
// |   NAME
//         { $$=new func_term( current_analysis->func_tab.symbol_put($1),
//                             new parameter_symbol_list); }
// ;

c_ground_f_head :
    /* Fix: Should restrict to constants, as in: */
    /* NAME c_const_symbols */
    /* ... but don't want to return a thing of type const list */

	OPEN_BRAC FUNCTION_SYMBOL c_parameter_symbols CLOSE_BRAC
		{ $$=new func_term( current_analysis->func_tab.symbol_get($2), $3); delete [] $2; }
|   OPEN_BRAC NAME c_parameter_symbols CLOSE_BRAC
        { $$=new func_term( current_analysis->func_tab.symbol_get($2), $3); delete [] $2; }
|   FUNCTION_SYMBOL
        { $$=new func_term( current_analysis->func_tab.symbol_get($1),
                            new parameter_symbol_list); delete [] $1;}
;

c_comparison_op :
     GREATER   { $$= E_GREATER; }
  |  GREATEQ   { $$= E_GREATEQ; }
  |  LESS      { $$= E_LESS; }
  |  LESSEQ    { $$= E_LESSEQ; }
  |  EQ        { $$= E_EQUALS; }
;

//c_f_comp :
//    OPEN_BRAC c_comparison_op c_f_exp c_f_exp CLOSE_BRAC

// Goals

// FIX: PDDL BNF distinguishes between -ve literals and general -ve goals.
//         (different reqs)

c_pre_goal_descriptor :
	c_pref_goal_descriptor
		{$$= $1;}
/*|	c_goal_descriptor
		{$$=$1;}
;

*/
|   OPEN_BRAC AND c_pre_goal_descriptor_list CLOSE_BRAC
		{$$ = new conj_goal($3);}
|   OPEN_BRAC c_forall OPEN_BRAC c_typed_var_list CLOSE_BRAC
        c_pre_goal_descriptor CLOSE_BRAC
        {$$= new qfied_goal(E_FORALL,$4,$6,current_analysis->var_tab_stack.pop());
        requires(E_UNIV_PRECS);}
 | OPEN_BRAC AND CLOSE_BRAC {$$ = new conj_goal(new goal_list);}
 | OPEN_BRAC CLOSE_BRAC {$$ = new conj_goal(new goal_list);}
;

c_pref_con_goal :
	OPEN_BRAC PREFERENCE c_constraint_goal CLOSE_BRAC
		{$$ = new preference($3);requires(E_PREFERENCES);}
|   OPEN_BRAC PREFERENCE NAME c_constraint_goal CLOSE_BRAC
		{$$ = new preference($3,$4);requires(E_PREFERENCES);}
|   OPEN_BRAC AND c_pref_con_goal_list CLOSE_BRAC
		{$$ = new conj_goal($3);}
|   OPEN_BRAC c_forall OPEN_BRAC c_typed_var_list CLOSE_BRAC
        c_pref_goal CLOSE_BRAC
        {$$= new qfied_goal(E_FORALL,$4,$6,current_analysis->var_tab_stack.pop());
                requires(E_UNIV_PRECS);}
|   c_constraint_goal
	{$$ = $1;}
;

c_pref_goal :
	OPEN_BRAC PREFERENCE c_constraint_goal CLOSE_BRAC
		{$$ = new preference($3);requires(E_PREFERENCES);}
|   OPEN_BRAC PREFERENCE NAME c_constraint_goal CLOSE_BRAC
		{$$ = new preference($3,$4);requires(E_PREFERENCES);}
|   OPEN_BRAC AND c_pref_con_goal_list CLOSE_BRAC
		{$$ = new conj_goal($3);}
|   OPEN_BRAC c_forall OPEN_BRAC c_typed_var_list CLOSE_BRAC
        c_pref_goal CLOSE_BRAC
        {$$= new qfied_goal(E_FORALL,$4,$6,current_analysis->var_tab_stack.pop());
                requires(E_UNIV_PRECS);}
;

c_pref_con_goal_list :
	c_pref_con_goal_list c_pref_con_goal
		{$$=$1; $1->push_back($2);}
|	c_pref_con_goal
		{$$= new goal_list; $$->push_back($1);}
;

c_pref_goal_descriptor :
	OPEN_BRAC PREFERENCE c_goal_descriptor CLOSE_BRAC
	{$$= new preference($3); requires(E_PREFERENCES);}
|   OPEN_BRAC PREFERENCE NAME c_goal_descriptor CLOSE_BRAC
	{$$= new preference($3,$4); requires(E_PREFERENCES);}
// Restored...

|	c_goal_descriptor
	{$$=$1;}
;

c_constraint_goal_list :
	c_constraint_goal_list c_constraint_goal
	{$$ = $1; $$->push_back($2);}
|       c_constraint_goal
	{$$ = new goal_list; $$->push_back($1);}
;

c_constraint_goal :
	OPEN_BRAC AND c_constraint_goal_list CLOSE_BRAC
		{$$= new conj_goal($3);}
|	OPEN_BRAC c_forall OPEN_BRAC c_typed_var_list CLOSE_BRAC c_constraint_goal CLOSE_BRAC
		{$$ = new qfied_goal(E_FORALL,$4,$6,current_analysis->var_tab_stack.pop());
        requires(E_UNIV_PRECS);}
|	OPEN_BRAC AT_END c_goal_descriptor CLOSE_BRAC
		{$$ = new constraint_goal(E_ATEND,$3);}
|   OPEN_BRAC ALWAYS c_goal_descriptor CLOSE_BRAC
		{$$ = new constraint_goal(E_ALWAYS,$3);}
| 	OPEN_BRAC SOMETIME c_goal_descriptor CLOSE_BRAC
		{$$ = new constraint_goal(E_SOMETIME,$3);}
|	OPEN_BRAC WITHIN c_number c_goal_descriptor CLOSE_BRAC
		{$$ = new constraint_goal(E_WITHIN,$4,NULL,$3->double_value(),0.0);delete $3;}
|	OPEN_BRAC ATMOSTONCE c_goal_descriptor CLOSE_BRAC
		{$$ = new constraint_goal(E_ATMOSTONCE,$3);}
|	OPEN_BRAC SOMETIMEAFTER c_goal_descriptor c_goal_descriptor CLOSE_BRAC
		{$$ = new constraint_goal(E_SOMETIMEAFTER,$4,$3);}
|	OPEN_BRAC SOMETIMEBEFORE c_goal_descriptor c_goal_descriptor CLOSE_BRAC
		{$$ = new constraint_goal(E_SOMETIMEBEFORE,$4,$3);}
|   OPEN_BRAC ALWAYSWITHIN c_number c_goal_descriptor c_goal_descriptor CLOSE_BRAC
		{$$ = new constraint_goal(E_ALWAYSWITHIN,$5,$4,$3->double_value(),0.0);delete $3;}
| 	OPEN_BRAC HOLDDURING c_number c_number c_goal_descriptor CLOSE_BRAC
		{$$ = new constraint_goal(E_HOLDDURING,$5,NULL,$4->double_value(),$3->double_value());delete $3;delete $4;}
|   OPEN_BRAC HOLDAFTER c_number c_goal_descriptor CLOSE_BRAC
		{$$ = new constraint_goal(E_HOLDAFTER,$4,NULL,0.0,$3->double_value());delete $3;}
;

c_goal_descriptor :
   c_proposition
       {$$= new simple_goal($1,E_POS);}
|  OPEN_BRAC NOT c_goal_descriptor CLOSE_BRAC
       {$$= new neg_goal($3);simple_goal * s = dynamic_cast<simple_goal *>($3);
       if(s && s->getProp()->head->getName()=="=") {requires(E_EQUALITY);}
       else{requires(E_NEGATIVE_PRECONDITIONS);};}
|  OPEN_BRAC AND c_goal_list CLOSE_BRAC
       {$$= new conj_goal($3);}
|  OPEN_BRAC OR c_goal_list CLOSE_BRAC
       {$$= new disj_goal($3);
        requires(E_DISJUNCTIVE_PRECONDS);}
|  OPEN_BRAC IMPLY c_goal_descriptor c_goal_descriptor CLOSE_BRAC
       {$$= new imply_goal($3,$4);
        requires(E_DISJUNCTIVE_PRECONDS);}
|  OPEN_BRAC c_forall OPEN_BRAC c_typed_var_list CLOSE_BRAC
       c_goal_descriptor CLOSE_BRAC
       {$$= new qfied_goal($2,$4,$6,current_analysis->var_tab_stack.pop());}
|  OPEN_BRAC c_exists OPEN_BRAC c_typed_var_list CLOSE_BRAC
       c_goal_descriptor CLOSE_BRAC
       {$$= new qfied_goal($2,$4,$6,current_analysis->var_tab_stack.pop());}
|  OPEN_BRAC c_comparison_op c_f_exp c_f_exp CLOSE_BRAC
       {$$= new comparison($2,$3,$4);
        requires(E_NFLUENTS);}
;

c_pre_goal_descriptor_list :
	c_pre_goal_descriptor_list c_pre_goal_descriptor
		{$$=$1; $1->push_back($2);}
|	c_pre_goal_descriptor
		{$$= new goal_list; $$->push_back($1);}
;

c_goal_list :
    c_goal_list c_goal_descriptor
        {$$=$1; $1->push_back($2);}
|   c_goal_descriptor
	{$$= new goal_list; $$->push_back($1);}
;

//c_quantifier :
//    c_forall {$$=$1;}
//|   c_exists {$$=$1;}
;

c_forall :
    FORALL
       {$$=E_FORALL;
        current_analysis->var_tab_stack.push(
        		current_analysis->buildForallTab());}
;

c_exists :
    EXISTS
       {$$=E_EXISTS;
        current_analysis->var_tab_stack.push(
        	current_analysis->buildExistsTab());}
;

c_proposition :
    OPEN_BRAC c_pred_symbol c_parameter_symbols CLOSE_BRAC
        {$$=new proposition($2,$3);}
;

c_derived_proposition :
	OPEN_BRAC c_pred_symbol c_typed_var_list CLOSE_BRAC
	 {$$ = new proposition($2,$3);}
;

c_init_proposition :
    OPEN_BRAC c_init_pred_symbol c_parameter_symbols CLOSE_BRAC
        {$$=new proposition($2,$3);}
;

c_predicates :
    OPEN_BRAC PREDS c_pred_decls CLOSE_BRAC
        {$$= $3;}
|   OPEN_BRAC PREDS error CLOSE_BRAC
	{yyerrok; $$=NULL;
	 log_error(E_FATAL,"Syntax error in (:predicates ...)");
	}
;

c_functions_def :
    OPEN_BRAC FUNCTIONS c_func_decls CLOSE_BRAC
        {$$= $3;}
|   OPEN_BRAC FUNCTIONS error CLOSE_BRAC
	{yyerrok; $$=NULL;
	 log_error(E_FATAL,"Syntax error in (:functions ...)");
	}
;

c_constraints_def :
	OPEN_BRAC CONSTRAINTS c_constraint_goal CLOSE_BRAC
		{$$ = $3;}
| OPEN_BRAC CONSTRAINTS error CLOSE_BRAC
    {yyerrok; $$=NULL;
      log_error(E_FATAL,"Syntax error in (:constraints ...)");
      }
;

c_constraints_probdef :
	OPEN_BRAC CONSTRAINTS c_pref_con_goal CLOSE_BRAC
		{$$ = $3;};
| OPEN_BRAC CONSTRAINTS error CLOSE_BRAC
	 {yyerrok; $$=NULL;
      log_error(E_FATAL,"Syntax error in (:constraints ...)");
      }
;

c_structure_defs :
    c_structure_defs c_structure_def { $$=$1; $$->push_back($2); }
|   c_structure_def  { $$= new structure_store; $$->push_back($1); }
;

c_structure_def :
    c_action_def          { $$= $1; }
|   c_event_def           { $$= $1; requires(E_TIME); }
|   c_process_def         { $$= $1; requires(E_TIME); }
|   c_durative_action_def { $$= $1; requires(E_DURATIVE_ACTIONS); }
|   c_derivation_rule     { $$= $1; requires(E_DERIVED_PREDICATES);}
|   c_class_def           { $$ = $1; requires(E_MODULES);}
;

c_class_def : OPEN_BRAC CLASS c_class
// Friends here
// Predicates next
                c_functions_def
// Supports finally
                CLOSE_BRAC {$$ = new class_def($3,$4);};


c_rule_head :
    DERIVED {$$= 0;
    	current_analysis->var_tab_stack.push(
    					current_analysis->buildRuleTab());}
;

c_derivation_rule :
	OPEN_BRAC
	c_rule_head
	c_derived_proposition
	c_goal_descriptor
	CLOSE_BRAC
	{$$ = new derivation_rule($3,$4,current_analysis->var_tab_stack.pop());}
;

c_action_def :
    OPEN_BRAC
    ACTION
    NAME
    c_args_head OPEN_BRAC c_typed_var_list
    CLOSE_BRAC
    PRE c_pre_goal_descriptor
    EFFECTS c_effect
    CLOSE_BRAC
    { $$= current_analysis->buildAction(current_analysis->op_tab.new_symbol_put($3),
			$6,$9,$11,
			current_analysis->var_tab_stack.pop()); delete [] $3; }
|   OPEN_BRAC ACTION error CLOSE_BRAC
	{yyerrok;
	 log_error(E_FATAL,"Syntax error in action declaration.");
	 $$= NULL; }
;

c_event_def :
    OPEN_BRAC
    EVENT
    NAME /* $3 */
    c_args_head OPEN_BRAC c_typed_var_list CLOSE_BRAC
    PRE c_goal_descriptor /* $9 */
    EFFECTS c_effect /* $11 */
    CLOSE_BRAC
    {$$= current_analysis->buildEvent(current_analysis->op_tab.new_symbol_put($3),
		   $6,$9,$11,
		   current_analysis->var_tab_stack.pop()); delete [] $3;}

|   OPEN_BRAC EVENT error CLOSE_BRAC
	{yyerrok;
	 log_error(E_FATAL,"Syntax error in event declaration.");
	 $$= NULL; };

c_process_def :
    OPEN_BRAC
    PROCESS
    NAME
    c_args_head OPEN_BRAC c_typed_var_list CLOSE_BRAC
    PRE c_goal_descriptor
    EFFECTS c_proc_effect
    CLOSE_BRAC
    {$$= current_analysis->buildProcess(current_analysis->op_tab.new_symbol_put($3),
		     $6,$9,$11,
                     current_analysis->var_tab_stack.pop()); delete [] $3;}
|   OPEN_BRAC PROCESS error CLOSE_BRAC
	{yyerrok;
	 log_error(E_FATAL,"Syntax error in process declaration.");
	 $$= NULL; };

c_durative_action_def :
    OPEN_BRAC
    DURATIVE_ACTION
    NAME /* $3 */
    c_args_head OPEN_BRAC c_typed_var_list CLOSE_BRAC
    DURATION c_duration_constraint /* $9 */
    c_da_def_body
    CLOSE_BRAC
    { $$= $10;
      $$->name= current_analysis->op_tab.new_symbol_put($3);
      $$->symtab= current_analysis->var_tab_stack.pop();
      $$->parameters= $6;
      $$->dur_constraint= $9;
      delete [] $3;
    }

|   OPEN_BRAC DURATIVE_ACTION error CLOSE_BRAC
	{yyerrok;
	 log_error(E_FATAL,"Syntax error in durative-action declaration.");
	 $$= NULL; }
;

c_da_def_body :
    c_da_def_body EFFECTS c_da_effect
        {$$=$1; $$->effects=$3;}
|   c_da_def_body CONDITION c_da_gd
        {$$=$1; $$->precondition=$3;}
|   /* empty */  {$$= current_analysis->buildDurativeAction();}
;

c_da_gd :
   c_timed_gd
       { $$=$1; }
|  OPEN_BRAC AND c_da_gds CLOSE_BRAC
       { $$= new conj_goal($3); }
;

c_da_gds :
   c_da_gds c_da_gd
       { $$=$1; $$->push_back($2); }
|  /* empty */
       { $$= new goal_list; }
;

c_timed_gd :
    OPEN_BRAC AT_START c_goal_descriptor CLOSE_BRAC
        {$$= new timed_goal($3,E_AT_START);}
|   OPEN_BRAC AT_END   c_goal_descriptor CLOSE_BRAC
        {$$= new timed_goal($3,E_AT_END);}
|   OPEN_BRAC OVER_ALL c_goal_descriptor CLOSE_BRAC
        {$$= new timed_goal($3,E_OVER_ALL);}
|   OPEN_BRAC PREFERENCE NAME c_timed_gd CLOSE_BRAC
		{timed_goal * tg = dynamic_cast<timed_goal *>($4);
		$$ = new timed_goal(new preference($3,tg->clearGoal()),tg->getTime());
			delete tg;
			requires(E_PREFERENCES);}
|   OPEN_BRAC PREFERENCE c_timed_gd CLOSE_BRAC
        {$$ = new preference($3);requires(E_PREFERENCES);}
;

c_args_head :
    ARGS {$$= 0; current_analysis->var_tab_stack.push(
    				current_analysis->buildOpTab());}
;

c_require_key :
     EQUALITY    {$$= E_EQUALITY;}
   | STRIPS      {$$= E_STRIPS;}

   | TYPING      {$$= E_TYPING;}
   | NEGATIVE_PRECONDITIONS
   				 {$$= E_NEGATIVE_PRECONDITIONS;}
   | DISJUNCTIVE_PRECONDS
                 {$$= E_DISJUNCTIVE_PRECONDS;}
   | EXT_PRECS   {$$= E_EXT_PRECS;}
   | UNIV_PRECS  {$$= E_UNIV_PRECS;}
   | COND_EFFS   {$$= E_COND_EFFS;}
   | FLUENTS     {$$= E_NFLUENTS | E_OFLUENTS;}
   | DURATIVE_ACTIONS
                 {$$= E_DURATIVE_ACTIONS;}
   | TIME        {$$= E_TIME |
                      E_NFLUENTS |
                      E_DURATIVE_ACTIONS; }
   | ACTIONCOSTS {$$=E_ACTIONCOSTS | E_NFLUENTS;} // Note that this is a hack: should
   											// just be ACTIONCOSTS and then checks
   											// throughout for the right requirement
   | OBJECTFLUENTS {$$=E_OFLUENTS;}
   | NUMERICFLUENTS {$$=E_NFLUENTS;}
   | MODULES {$$=E_MODULES;}

   | ADL         {$$= E_STRIPS |
		      E_TYPING |
		      E_NEGATIVE_PRECONDITIONS |
		      E_DISJUNCTIVE_PRECONDS |
		      E_EQUALITY |
		      E_EXT_PRECS |
		      E_UNIV_PRECS |
		      E_COND_EFFS;}

   | QUANT_PRECS {$$= E_EXT_PRECS |
		      E_UNIV_PRECS;}

   | DURATION_INEQUALITIES
                 {$$= E_DURATION_INEQUALITIES;}

   | CONTINUOUS_EFFECTS
                 {$$= E_CONTINUOUS_EFFECTS;}
   | DERIVED_PREDICATES
   				 {$$ = E_DERIVED_PREDICATES;}
   | TIMED_INITIAL_LITERALS
   				{$$ = E_TIMED_INITIAL_LITERALS;}
   | PREFERENCES
   				{$$ = E_PREFERENCES;}
   | CONSTRAINTS
                {$$ = E_CONSTRAINTS;}
   | NAME
      {log_error(E_WARNING,"Unrecognised requirements declaration ");
       $$= 0; delete [] $1;}
;


c_domain_constants : OPEN_BRAC CONSTANTS c_typed_consts CLOSE_BRAC
    {$$=$3;}
;

c_type_names : OPEN_BRAC TYPES c_typed_types CLOSE_BRAC
    {$$=$3; requires(E_TYPING);}
;


c_problem : OPEN_BRAC
              DEFINE
              OPEN_BRAC PROBLEM NAME CLOSE_BRAC
              OPEN_BRAC FORDOMAIN NAME CLOSE_BRAC
              c_problem_body
            CLOSE_BRAC
            {$$=$11; $$->name = $5; $$->domain_name = $9;
		if (types_used && !types_defined) {
			yyerrok; log_error(E_FATAL,"Syntax error in problem file - types used, but no :types section in domain file.");
		}

	}
|   OPEN_BRAC DEFINE OPEN_BRAC PROBLEM error
    	{yyerrok; $$=NULL;
       	log_error(E_FATAL,"Syntax error in problem definition."); }

;

c_problem_body :
     c_domain_require_def c_problem_body {$$=$2; $$->req= $1;}
|    c_objects c_problem_body       {$$=$2; $$->objects= $1;}
|    c_initial_state c_problem_body {$$=$2; $$->initial_state= $1;}
|    c_goal_spec c_problem_body     {$$=$2; $$->the_goal= $1;}
|	 c_constraints_probdef c_problem_body
									{$$=$2; $$->constraints = $1;}
|    c_metric_spec c_problem_body   {$$=$2; if($$->metric == 0) {$$->metric= $1;}
											else {$$->metric->add($1);}}
|    c_length_spec c_problem_body   {$$=$2; $$->length= $1;}
|   /* Empty */                     {$$=new problem;}
;

c_objects : OPEN_BRAC OBJECTS c_typed_consts CLOSE_BRAC {$$=$3;}
;

c_initial_state : OPEN_BRAC INITIALLY c_init_els CLOSE_BRAC {$$=$3;}
;

c_goals : GOALS {$$ = current_analysis->buildOpTab();}
;

c_goal_spec : OPEN_BRAC c_goals c_pre_goal_descriptor CLOSE_BRAC {$$=$3;delete $2;}
;

c_metric_spec :
   OPEN_BRAC METRIC c_optimization c_ground_f_exp CLOSE_BRAC
       { $$= new metric_spec($3,$4); }
|  OPEN_BRAC METRIC error CLOSE_BRAC
       {yyerrok;
        log_error(E_FATAL,"Syntax error in metric declaration.");
        $$= NULL; }
;

c_length_spec :
   OPEN_BRAC LENGTH SERIAL INTVAL PARALLEL INTVAL CLOSE_BRAC
       {$$= new length_spec(E_BOTH,$4,$6);}
|
	OPEN_BRAC LENGTH SERIAL INTVAL CLOSE_BRAC
		{$$ = new length_spec(E_SERIAL,$4);}

|
	OPEN_BRAC LENGTH PARALLEL INTVAL CLOSE_BRAC
		{$$ = new length_spec(E_PARALLEL,$4);}
;



c_optimization :
   MINIMIZE {$$= E_MINIMIZE;}
|  MAXIMIZE {$$= E_MAXIMIZE;}
;


c_ground_f_exp :
    OPEN_BRAC c_binary_ground_f_exp CLOSE_BRAC {$$= $2;}
|   c_ground_f_head {$$= $1;}
|   c_number {$$= $1;}
|   TOTAL_TIME { $$= new special_val_expr(E_TOTAL_TIME); }
|   OPEN_BRAC ISVIOLATED NAME CLOSE_BRAC
		{$$ = new violation_term($3);}
|  OPEN_BRAC TOTAL_TIME CLOSE_BRAC { $$= new special_val_expr(E_TOTAL_TIME); }
;

c_binary_ground_f_exp :
    PLUS c_ground_f_exp c_binary_ground_f_pexps   { $$= new plus_expression($2,$3); }
|   HYPHEN c_ground_f_exp c_ground_f_exp { $$= new minus_expression($2,$3); }
|   MUL c_ground_f_exp c_binary_ground_f_mexps    { $$= new mul_expression($2,$3); }
|   DIV c_ground_f_exp c_ground_f_exp    { $$= new div_expression($2,$3); }
;

c_binary_ground_f_pexps :
	c_ground_f_exp {$$ = $1;}
|	c_ground_f_exp c_binary_ground_f_pexps
	{$$ = new plus_expression($1,$2);}
;

c_binary_ground_f_mexps :
	c_ground_f_exp {$$ = $1;}
|	c_ground_f_exp c_binary_ground_f_mexps
	{$$ = new mul_expression($1,$2);}
;
// Plans

c_plan :
    c_step_t_d c_plan
        {$$= $2;
         $$->push_front($1); }
|  TIME FLOATVAL c_plan
		{$$ = $3;$$->insertTime($2);}
|  TIME INTVAL c_plan
		{$$ = $3;$$->insertTime($2);}
| /* empty */
        {$$= new plan;}
;

c_step_t_d :
    c_float COLON c_step_d
        {$$=$3;
         $$->start_time_given=1;
         $$->start_time=$1;}
|   c_step_d
        {$$=$1;
	 $$->start_time_given=0;}
;

c_step_d :
    c_step OPEN_SQ c_float CLOSE_SQ
        {$$= $1;
	 $$->duration_given=1;
         $$->duration= $3;}
|   c_step
        {$$= $1;
         $$->duration_given=0;}
;

c_step :
    OPEN_BRAC NAME c_const_symbols CLOSE_BRAC
      {$$= new plan_step(
              current_analysis->op_tab.symbol_get($2),
	      $3); delete [] $2;
      }
;

c_float :
    FLOATVAL {$$= $1;}
|   INTVAL   {$$= (float) $1;}
;

%%

#include <cstdio>
#include <iostream>
int line_no= 1;
using std::istream;
#include "lex.yy.cc"

namespace VAL {
extern yyFlexLexer* yfl;
};


int yyerror(char * s)
{
    return 0;
}

int yylex()
{
    return yfl->yylex();
}
