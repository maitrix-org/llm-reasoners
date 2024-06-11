/* A Bison parser, made by GNU Bison 1.875d.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

/* Written by Richard Stallman by simplifying the original so called
   ``semantic'' parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     OPEN_BRAC = 258,
     CLOSE_BRAC = 259,
     OPEN_SQ = 260,
     CLOSE_SQ = 261,
     DEFINE = 262,
     PDDLDOMAIN = 263,
     REQS = 264,
     EQUALITY = 265,
     STRIPS = 266,
     ADL = 267,
     NEGATIVE_PRECONDITIONS = 268,
     TYPING = 269,
     DISJUNCTIVE_PRECONDS = 270,
     EXT_PRECS = 271,
     UNIV_PRECS = 272,
     QUANT_PRECS = 273,
     COND_EFFS = 274,
     FLUENTS = 275,
     TIME = 276,
     DURATIVE_ACTIONS = 277,
     DURATION_INEQUALITIES = 278,
     CONTINUOUS_EFFECTS = 279,
     DERIVED_PREDICATES = 280,
     TIMED_INITIAL_LITERALS = 281,
     PREFERENCES = 282,
     CONSTRAINTS = 283,
     ACTION = 284,
     PROCESS = 285,
     EVENT = 286,
     DURATIVE_ACTION = 287,
     DERIVED = 288,
     CONSTANTS = 289,
     PREDS = 290,
     FUNCTIONS = 291,
     TYPES = 292,
     ARGS = 293,
     PRE = 294,
     CONDITION = 295,
     PREFERENCE = 296,
     START_PRE = 297,
     END_PRE = 298,
     EFFECTS = 299,
     INITIAL_EFFECT = 300,
     FINAL_EFFECT = 301,
     INVARIANT = 302,
     DURATION = 303,
     AT_START = 304,
     AT_END = 305,
     OVER_ALL = 306,
     AND = 307,
     OR = 308,
     EXISTS = 309,
     FORALL = 310,
     IMPLY = 311,
     NOT = 312,
     WHEN = 313,
     EITHER = 314,
     PROBLEM = 315,
     FORDOMAIN = 316,
     INITIALLY = 317,
     OBJECTS = 318,
     GOALS = 319,
     EQ = 320,
     LENGTH = 321,
     SERIAL = 322,
     PARALLEL = 323,
     METRIC = 324,
     MINIMIZE = 325,
     MAXIMIZE = 326,
     HASHT = 327,
     DURATION_VAR = 328,
     TOTAL_TIME = 329,
     INCREASE = 330,
     DECREASE = 331,
     SCALE_UP = 332,
     SCALE_DOWN = 333,
     ASSIGN = 334,
     GREATER = 335,
     GREATEQ = 336,
     LESS = 337,
     LESSEQ = 338,
     Q = 339,
     COLON = 340,
     ALWAYS = 341,
     SOMETIME = 342,
     WITHIN = 343,
     ATMOSTONCE = 344,
     SOMETIMEAFTER = 345,
     SOMETIMEBEFORE = 346,
     ALWAYSWITHIN = 347,
     HOLDDURING = 348,
     HOLDAFTER = 349,
     ISVIOLATED = 350,
     BOGUS = 351,
     NAME = 352,
     FUNCTION_SYMBOL = 353,
     INTVAL = 354,
     FLOATVAL = 355,
     AT_TIME = 356,
     PLUS = 357,
     HYPHEN = 358,
     DIV = 359,
     MUL = 360,
     UMINUS = 361
   };
#endif
#define OPEN_BRAC 258
#define CLOSE_BRAC 259
#define OPEN_SQ 260
#define CLOSE_SQ 261
#define DEFINE 262
#define PDDLDOMAIN 263
#define REQS 264
#define EQUALITY 265
#define STRIPS 266
#define ADL 267
#define NEGATIVE_PRECONDITIONS 268
#define TYPING 269
#define DISJUNCTIVE_PRECONDS 270
#define EXT_PRECS 271
#define UNIV_PRECS 272
#define QUANT_PRECS 273
#define COND_EFFS 274
#define FLUENTS 275
#define TIME 276
#define DURATIVE_ACTIONS 277
#define DURATION_INEQUALITIES 278
#define CONTINUOUS_EFFECTS 279
#define DERIVED_PREDICATES 280
#define TIMED_INITIAL_LITERALS 281
#define PREFERENCES 282
#define CONSTRAINTS 283
#define ACTION 284
#define PROCESS 285
#define EVENT 286
#define DURATIVE_ACTION 287
#define DERIVED 288
#define CONSTANTS 289
#define PREDS 290
#define FUNCTIONS 291
#define TYPES 292
#define ARGS 293
#define PRE 294
#define CONDITION 295
#define PREFERENCE 296
#define START_PRE 297
#define END_PRE 298
#define EFFECTS 299
#define INITIAL_EFFECT 300
#define FINAL_EFFECT 301
#define INVARIANT 302
#define DURATION 303
#define AT_START 304
#define AT_END 305
#define OVER_ALL 306
#define AND 307
#define OR 308
#define EXISTS 309
#define FORALL 310
#define IMPLY 311
#define NOT 312
#define WHEN 313
#define EITHER 314
#define PROBLEM 315
#define FORDOMAIN 316
#define INITIALLY 317
#define OBJECTS 318
#define GOALS 319
#define EQ 320
#define LENGTH 321
#define SERIAL 322
#define PARALLEL 323
#define METRIC 324
#define MINIMIZE 325
#define MAXIMIZE 326
#define HASHT 327
#define DURATION_VAR 328
#define TOTAL_TIME 329
#define INCREASE 330
#define DECREASE 331
#define SCALE_UP 332
#define SCALE_DOWN 333
#define ASSIGN 334
#define GREATER 335
#define GREATEQ 336
#define LESS 337
#define LESSEQ 338
#define Q 339
#define COLON 340
#define ALWAYS 341
#define SOMETIME 342
#define WITHIN 343
#define ATMOSTONCE 344
#define SOMETIMEAFTER 345
#define SOMETIMEBEFORE 346
#define ALWAYSWITHIN 347
#define HOLDDURING 348
#define HOLDAFTER 349
#define ISVIOLATED 350
#define BOGUS 351
#define NAME 352
#define FUNCTION_SYMBOL 353
#define INTVAL 354
#define FLOATVAL 355
#define AT_TIME 356
#define PLUS 357
#define HYPHEN 358
#define DIV 359
#define MUL 360
#define UMINUS 361




/* Copy the first part of user declarations.  */
#line 17 "pddl+.yacc"

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


extern int yylex();

using namespace VAL;



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 68 "pddl+.yacc"
typedef union YYSTYPE {
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
} YYSTYPE;
/* Line 191 of yacc.c.  */
#line 410 "pddl+.cpp"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 214 of yacc.c.  */
#line 422 "pddl+.cpp"

#if ! defined (yyoverflow) || YYERROR_VERBOSE

# ifndef YYFREE
#  define YYFREE free
# endif
# ifndef YYMALLOC
#  define YYMALLOC malloc
# endif

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   define YYSTACK_ALLOC alloca
#  endif
# else
#  if defined (alloca) || defined (_ALLOCA_H)
#   define YYSTACK_ALLOC alloca
#  else
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning. */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
# else
#  if defined (__STDC__) || defined (__cplusplus)
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   define YYSIZE_T size_t
#  endif
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
# endif
#endif /* ! defined (yyoverflow) || YYERROR_VERBOSE */


#if (! defined (yyoverflow) \
     && (! defined (__cplusplus) \
	 || (defined (YYSTYPE_IS_TRIVIAL) && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  short int yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (short int) + sizeof (YYSTYPE))			\
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined (__GNUC__) && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  register YYSIZE_T yyi;		\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (0)
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (0)

#endif

#if defined (__STDC__) || defined (__cplusplus)
   typedef signed char yysigned_char;
#else
   typedef short int yysigned_char;
#endif

/* YYFINAL -- State number of the termination state. */
#define YYFINAL  17
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   851

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  107
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  122
/* YYNRULES -- Number of rules. */
#define YYNRULES  320
/* YYNRULES -- Number of states. */
#define YYNSTATES  723

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   361

#define YYTRANSLATE(YYX) 						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const unsigned short int yyprhs[] =
{
       0,     0,     3,     5,     7,     9,    15,    20,    23,    26,
      29,    32,    35,    38,    40,    45,    50,    55,    58,    59,
      62,    64,    69,    73,    75,    77,    79,    81,    84,    85,
      90,    94,    96,   101,   106,   108,   112,   113,   118,   123,
     125,   128,   129,   132,   133,   138,   143,   145,   148,   152,
     153,   155,   157,   159,   161,   166,   168,   170,   173,   174,
     177,   178,   185,   188,   191,   194,   195,   200,   203,   206,
     209,   210,   212,   214,   216,   218,   220,   225,   227,   229,
     231,   233,   236,   239,   242,   243,   248,   253,   258,   266,
     272,   274,   276,   279,   280,   285,   290,   296,   302,   306,
     311,   313,   315,   317,   319,   322,   325,   328,   329,   335,
     341,   347,   353,   359,   365,   371,   376,   379,   380,   382,
     385,   387,   389,   395,   401,   407,   413,   418,   425,   435,
     445,   447,   449,   451,   453,   456,   457,   462,   464,   469,
     471,   479,   485,   491,   497,   503,   509,   515,   520,   526,
     532,   538,   544,   546,   548,   554,   560,   562,   564,   566,
     571,   576,   578,   583,   588,   590,   592,   594,   596,   598,
     600,   602,   607,   615,   620,   626,   631,   639,   641,   644,
     645,   650,   656,   658,   661,   662,   667,   675,   680,   685,
     690,   696,   701,   707,   713,   720,   727,   733,   735,   740,
     745,   750,   756,   764,   770,   773,   774,   777,   778,   780,
     782,   784,   786,   791,   796,   801,   806,   811,   816,   821,
     826,   831,   836,   841,   844,   846,   848,   850,   852,   854,
     856,   858,   864,   877,   882,   895,   900,   913,   918,   930,
     935,   939,   943,   944,   946,   951,   954,   955,   960,   965,
     970,   976,   981,   983,   985,   987,   989,   991,   993,   995,
     997,   999,  1001,  1003,  1005,  1007,  1009,  1011,  1013,  1015,
    1017,  1019,  1021,  1023,  1028,  1033,  1046,  1052,  1055,  1058,
    1061,  1064,  1067,  1070,  1073,  1074,  1079,  1084,  1086,  1091,
    1097,  1102,  1107,  1108,  1114,  1116,  1118,  1122,  1124,  1126,
    1128,  1133,  1137,  1141,  1145,  1149,  1153,  1155,  1158,  1160,
    1163,  1166,  1170,  1174,  1175,  1179,  1181,  1186,  1188,  1193,
    1195
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short int yyrhs[] =
{
     108,     0,    -1,   109,    -1,   209,    -1,   224,    -1,     3,
       7,   111,   110,     4,    -1,     3,     7,   111,     1,    -1,
     112,   110,    -1,   208,   110,    -1,   207,   110,    -1,   189,
     110,    -1,   190,   110,    -1,   191,   110,    -1,   193,    -1,
       3,     8,    97,     4,    -1,     3,     9,   113,     4,    -1,
       3,     9,     1,     4,    -1,   113,   206,    -1,    -1,   115,
     114,    -1,   115,    -1,     3,   116,   122,     4,    -1,     3,
       1,     4,    -1,    97,    -1,    65,    -1,    97,    -1,    97,
      -1,   119,   120,    -1,    -1,     3,   121,   122,     4,    -1,
       3,     1,     4,    -1,    97,    -1,   123,   103,   135,   122,
      -1,   123,   103,   133,   122,    -1,   123,    -1,    84,   129,
     123,    -1,    -1,   126,   103,   135,   124,    -1,   126,   103,
     133,   124,    -1,   126,    -1,   131,   125,    -1,    -1,   132,
     126,    -1,    -1,   136,   103,   135,   127,    -1,   136,   103,
     133,   127,    -1,   136,    -1,   128,   131,    -1,   128,    84,
     130,    -1,    -1,    97,    -1,    97,    -1,    97,    -1,    97,
      -1,     3,    59,   137,     4,    -1,    97,    -1,    97,    -1,
     136,   134,    -1,    -1,   137,   135,    -1,    -1,   138,     3,
      65,   171,   170,     4,    -1,   138,   164,    -1,   138,   163,
      -1,   138,   139,    -1,    -1,     3,   101,   138,     4,    -1,
     142,   140,    -1,   166,   140,    -1,   165,   140,    -1,    -1,
     145,    -1,   162,    -1,   161,    -1,   166,    -1,   165,    -1,
       3,    52,   144,     4,    -1,   143,    -1,   161,    -1,   162,
      -1,   167,    -1,   144,   161,    -1,   144,   162,    -1,   144,
     167,    -1,    -1,     3,    52,   140,     4,    -1,     3,    52,
       1,     4,    -1,     3,    52,   147,     4,    -1,     3,   184,
       3,   122,     4,   146,     4,    -1,     3,    58,   202,   146,
       4,    -1,   148,    -1,   167,    -1,   147,   146,    -1,    -1,
       3,    49,   149,     4,    -1,     3,    50,   149,     4,    -1,
       3,    75,   171,   169,     4,    -1,     3,    76,   171,   169,
       4,    -1,     3,     1,     4,    -1,     3,    52,   151,     4,
      -1,   150,    -1,   161,    -1,   162,    -1,   152,    -1,   151,
     161,    -1,   151,   162,    -1,   151,   152,    -1,    -1,     3,
      79,   171,   155,     4,    -1,     3,    75,   171,   155,     4,
      -1,     3,    76,   171,   155,     4,    -1,     3,    77,   171,
     155,     4,    -1,     3,    78,   171,   155,     4,    -1,     3,
      75,   171,   169,     4,    -1,     3,    76,   171,   169,     4,
      -1,     3,    52,   154,     4,    -1,   154,   153,    -1,    -1,
     156,    -1,    84,    73,    -1,   170,    -1,   171,    -1,     3,
     102,   155,   155,     4,    -1,     3,   103,   155,   155,     4,
      -1,     3,   105,   155,   155,     4,    -1,     3,   104,   155,
     155,     4,    -1,     3,    52,   160,     4,    -1,     3,   158,
      84,    73,   159,     4,    -1,     3,    49,     3,   158,    84,
      73,   159,     4,     4,    -1,     3,    50,     3,   158,    84,
      73,   159,     4,     4,    -1,    83,    -1,    81,    -1,    65,
      -1,   168,    -1,   160,   157,    -1,    -1,     3,    57,   186,
       4,    -1,   186,    -1,     3,    57,   188,     4,    -1,   188,
      -1,     3,   184,     3,   122,     4,   141,     4,    -1,     3,
      58,   180,   140,     4,    -1,     3,    79,   171,   168,     4,
      -1,     3,    75,   171,   168,     4,    -1,     3,    76,   171,
     168,     4,    -1,     3,    77,   171,   168,     4,    -1,     3,
      78,   171,   168,     4,    -1,     3,   103,   168,     4,    -1,
       3,   102,   168,   168,     4,    -1,     3,   103,   168,   168,
       4,    -1,     3,   105,   168,   168,     4,    -1,     3,   104,
     168,   168,     4,    -1,   170,    -1,   171,    -1,     3,   105,
      72,   168,     4,    -1,     3,   105,   168,    72,     4,    -1,
      72,    -1,    99,    -1,   100,    -1,     3,    98,   128,     4,
      -1,     3,    97,   128,     4,    -1,    98,    -1,     3,    98,
     128,     4,    -1,     3,    97,   128,     4,    -1,    98,    -1,
      80,    -1,    81,    -1,    82,    -1,    83,    -1,    65,    -1,
     177,    -1,     3,    52,   181,     4,    -1,     3,   184,     3,
     122,     4,   174,     4,    -1,     3,    41,   179,     4,    -1,
       3,    41,    97,   179,     4,    -1,     3,    52,   176,     4,
      -1,     3,   184,     3,   122,     4,   175,     4,    -1,   179,
      -1,   176,   175,    -1,    -1,     3,    41,   180,     4,    -1,
       3,    41,    97,   180,     4,    -1,   180,    -1,   178,   179,
      -1,    -1,     3,    52,   178,     4,    -1,     3,   184,     3,
     122,     4,   179,     4,    -1,     3,    50,   180,     4,    -1,
       3,    86,   180,     4,    -1,     3,    87,   180,     4,    -1,
       3,    88,   170,   180,     4,    -1,     3,    89,   180,     4,
      -1,     3,    90,   180,   180,     4,    -1,     3,    91,   180,
     180,     4,    -1,     3,    92,   170,   180,   180,     4,    -1,
       3,    93,   170,   170,   180,     4,    -1,     3,    94,   170,
     180,     4,    -1,   186,    -1,     3,    57,   180,     4,    -1,
       3,    52,   182,     4,    -1,     3,    53,   182,     4,    -1,
       3,    56,   180,   180,     4,    -1,     3,   183,     3,   122,
       4,   180,     4,    -1,     3,   173,   168,   168,     4,    -1,
     181,   174,    -1,    -1,   182,   180,    -1,    -1,   184,    -1,
     185,    -1,    55,    -1,    54,    -1,     3,   117,   128,     4,
      -1,     3,   117,   122,     4,    -1,     3,   118,   128,     4,
      -1,     3,    35,   114,     4,    -1,     3,    35,     1,     4,
      -1,     3,    36,   119,     4,    -1,     3,    36,     1,     4,
      -1,     3,    28,   179,     4,    -1,     3,    28,     1,     4,
      -1,     3,    28,   175,     4,    -1,     3,    28,     1,     4,
      -1,   193,   194,    -1,   194,    -1,   197,    -1,   198,    -1,
     199,    -1,   200,    -1,   196,    -1,    33,    -1,     3,   195,
     187,   180,     4,    -1,     3,    29,    97,   205,     3,   122,
       4,    39,   174,    44,   141,     4,    -1,     3,    29,     1,
       4,    -1,     3,    31,    97,   205,     3,   122,     4,    39,
     180,    44,   141,     4,    -1,     3,    31,     1,     4,    -1,
       3,    30,    97,   205,     3,   122,     4,    39,   180,    44,
     153,     4,    -1,     3,    30,     1,     4,    -1,     3,    32,
      97,   205,     3,   122,     4,    48,   157,   201,     4,    -1,
       3,    32,     1,     4,    -1,   201,    44,   146,    -1,   201,
      40,   202,    -1,    -1,   204,    -1,     3,    52,   203,     4,
      -1,   203,   202,    -1,    -1,     3,    49,   180,     4,    -1,
       3,    50,   180,     4,    -1,     3,    51,   180,     4,    -1,
       3,    41,    97,   204,     4,    -1,     3,    41,   204,     4,
      -1,    38,    -1,    10,    -1,    11,    -1,    14,    -1,    13,
      -1,    15,    -1,    16,    -1,    17,    -1,    19,    -1,    20,
      -1,    22,    -1,    21,    -1,    12,    -1,    18,    -1,    23,
      -1,    24,    -1,    25,    -1,    26,    -1,    27,    -1,    28,
      -1,    97,    -1,     3,    34,   124,     4,    -1,     3,    37,
     127,     4,    -1,     3,     7,     3,    60,    97,     4,     3,
      61,    97,     4,   210,     4,    -1,     3,     7,     3,    60,
       1,    -1,   112,   210,    -1,   211,   210,    -1,   212,   210,
      -1,   214,   210,    -1,   192,   210,    -1,   215,   210,    -1,
     216,   210,    -1,    -1,     3,    63,   124,     4,    -1,     3,
      62,   138,     4,    -1,    64,    -1,     3,   213,   174,     4,
      -1,     3,    69,   219,   220,     4,    -1,     3,    69,     1,
       4,    -1,     3,    66,   217,     4,    -1,    -1,    67,    99,
     218,    68,    99,    -1,    70,    -1,    71,    -1,     3,   221,
       4,    -1,   172,    -1,   170,    -1,    74,    -1,     3,    95,
      97,     4,    -1,     3,    74,     4,    -1,   102,   220,   222,
      -1,   103,   220,   220,    -1,   105,   220,   223,    -1,   104,
     220,   220,    -1,   220,    -1,   220,   222,    -1,   220,    -1,
     220,   223,    -1,   225,   224,    -1,    21,   100,   224,    -1,
      21,    99,   224,    -1,    -1,   228,    85,   226,    -1,   226,
      -1,   227,     5,   228,     6,    -1,   227,    -1,     3,    97,
     125,     4,    -1,   100,    -1,    99,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short int yyrline[] =
{
       0,   240,   240,   241,   242,   246,   248,   255,   256,   257,
     258,   260,   262,   264,   267,   272,   279,   286,   287,   292,
     294,   299,   301,   309,   317,   319,   327,   332,   334,   338,
     340,   347,   360,   368,   376,   388,   390,   396,   404,   413,
     418,   419,   423,   424,   432,   439,   448,   454,   456,   458,
     465,   471,   475,   479,   483,   488,   495,   500,   502,   506,
     508,   512,   517,   519,   521,   524,   528,   534,   535,   537,
     539,   548,   549,   550,   551,   552,   556,   557,   561,   563,
     565,   572,   573,   574,   576,   580,   582,   590,   592,   600,
     605,   608,   615,   616,   620,   622,   624,   628,   632,   639,
     640,   644,   646,   648,   655,   656,   657,   659,   664,   666,
     668,   670,   672,   677,   683,   689,   694,   695,   699,   700,
     702,   703,   707,   709,   711,   713,   718,   720,   723,   726,
     732,   733,   734,   742,   746,   749,   753,   758,   765,   770,
     775,   780,   785,   787,   789,   791,   793,   798,   800,   802,
     804,   806,   808,   809,   813,   815,   817,   823,   824,   827,
     830,   832,   850,   852,   854,   860,   861,   862,   863,   864,
     876,   878,   880,   887,   889,   891,   893,   897,   902,   905,
     909,   911,   913,   918,   921,   925,   927,   930,   932,   934,
     936,   938,   940,   942,   944,   946,   948,   953,   955,   959,
     961,   964,   967,   970,   976,   979,   983,   986,   990,   991,
     995,  1002,  1009,  1014,  1019,  1024,  1026,  1033,  1035,  1042,
    1044,  1051,  1053,  1060,  1061,  1065,  1066,  1067,  1068,  1069,
    1073,  1079,  1088,  1099,  1106,  1117,  1123,  1133,  1139,  1154,
    1161,  1163,  1165,  1169,  1171,  1176,  1179,  1183,  1185,  1187,
    1189,  1194,  1199,  1204,  1205,  1207,  1208,  1210,  1212,  1213,
    1214,  1215,  1216,  1218,  1222,  1231,  1234,  1237,  1239,  1241,
    1243,  1245,  1247,  1253,  1257,  1262,  1269,  1276,  1277,  1278,
    1279,  1280,  1282,  1283,  1284,  1287,  1290,  1293,  1296,  1300,
    1302,  1309,  1314,  1314,  1319,  1320,  1325,  1326,  1327,  1328,
    1329,  1331,  1335,  1336,  1337,  1338,  1342,  1343,  1348,  1349,
    1355,  1358,  1360,  1363,  1367,  1371,  1377,  1381,  1387,  1395,
    1396
};
#endif

#if YYDEBUG || YYERROR_VERBOSE
/* YYTNME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals. */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "OPEN_BRAC", "CLOSE_BRAC", "OPEN_SQ",
  "CLOSE_SQ", "DEFINE", "PDDLDOMAIN", "REQS", "EQUALITY", "STRIPS", "ADL",
  "NEGATIVE_PRECONDITIONS", "TYPING", "DISJUNCTIVE_PRECONDS", "EXT_PRECS",
  "UNIV_PRECS", "QUANT_PRECS", "COND_EFFS", "FLUENTS", "TIME",
  "DURATIVE_ACTIONS", "DURATION_INEQUALITIES", "CONTINUOUS_EFFECTS",
  "DERIVED_PREDICATES", "TIMED_INITIAL_LITERALS", "PREFERENCES",
  "CONSTRAINTS", "ACTION", "PROCESS", "EVENT", "DURATIVE_ACTION",
  "DERIVED", "CONSTANTS", "PREDS", "FUNCTIONS", "TYPES", "ARGS", "PRE",
  "CONDITION", "PREFERENCE", "START_PRE", "END_PRE", "EFFECTS",
  "INITIAL_EFFECT", "FINAL_EFFECT", "INVARIANT", "DURATION", "AT_START",
  "AT_END", "OVER_ALL", "AND", "OR", "EXISTS", "FORALL", "IMPLY", "NOT",
  "WHEN", "EITHER", "PROBLEM", "FORDOMAIN", "INITIALLY", "OBJECTS",
  "GOALS", "EQ", "LENGTH", "SERIAL", "PARALLEL", "METRIC", "MINIMIZE",
  "MAXIMIZE", "HASHT", "DURATION_VAR", "TOTAL_TIME", "INCREASE",
  "DECREASE", "SCALE_UP", "SCALE_DOWN", "ASSIGN", "GREATER", "GREATEQ",
  "LESS", "LESSEQ", "Q", "COLON", "ALWAYS", "SOMETIME", "WITHIN",
  "ATMOSTONCE", "SOMETIMEAFTER", "SOMETIMEBEFORE", "ALWAYSWITHIN",
  "HOLDDURING", "HOLDAFTER", "ISVIOLATED", "BOGUS", "NAME",
  "FUNCTION_SYMBOL", "INTVAL", "FLOATVAL", "AT_TIME", "PLUS", "HYPHEN",
  "DIV", "MUL", "UMINUS", "$accept", "mystartsymbol", "c_domain",
  "c_preamble", "c_domain_name", "c_domain_require_def", "c_reqs",
  "c_pred_decls", "c_pred_decl", "c_new_pred_symbol", "c_pred_symbol",
  "c_init_pred_symbol", "c_func_decls", "c_func_decl", "c_new_func_symbol",
  "c_typed_var_list", "c_var_symbol_list", "c_typed_consts",
  "c_const_symbols", "c_new_const_symbols", "c_typed_types",
  "c_parameter_symbols", "c_declaration_var_symbol", "c_var_symbol",
  "c_const_symbol", "c_new_const_symbol", "c_either_type",
  "c_new_primitive_type", "c_primitive_type", "c_new_primitive_types",
  "c_primitive_types", "c_init_els", "c_timed_initial_literal",
  "c_effects", "c_effect", "c_a_effect", "c_p_effect", "c_p_effects",
  "c_conj_effect", "c_da_effect", "c_da_effects", "c_timed_effect",
  "c_a_effect_da", "c_p_effect_da", "c_p_effects_da", "c_f_assign_da",
  "c_proc_effect", "c_proc_effects", "c_f_exp_da", "c_binary_expr_da",
  "c_duration_constraint", "c_d_op", "c_d_value", "c_duration_constraints",
  "c_neg_simple_effect", "c_pos_simple_effect", "c_init_neg_simple_effect",
  "c_init_pos_simple_effect", "c_forall_effect", "c_cond_effect",
  "c_assignment", "c_f_exp", "c_f_exp_t", "c_number", "c_f_head",
  "c_ground_f_head", "c_comparison_op", "c_pre_goal_descriptor",
  "c_pref_con_goal", "c_pref_con_goal_list", "c_pref_goal_descriptor",
  "c_constraint_goal_list", "c_constraint_goal", "c_goal_descriptor",
  "c_pre_goal_descriptor_list", "c_goal_list", "c_quantifier", "c_forall",
  "c_exists", "c_proposition", "c_derived_proposition",
  "c_init_proposition", "c_predicates", "c_functions_def",
  "c_constraints_def", "c_constraints_probdef", "c_structure_defs",
  "c_structure_def", "c_rule_head", "c_derivation_rule", "c_action_def",
  "c_event_def", "c_process_def", "c_durative_action_def", "c_da_def_body",
  "c_da_gd", "c_da_gds", "c_timed_gd", "c_args_head", "c_require_key",
  "c_domain_constants", "c_type_names", "c_problem", "c_problem_body",
  "c_objects", "c_initial_state", "c_goals", "c_goal_spec",
  "c_metric_spec", "c_length_spec", "c_length_field", "@1",
  "c_optimization", "c_ground_f_exp", "c_binary_ground_f_exp",
  "c_binary_ground_f_pexps", "c_binary_ground_f_mexps", "c_plan",
  "c_step_t_d", "c_step_d", "c_step", "c_float", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const unsigned short int yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,   107,   108,   108,   108,   109,   109,   110,   110,   110,
     110,   110,   110,   110,   111,   112,   112,   113,   113,   114,
     114,   115,   115,   116,   117,   117,   118,   119,   119,   120,
     120,   121,   122,   122,   122,   123,   123,   124,   124,   124,
     125,   125,   126,   126,   127,   127,   127,   128,   128,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   136,   137,
     137,   138,   138,   138,   138,   138,   139,   140,   140,   140,
     140,   141,   141,   141,   141,   141,   142,   142,   143,   143,
     143,   144,   144,   144,   144,   145,   145,   146,   146,   146,
     146,   146,   147,   147,   148,   148,   148,   148,   148,   149,
     149,   150,   150,   150,   151,   151,   151,   151,   152,   152,
     152,   152,   152,   153,   153,   153,   154,   154,   155,   155,
     155,   155,   156,   156,   156,   156,   157,   157,   157,   157,
     158,   158,   158,   159,   160,   160,   161,   162,   163,   164,
     165,   166,   167,   167,   167,   167,   167,   168,   168,   168,
     168,   168,   168,   168,   169,   169,   169,   170,   170,   171,
     171,   171,   172,   172,   172,   173,   173,   173,   173,   173,
     174,   174,   174,   175,   175,   175,   175,   175,   176,   176,
     177,   177,   177,   178,   178,   179,   179,   179,   179,   179,
     179,   179,   179,   179,   179,   179,   179,   180,   180,   180,
     180,   180,   180,   180,   181,   181,   182,   182,   183,   183,
     184,   185,   186,   187,   188,   189,   189,   190,   190,   191,
     191,   192,   192,   193,   193,   194,   194,   194,   194,   194,
     195,   196,   197,   197,   198,   198,   199,   199,   200,   200,
     201,   201,   201,   202,   202,   203,   203,   204,   204,   204,
     204,   204,   205,   206,   206,   206,   206,   206,   206,   206,
     206,   206,   206,   206,   206,   206,   206,   206,   206,   206,
     206,   206,   206,   207,   208,   209,   209,   210,   210,   210,
     210,   210,   210,   210,   210,   211,   212,   213,   214,   215,
     215,   216,   218,   217,   219,   219,   220,   220,   220,   220,
     220,   220,   221,   221,   221,   221,   222,   222,   223,   223,
     224,   224,   224,   224,   225,   225,   226,   226,   227,   228,
     228
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     1,     1,     5,     4,     2,     2,     2,
       2,     2,     2,     1,     4,     4,     4,     2,     0,     2,
       1,     4,     3,     1,     1,     1,     1,     2,     0,     4,
       3,     1,     4,     4,     1,     3,     0,     4,     4,     1,
       2,     0,     2,     0,     4,     4,     1,     2,     3,     0,
       1,     1,     1,     1,     4,     1,     1,     2,     0,     2,
       0,     6,     2,     2,     2,     0,     4,     2,     2,     2,
       0,     1,     1,     1,     1,     1,     4,     1,     1,     1,
       1,     2,     2,     2,     0,     4,     4,     4,     7,     5,
       1,     1,     2,     0,     4,     4,     5,     5,     3,     4,
       1,     1,     1,     1,     2,     2,     2,     0,     5,     5,
       5,     5,     5,     5,     5,     4,     2,     0,     1,     2,
       1,     1,     5,     5,     5,     5,     4,     6,     9,     9,
       1,     1,     1,     1,     2,     0,     4,     1,     4,     1,
       7,     5,     5,     5,     5,     5,     5,     4,     5,     5,
       5,     5,     1,     1,     5,     5,     1,     1,     1,     4,
       4,     1,     4,     4,     1,     1,     1,     1,     1,     1,
       1,     4,     7,     4,     5,     4,     7,     1,     2,     0,
       4,     5,     1,     2,     0,     4,     7,     4,     4,     4,
       5,     4,     5,     5,     6,     6,     5,     1,     4,     4,
       4,     5,     7,     5,     2,     0,     2,     0,     1,     1,
       1,     1,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     4,     2,     1,     1,     1,     1,     1,     1,
       1,     5,    12,     4,    12,     4,    12,     4,    11,     4,
       3,     3,     0,     1,     4,     2,     0,     4,     4,     4,
       5,     4,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     4,     4,    12,     5,     2,     2,     2,
       2,     2,     2,     2,     0,     4,     4,     1,     4,     5,
       4,     4,     0,     5,     1,     1,     3,     1,     1,     1,
       4,     3,     3,     3,     3,     3,     1,     2,     1,     2,
       2,     3,     3,     0,     3,     1,     4,     1,     4,     1,
       1
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned short int yydefact[] =
{
     313,     0,     0,   320,   319,     0,     2,     3,     4,   313,
     315,   317,     0,     0,    41,   313,   313,     1,     0,   310,
       0,     0,     0,     0,    52,     0,    41,   312,   311,     0,
     314,     0,     0,     6,     0,     0,     0,     0,     0,     0,
      13,   224,   229,   225,   226,   227,   228,     0,     0,   318,
      40,   316,     0,   276,     0,     0,     0,     0,     0,     0,
       0,   230,    43,     0,     0,    58,     0,     5,     7,    10,
      11,    12,     0,   223,     9,     8,    14,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    53,     0,    39,    43,     0,     0,     0,    20,     0,
       0,     0,    46,     0,     0,     0,    16,    15,   253,   254,
     264,   256,   255,   257,   258,   259,   265,   260,   261,   263,
     262,   266,   267,   268,   269,   270,   271,   272,    17,   220,
       0,   184,   210,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   219,   233,   252,     0,   237,     0,   235,
       0,   239,     0,   273,     0,    42,   216,     0,    23,    36,
     215,    19,   218,     0,   217,    27,   274,    55,     0,    57,
      24,    25,    36,     0,     0,   197,     0,     0,     0,     0,
       0,   157,   158,     0,     0,     0,     0,     0,     0,     0,
      36,    36,    36,    36,    36,     0,    56,    43,    43,    22,
       0,     0,    34,     0,    31,    36,    58,    58,     0,   207,
     207,   211,     0,     0,   169,   165,   166,   167,   168,    49,
       0,     0,   208,   209,   231,     0,   187,   185,   183,   188,
     189,     0,   191,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    60,    38,    37,    50,    36,    21,     0,
      30,     0,    45,    44,   213,     0,     0,     0,     0,     0,
       0,   161,     0,   152,   153,    36,   284,   190,   192,   193,
       0,     0,   196,     0,     0,     0,     0,     0,     0,    35,
      36,    36,    29,   199,   206,   200,     0,   198,   212,     0,
      47,    49,    49,     0,     0,     0,     0,     0,     0,     0,
     284,   284,     0,   284,   284,   284,   284,   284,   194,   195,
       0,     0,     0,     0,     0,    54,    59,    33,    32,   201,
      51,    48,     0,     0,     0,     0,     0,     0,   203,     0,
       0,    65,    43,   287,     0,     0,     0,   277,   281,   275,
     278,   279,   280,   282,   283,   186,     0,     0,   170,   182,
       0,     0,     0,   242,   160,   159,     0,   147,     0,     0,
       0,     0,     0,     0,     0,   177,     0,     0,     0,     0,
       0,   294,   295,     0,     0,     0,   205,     0,     0,     0,
       0,     0,     0,   135,   132,   131,   130,     0,     0,   148,
     149,   151,   150,   202,   222,     0,   179,     0,   221,     0,
     286,    64,    63,    62,   139,   285,   292,   291,   290,     0,
     299,   164,   298,   297,     0,   288,     0,     0,     0,    36,
       0,     0,    71,    73,    72,    75,    74,   137,     0,     0,
       0,     0,     0,     0,     0,   238,     0,     0,     0,     0,
       0,    36,     0,     0,    26,    65,    49,     0,     0,     0,
      49,    49,     0,     0,     0,     0,     0,   289,     0,   180,
     171,   204,     0,     0,     0,     0,     0,   232,   117,     0,
       0,   236,   234,     0,     0,   126,   134,     0,     0,   241,
     243,     0,   240,    90,    91,     0,   173,   175,   178,     0,
       0,     0,     0,     0,     0,     0,     0,   301,     0,     0,
       0,     0,     0,     0,     0,   296,   181,     0,     0,     0,
       0,    70,    77,    78,    79,    70,    70,    80,     0,     0,
      70,    36,     0,     0,     0,     0,     0,     0,   133,     0,
       0,     0,     0,   246,     0,     0,     0,    93,     0,     0,
       0,     0,     0,     0,     0,   174,     0,   138,     0,    66,
     214,   293,   300,   163,   162,   306,   302,   303,   305,   308,
     304,     0,    86,    84,     0,     0,    85,    67,    69,    68,
     136,     0,     0,   115,   116,     0,   156,     0,     0,     0,
       0,   127,     0,     0,     0,     0,     0,     0,     0,    98,
       0,     0,   100,   103,   101,   102,     0,     0,     0,     0,
       0,     0,     0,     0,    36,     0,     0,    61,   307,   309,
     172,     0,     0,     0,   141,     0,     0,   113,   114,     0,
       0,     0,   251,   247,   248,   249,   244,   245,   107,     0,
       0,     0,     0,     0,    94,    95,    87,    92,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   176,     0,
      76,    81,    82,    83,     0,     0,     0,     0,     0,   250,
       0,     0,     0,     0,     0,     0,    89,     0,   143,    96,
     144,    97,   145,   146,   142,     0,   140,     0,     0,   128,
     129,     0,    99,   106,   104,   105,     0,     0,     0,   118,
     120,   121,     0,     0,     0,     0,     0,     0,   154,   155,
       0,     0,     0,     0,   119,   109,   110,   111,   112,   108,
      88,     0,     0,     0,     0,     0,     0,     0,     0,   122,
     123,   125,   124
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short int yydefgoto[] =
{
      -1,     5,     6,    35,    23,   300,    79,    97,    98,   159,
     219,   446,   100,   165,   205,   201,   202,    92,    25,    93,
     101,   259,   247,   321,   290,    94,   197,   169,   198,   102,
     278,   366,   401,   510,   421,   511,   512,   611,   422,   482,
     597,   483,   591,   592,   660,   593,   429,   522,   688,   689,
     353,   387,   527,   433,   513,   514,   402,   403,   515,   516,
     517,   528,   577,   263,   264,   413,   220,   347,   364,   440,
     348,   178,   365,   349,   418,   255,   221,   466,   223,   175,
     104,   404,    37,    38,    39,   301,    40,    41,    66,    42,
      43,    44,    45,    46,   388,   479,   588,   480,   146,   128,
      47,    48,     7,   302,   303,   304,   336,   305,   306,   307,
     369,   447,   373,   555,   456,   556,   560,     8,     9,    10,
      11,    12
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -482
static const short int yypact[] =
{
      43,    33,   174,  -482,  -482,    56,  -482,  -482,  -482,    67,
    -482,    42,   -19,    75,   -13,    67,    67,  -482,    -6,  -482,
     306,   135,    40,   246,  -482,   150,   -13,  -482,  -482,   165,
    -482,   110,    13,  -482,   645,   213,   267,   267,   267,   267,
     278,  -482,  -482,  -482,  -482,  -482,  -482,   267,   267,  -482,
    -482,  -482,   274,  -482,   291,   209,   289,    23,    30,    64,
      85,  -482,   218,   298,    58,  -482,   329,  -482,  -482,  -482,
    -482,  -482,   313,  -482,  -482,  -482,  -482,   346,   324,   483,
     352,   550,   355,   361,   336,   377,   336,   381,   336,   388,
     336,  -482,   393,   334,   218,   424,    91,   441,   436,   478,
     305,   480,   -45,   -23,   489,   453,  -482,  -482,  -482,  -482,
    -482,  -482,  -482,  -482,  -482,  -482,  -482,  -482,  -482,  -482,
    -482,  -482,  -482,  -482,  -482,  -482,  -482,  -482,  -482,  -482,
     489,  -482,  -482,   489,   489,   317,   489,   489,   489,   317,
     317,   317,   513,  -482,  -482,  -482,   531,  -482,   549,  -482,
     553,  -482,   561,  -482,    18,  -482,  -482,   565,  -482,   509,
    -482,  -482,  -482,   105,  -482,  -482,  -482,  -482,    18,  -482,
    -482,  -482,   509,   529,   594,  -482,   504,   600,   415,   602,
     603,  -482,  -482,   489,   604,   489,   489,   489,   317,   489,
     509,   509,   509,   509,   509,   560,  -482,   218,   218,  -482,
     521,   617,   522,   620,  -482,   509,  -482,  -482,   627,  -482,
    -482,  -482,   489,   489,   199,  -482,  -482,  -482,  -482,  -482,
     101,   629,  -482,  -482,  -482,   630,  -482,  -482,  -482,  -482,
    -482,   642,  -482,   643,   644,   489,   489,   647,   648,   651,
     652,   653,   655,  -482,  -482,  -482,  -482,   509,  -482,    18,
    -482,   656,  -482,  -482,  -482,   419,   422,   489,   657,   207,
     456,  -482,   101,  -482,  -482,   509,   646,  -482,  -482,  -482,
     658,   679,  -482,   681,   649,   650,   659,   585,    11,  -482,
     509,   509,  -482,  -482,  -482,  -482,   693,  -482,  -482,   609,
    -482,  -482,  -482,   101,   101,   101,   101,   695,   708,   332,
     646,   646,   709,   646,   646,   646,   646,   646,  -482,  -482,
     710,   712,   489,   489,   713,  -482,  -482,  -482,  -482,  -482,
    -482,  -482,   210,   237,   101,    41,   101,   101,  -482,   489,
     390,  -482,   218,  -482,   654,    32,   712,  -482,  -482,  -482,
    -482,  -482,  -482,  -482,  -482,  -482,   490,   673,  -482,  -482,
     674,   675,   248,  -482,  -482,  -482,   716,  -482,   718,   719,
     720,   721,   722,   386,   723,  -482,   452,   724,   631,   725,
     727,  -482,  -482,    47,   728,    34,  -482,   730,   731,   732,
     731,   733,   734,  -482,  -482,  -482,  -482,   660,   235,  -482,
    -482,  -482,  -482,  -482,  -482,    82,  -482,   735,  -482,   215,
    -482,  -482,  -482,  -482,  -482,  -482,  -482,  -482,  -482,   525,
    -482,  -482,  -482,  -482,   736,  -482,   489,   737,   462,   509,
     206,   738,  -482,  -482,  -482,  -482,  -482,  -482,   295,   739,
     741,     6,     6,   464,   666,  -482,   743,   744,   681,   745,
     485,   509,   747,    52,  -482,  -482,  -482,   680,   748,   661,
    -482,  -482,    47,    47,    47,    47,   749,  -482,   750,  -482,
    -482,  -482,   751,   205,   753,   489,   754,  -482,  -482,    52,
      52,  -482,  -482,   667,   676,  -482,  -482,   101,   472,  -482,
    -482,   275,  -482,  -482,  -482,   755,  -482,  -482,  -482,   757,
     665,   759,   435,   317,   516,   238,   668,  -482,   760,   239,
     242,    47,    47,    47,    47,  -482,  -482,   712,   761,   460,
     762,   765,  -482,  -482,  -482,   765,   765,  -482,   -23,   766,
     765,   509,   537,    90,    90,   696,   698,   768,  -482,   115,
     489,   489,   489,  -482,   769,   771,   771,  -482,   743,    52,
      52,    52,    52,    52,   772,  -482,   773,  -482,   774,  -482,
    -482,  -482,  -482,  -482,  -482,    47,  -482,  -482,  -482,    47,
    -482,   775,  -482,  -482,    52,    52,  -482,  -482,  -482,  -482,
    -482,   776,   777,  -482,  -482,   672,  -482,   778,   779,   101,
     101,  -482,   380,   781,   782,   783,   784,   785,   545,  -482,
     538,   786,  -482,  -482,  -482,  -482,   787,   547,   744,    57,
      57,   101,   101,   101,   509,   788,   710,  -482,  -482,  -482,
    -482,   559,   101,   101,  -482,   731,    60,  -482,  -482,   789,
     790,   791,  -482,  -482,  -482,  -482,  -482,  -482,  -482,    52,
      52,    52,    52,    52,  -482,  -482,  -482,  -482,   792,   589,
     793,   794,   795,   796,   797,   798,   799,   800,  -482,   588,
    -482,  -482,  -482,  -482,   801,   101,   740,   802,   803,  -482,
     571,    92,    92,    92,    92,    92,  -482,    60,  -482,  -482,
    -482,  -482,  -482,  -482,  -482,   744,  -482,   804,   805,  -482,
    -482,   593,  -482,  -482,  -482,  -482,   598,   742,   806,  -482,
    -482,  -482,   807,   809,   810,   812,    65,   813,  -482,  -482,
      92,    92,    92,    92,  -482,  -482,  -482,  -482,  -482,  -482,
    -482,    92,    92,    92,    92,   814,   815,   816,   817,  -482,
    -482,  -482,  -482
};

/* YYPGOTO[NTERM-NUM].  */
static const short int yypgoto[] =
{
    -482,  -482,  -482,   396,  -482,   325,  -482,   687,  -482,  -482,
     726,  -482,  -482,  -482,  -482,  -164,   575,  -162,   808,   729,
     382,  -253,  -482,  -482,    53,  -482,  -150,  -482,  -159,  -482,
    -482,   379,  -482,   -30,  -370,  -482,  -482,  -482,  -482,  -393,
    -482,  -482,   290,  -482,  -482,   167,   303,  -482,  -250,  -482,
     395,   160,    17,  -482,  -358,  -355,  -482,  -482,  -361,  -346,
    -425,  -213,  -262,  -135,  -254,  -482,  -482,  -320,  -437,  -482,
    -482,  -482,   -55,   -61,  -482,   621,  -482,   -79,  -482,  -367,
    -482,   391,  -482,  -482,  -482,  -482,  -482,   811,  -482,  -482,
    -482,  -482,  -482,  -482,  -482,  -481,  -482,  -478,   294,  -482,
    -482,  -482,  -482,   404,  -482,  -482,  -482,  -482,  -482,  -482,
    -482,  -482,  -482,  -319,  -482,   277,   271,   250,  -482,   818,
    -482,   820
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -71
static const short int yytable[] =
{
     183,    82,   142,   488,   187,   188,   189,   262,   208,   207,
     430,   427,   484,   427,    53,   315,   374,   425,   206,   425,
     423,   195,   423,   424,    83,   424,   238,   239,   240,   241,
     242,    85,   426,   370,   426,   244,   245,   173,   322,   323,
      13,   251,   170,   174,   260,   357,     1,    20,    31,   297,
     409,   584,   167,   236,   414,   492,    17,   598,   168,    99,
     639,   -28,   -28,   260,     2,    87,    21,    26,   260,   177,
      18,   384,   179,   180,   171,   184,   185,   186,    22,    26,
     324,   325,   326,   327,    24,    81,    89,   385,     2,   386,
     281,    14,   157,   575,   222,   686,   427,   519,   461,   280,
      32,   298,   371,   372,   260,   621,   203,   627,   196,   605,
      54,   356,   358,   359,   360,   196,   317,   318,   582,   316,
      84,   410,   231,   228,   233,   234,   235,    86,   237,   576,
      14,   416,   655,   501,   502,   503,   504,   678,    18,   261,
     181,   182,     3,     4,   427,   411,   181,   182,   427,   427,
     261,   257,   258,   427,    49,   261,   181,   182,   261,   181,
     182,    88,   576,   261,   181,   182,     3,     4,   427,   427,
     367,    51,   484,   484,   270,   271,   687,   594,   594,   438,
     595,   595,    90,   557,   558,   559,   653,   561,   158,   493,
     261,   181,   182,   495,   284,   284,   286,   499,   500,   261,
     181,   182,   204,   -24,   637,   638,   508,    52,   509,   -70,
      78,   288,   583,   -18,   354,   523,   524,    67,   310,   -18,
     -18,   -18,   -18,   -18,   -18,   -18,   -18,   -18,   -18,   -18,
     -18,   -18,   -18,   -18,   -18,   -18,   -18,   -18,   412,   435,
     559,   355,   550,   553,   427,   654,   554,    33,   427,    34,
     484,   350,   351,   651,   425,   462,   652,   423,   463,    19,
     424,   132,   578,   464,   465,    27,    28,   377,   361,   426,
      34,   170,   442,    15,    16,   436,   534,   489,    76,   437,
     443,    72,   697,   -24,   397,   599,   600,   601,   602,   603,
      80,   289,    81,   427,   289,    77,   -24,   381,   382,    95,
     383,    96,   684,   171,    24,   685,   -18,    24,   163,   164,
     612,   613,   444,   384,   417,    91,   445,   412,   412,   412,
     412,   289,   289,   289,   535,   536,   289,   537,   106,   385,
     132,   386,   103,   538,    24,    24,    24,   641,   643,    24,
     439,    55,    57,    58,    59,    60,    61,   468,    36,   105,
     539,   540,   541,   542,   543,   458,   129,   572,   548,   143,
     330,    36,    36,    36,    36,   144,   412,   412,   412,   412,
     469,   470,    36,    36,   145,   661,   662,   663,   664,   665,
     148,   147,   150,   485,   152,   149,   640,   642,   644,   645,
     646,   362,   151,   363,   331,   332,   333,   153,   334,   640,
     642,   335,   544,   656,   520,     3,     4,   691,   691,   691,
     691,   691,   692,   693,   694,   695,   181,   182,    81,   227,
     412,   529,   173,   283,   412,   173,   285,   395,   156,   530,
     531,   532,    68,    69,    70,    71,   130,   154,   396,    96,
     647,   132,   677,    74,    75,   160,   691,   691,   691,   691,
     711,   712,   713,   714,   696,   399,   400,   691,   691,   691,
     691,   715,   716,   717,   718,   346,   460,   352,   475,   585,
     586,   587,   133,   134,   135,   136,   137,   138,   139,   140,
     141,   567,   162,   360,   166,   568,   569,   107,   363,   487,
     571,   606,   173,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   563,   529,   176,   132,   190,   464,   465,   399,
     549,   530,   531,   532,   533,   170,   690,   690,   690,   690,
     690,   375,   291,   292,   191,   564,   565,   541,   542,   543,
     428,   573,   376,   210,   211,   132,   212,   213,   478,   626,
     481,   636,   192,   291,   292,   214,   193,   171,   293,   294,
     295,   296,   649,   650,   194,   690,   690,   690,   690,   199,
     215,   216,   217,   218,   681,   682,   690,   690,   690,   690,
     127,   209,   210,   211,   132,   212,   213,   171,   252,   253,
     628,   473,   474,   200,   214,   464,   619,   620,   224,   448,
     130,   225,   131,   170,   226,   132,   229,   230,   232,   215,
     216,   217,   218,   629,   630,   631,   632,   633,   246,   243,
     449,   248,   450,   451,   250,   249,   171,   452,   453,   454,
     455,   254,   265,   314,   266,   171,   133,   134,   135,   136,
     137,   138,   139,   140,   141,   464,   267,   268,   269,   299,
     464,   272,   273,   170,    55,   274,   275,   276,   170,   277,
     282,   287,   308,   564,   565,   541,   542,   543,   629,   630,
     631,   632,   633,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,   309,    81,   171,   291,   292,   311,   312,
     171,   293,   294,   295,   667,   291,   292,   319,   313,   328,
     700,   701,   702,   703,   337,   338,   320,   340,   341,   342,
     343,   344,   329,   339,   345,   346,   352,   378,   379,   380,
     389,   368,   390,   391,   392,   393,   394,   398,   405,   407,
     406,   408,   415,   419,   420,   428,   431,   432,   441,   477,
     457,   459,   467,   471,   434,   472,   478,   481,   496,   486,
     490,   525,   497,   505,   506,   507,   518,   521,   498,   545,
     526,   546,   444,   547,   552,   562,   566,   551,   509,   579,
     570,   580,   581,   589,   590,   604,   363,   616,   607,   610,
     614,   615,   617,   618,   582,   161,   622,   623,   624,   625,
     634,   635,   648,   657,   658,   659,   666,   668,   669,   670,
     671,   672,   673,   674,   675,   676,   679,   680,   698,   699,
     705,   706,   678,   707,   708,   704,   709,   710,   719,   720,
     721,   722,   279,   155,   494,   574,   596,   683,   476,   172,
     609,   256,   608,   491,    50,     0,     0,     0,     0,    30,
      29,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    73
};

static const short int yycheck[] =
{
     135,    56,    81,   440,   139,   140,   141,   220,   172,   168,
     380,   378,   437,   380,     1,     4,   336,   378,   168,   380,
     378,     3,   380,   378,     1,   380,   190,   191,   192,   193,
     194,     1,   378,     1,   380,   197,   198,     3,   291,   292,
       7,   205,    65,   104,     3,     4,     3,     5,     8,   262,
       3,   529,    97,   188,   373,     3,     0,   538,   103,     1,
       3,     3,     4,     3,    21,     1,    85,    14,     3,   130,
       3,    65,   133,   134,    97,   136,   137,   138,     3,    26,
     293,   294,   295,   296,    97,     3,     1,    81,    21,    83,
     249,    97,     1,     3,   173,     3,   463,   464,   418,   249,
      60,   265,    70,    71,     3,   583,     1,   588,    97,   546,
      97,   324,   325,   326,   327,    97,   280,   281,     3,   278,
      97,    74,   183,   178,   185,   186,   187,    97,   189,    72,
      97,    97,    72,   452,   453,   454,   455,    72,     3,    98,
      99,   100,    99,   100,   511,    98,    99,   100,   515,   516,
      98,   212,   213,   520,     4,    98,    99,   100,    98,    99,
     100,    97,    72,    98,    99,   100,    99,   100,   535,   536,
     332,     6,   597,   598,   235,   236,    84,   535,   536,    97,
     535,   536,    97,   502,   503,   504,   611,   507,    97,   443,
      98,    99,   100,   446,   255,   256,   257,   450,   451,    98,
      99,   100,    97,     4,   597,   598,     1,    97,     3,     4,
       1,     4,    97,     4,     4,   469,   470,     4,   273,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,   373,     4,
     559,     4,     4,     4,   611,   615,     4,     1,   615,     3,
     675,   312,   313,   611,   615,   419,   611,   615,    52,     9,
     615,    55,   524,    57,    58,    15,    16,   346,   329,   615,
       3,    65,    57,    99,   100,    40,     1,   441,     4,    44,
      65,     3,   675,    84,   363,   539,   540,   541,   542,   543,
       1,    84,     3,   660,    84,     4,    97,    49,    50,     1,
      52,     3,   660,    97,    97,   660,    97,    97,     3,     4,
     564,   565,    97,    65,   375,    97,   101,   452,   453,   454,
     455,    84,    84,    84,    49,    50,    84,    52,     4,    81,
      55,    83,     3,    58,    97,    97,    97,   599,   600,    97,
     395,     9,    29,    30,    31,    32,    33,    52,    23,     3,
      75,    76,    77,    78,    79,   416,     4,   521,   493,     4,
      28,    36,    37,    38,    39,     4,   501,   502,   503,   504,
      75,    76,    47,    48,    38,   629,   630,   631,   632,   633,
      86,     4,    88,   438,    90,     4,   599,   600,   601,   602,
     603,     1,     4,     3,    62,    63,    64,     4,    66,   612,
     613,    69,   481,   616,   465,    99,   100,   661,   662,   663,
     664,   665,   662,   663,   664,   665,    99,   100,     3,     4,
     555,    41,     3,     4,   559,     3,     4,    41,     4,    49,
      50,    51,    36,    37,    38,    39,    50,   103,    52,     3,
     604,    55,   655,    47,    48,     4,   700,   701,   702,   703,
     700,   701,   702,   703,   667,     3,     4,   711,   712,   713,
     714,   711,   712,   713,   714,     3,     4,     3,     4,   530,
     531,   532,    86,    87,    88,    89,    90,    91,    92,    93,
      94,   511,     4,   696,     4,   515,   516,     4,     3,     4,
     520,   546,     3,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    52,    41,    61,    55,     3,    57,    58,     3,
       4,    49,    50,    51,    52,    65,   661,   662,   663,   664,
     665,    41,    97,    98,     3,    75,    76,    77,    78,    79,
       3,     4,    52,    53,    54,    55,    56,    57,     3,     4,
       3,     4,     3,    97,    98,    65,     3,    97,   102,   103,
     104,   105,     3,     4,     3,   700,   701,   702,   703,     4,
      80,    81,    82,    83,     3,     4,   711,   712,   713,   714,
      97,    52,    53,    54,    55,    56,    57,    97,   206,   207,
      52,   431,   432,    84,    65,    57,   579,   580,     4,    74,
      50,    97,    52,    65,     4,    55,     4,     4,     4,    80,
      81,    82,    83,    75,    76,    77,    78,    79,    97,    59,
      95,     4,    97,    98,     4,   103,    97,   102,   103,   104,
     105,     4,     3,    48,     4,    97,    86,    87,    88,    89,
      90,    91,    92,    93,    94,    57,     4,     4,     4,     3,
      57,     4,     4,    65,     9,     4,     4,     4,    65,     4,
       4,     4,     4,    75,    76,    77,    78,    79,    75,    76,
      77,    78,    79,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,     4,     3,    97,    97,    98,    39,    39,
      97,   102,   103,   104,   105,    97,    98,     4,    39,     4,
     102,   103,   104,   105,   300,   301,    97,   303,   304,   305,
     306,   307,     4,     4,     4,     3,     3,    44,    44,    44,
       4,    67,     4,     4,     4,     4,     4,     4,     4,     4,
      99,     4,     4,     3,     3,     3,     3,     3,     3,    73,
       4,     4,     4,     4,    84,     4,     3,     3,    68,     4,
       3,    84,     4,     4,     4,     4,     3,     3,    97,     4,
      84,     4,    97,     4,     4,     4,     4,    99,     3,    73,
       4,    73,     4,     4,     3,     3,     3,   105,     4,     4,
       4,     4,     4,     4,     3,    98,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     4,    72,     4,     4,    73,     4,     4,     4,     4,
       4,     4,   247,    94,   445,   522,   536,   660,   433,   103,
     559,   210,   555,   442,    26,    -1,    -1,    -1,    -1,    21,
      20,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    40
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,     3,    21,    99,   100,   108,   109,   209,   224,   225,
     226,   227,   228,     7,    97,    99,   100,     0,     3,   224,
       5,    85,     3,   111,    97,   125,   131,   224,   224,   228,
     226,     8,    60,     1,     3,   110,   112,   189,   190,   191,
     193,   194,   196,   197,   198,   199,   200,   207,   208,     4,
     125,     6,    97,     1,    97,     9,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,   195,     4,   110,   110,
     110,   110,     3,   194,   110,   110,     4,     4,     1,   113,
       1,     3,   179,     1,    97,     1,    97,     1,    97,     1,
      97,    97,   124,   126,   132,     1,     3,   114,   115,     1,
     119,   127,   136,     3,   187,     3,     4,     4,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    97,   206,     4,
      50,    52,    55,    86,    87,    88,    89,    90,    91,    92,
      93,    94,   184,     4,     4,    38,   205,     4,   205,     4,
     205,     4,   205,     4,   103,   126,     4,     1,    97,   116,
       4,   114,     4,     3,     4,   120,     4,    97,   103,   134,
      65,    97,   117,     3,   180,   186,    61,   180,   178,   180,
     180,    99,   100,   170,   180,   180,   180,   170,   170,   170,
       3,     3,     3,     3,     3,     3,    97,   133,   135,     4,
      84,   122,   123,     1,    97,   121,   133,   135,   122,    52,
      53,    54,    56,    57,    65,    80,    81,    82,    83,   117,
     173,   183,   184,   185,     4,    97,     4,     4,   179,     4,
       4,   180,     4,   180,   180,   180,   170,   180,   122,   122,
     122,   122,   122,    59,   124,   124,    97,   129,     4,   103,
       4,   122,   127,   127,     4,   182,   182,   180,   180,   128,
       3,    98,   168,   170,   171,     3,     4,     4,     4,     4,
     180,   180,     4,     4,     4,     4,     4,     4,   137,   123,
     133,   135,     4,     4,   180,     4,   180,     4,     4,    84,
     131,    97,    98,   102,   103,   104,   105,   168,   122,     3,
     112,   192,   210,   211,   212,   214,   215,   216,     4,     4,
     179,    39,    39,    39,    48,     4,   135,   122,   122,     4,
      97,   130,   128,   128,   168,   168,   168,   168,     4,     4,
      28,    62,    63,    64,    66,    69,   213,   210,   210,     4,
     210,   210,   210,   210,   210,     4,     3,   174,   177,   180,
     180,   180,     3,   157,     4,     4,   168,     4,   168,   168,
     168,   180,     1,     3,   175,   179,   138,   124,    67,   217,
       1,    70,    71,   219,   174,    41,    52,   184,    44,    44,
      44,    49,    50,    52,    65,    81,    83,   158,   201,     4,
       4,     4,     4,     4,     4,    41,    52,   184,     4,     3,
       4,   139,   163,   164,   188,     4,    99,     4,     4,     3,
      74,    98,   170,   172,   220,     4,    97,   180,   181,     3,
       3,   141,   145,   161,   162,   165,   166,   186,     3,   153,
     141,     3,     3,   160,    84,     4,    40,    44,    97,   179,
     176,     3,    57,    65,    97,   101,   118,   218,    74,    95,
      97,    98,   102,   103,   104,   105,   221,     4,   180,     4,
       4,   174,   122,    52,    57,    58,   184,     4,    52,    75,
      76,     4,     4,   158,   158,     4,   157,    73,     3,   202,
     204,     3,   146,   148,   167,   179,     4,     4,   175,   122,
       3,   188,     3,   171,   138,   128,    68,     4,    97,   128,
     128,   220,   220,   220,   220,     4,     4,     4,     1,     3,
     140,   142,   143,   161,   162,   165,   166,   167,     3,   186,
     180,     3,   154,   171,   171,    84,    84,   159,   168,    41,
      49,    50,    51,    52,     1,    49,    50,    52,    58,    75,
      76,    77,    78,    79,   184,     4,     4,     4,   170,     4,
       4,    99,     4,     4,     4,   220,   222,   220,   220,   220,
     223,   174,     4,    52,    75,    76,     4,   140,   140,   140,
       4,   140,   122,     4,   153,     3,    72,   169,   169,    73,
      73,     4,     3,    97,   204,   180,   180,   180,   203,     4,
       3,   149,   150,   152,   161,   162,   149,   147,   202,   171,
     171,   171,   171,   171,     3,   175,   179,     4,   222,   223,
       4,   144,   171,   171,     4,     4,   105,     4,     4,   159,
     159,   204,     4,     4,     4,     4,     4,   202,    52,    75,
      76,    77,    78,    79,     4,     4,     4,   146,   146,     3,
     168,   169,   168,   169,   168,   168,   168,   122,     4,     3,
       4,   161,   162,   167,   141,    72,   168,     4,     4,     4,
     151,   171,   171,   171,   171,   171,     4,   105,     4,     4,
       4,     4,     4,     4,     4,     4,     4,   168,    72,     4,
       4,     3,     4,   152,   161,   162,     3,    84,   155,   156,
     170,   171,   155,   155,   155,   155,   168,   146,     4,     4,
     102,   103,   104,   105,    73,     4,     4,     4,     4,     4,
       4,   155,   155,   155,   155,   155,   155,   155,   155,     4,
       4,     4,     4
};

#if ! defined (YYSIZE_T) && defined (__SIZE_TYPE__)
# define YYSIZE_T __SIZE_TYPE__
#endif
#if ! defined (YYSIZE_T) && defined (size_t)
# define YYSIZE_T size_t
#endif
#if ! defined (YYSIZE_T)
# if defined (__STDC__) || defined (__cplusplus)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# endif
#endif
#if ! defined (YYSIZE_T)
# define YYSIZE_T unsigned int
#endif

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK;						\
      goto yybackup;						\
    }								\
  else								\
    { 								\
      yyerror ("syntax error: cannot back up");\
      YYERROR;							\
    }								\
while (0)

#define YYTERROR	1
#define YYERRCODE	256

/* YYLLOC_DEFAULT -- Compute the default location (before the actions
   are run).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)		\
   ((Current).first_line   = (Rhs)[1].first_line,	\
    (Current).first_column = (Rhs)[1].first_column,	\
    (Current).last_line    = (Rhs)[N].last_line,	\
    (Current).last_column  = (Rhs)[N].last_column)
#endif

/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (0)

# define YYDSYMPRINT(Args)			\
do {						\
  if (yydebug)					\
    yysymprint Args;				\
} while (0)

# define YYDSYMPRINTF(Title, Token, Value, Location)		\
do {								\
  if (yydebug)							\
    {								\
      YYFPRINTF (stderr, "%s ", Title);				\
      yysymprint (stderr, 					\
                  Token, Value);	\
      YYFPRINTF (stderr, "\n");					\
    }								\
} while (0)

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_stack_print (short int *bottom, short int *top)
#else
static void
yy_stack_print (bottom, top)
    short int *bottom;
    short int *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (/* Nothing. */; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_reduce_print (int yyrule)
#else
static void
yy_reduce_print (yyrule)
    int yyrule;
#endif
{
  int yyi;
  unsigned int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %u), ",
             yyrule - 1, yylno);
  /* Print the symbols being reduced, and their result.  */
  for (yyi = yyprhs[yyrule]; 0 <= yyrhs[yyi]; yyi++)
    YYFPRINTF (stderr, "%s ", yytname [yyrhs[yyi]]);
  YYFPRINTF (stderr, "-> %s\n", yytname [yyr1[yyrule]]);
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (Rule);		\
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YYDSYMPRINT(Args)
# define YYDSYMPRINTF(Title, Token, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   SIZE_MAX < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#if defined (YYMAXDEPTH) && YYMAXDEPTH == 0
# undef YYMAXDEPTH
#endif

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined (__GLIBC__) && defined (_STRING_H)
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
#   if defined (__STDC__) || defined (__cplusplus)
yystrlen (const char *yystr)
#   else
yystrlen (yystr)
     const char *yystr;
#   endif
{
  register const char *yys = yystr;

  while (*yys++ != '\0')
    continue;

  return yys - yystr - 1;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined (__GLIBC__) && defined (_STRING_H) && defined (_GNU_SOURCE)
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
#   if defined (__STDC__) || defined (__cplusplus)
yystpcpy (char *yydest, const char *yysrc)
#   else
yystpcpy (yydest, yysrc)
     char *yydest;
     const char *yysrc;
#   endif
{
  register char *yyd = yydest;
  register const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

#endif /* !YYERROR_VERBOSE */



#if YYDEBUG
/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yysymprint (FILE *yyoutput, int yytype, YYSTYPE *yyvaluep)
#else
static void
yysymprint (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

  if (yytype < YYNTOKENS)
    {
      YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
# ifdef YYPRINT
      YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
    }
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  switch (yytype)
    {
      default:
        break;
    }
  YYFPRINTF (yyoutput, ")");
}

#endif /* ! YYDEBUG */
/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yydestruct (int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yytype, yyvaluep)
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

  switch (yytype)
    {

      default:
        break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
# if defined (__STDC__) || defined (__cplusplus)
int yyparse (void *YYPARSE_PARAM);
# else
int yyparse ();
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
# if defined (__STDC__) || defined (__cplusplus)
int yyparse (void *YYPARSE_PARAM)
# else
int yyparse (YYPARSE_PARAM)
  void *YYPARSE_PARAM;
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  register int yystate;
  register int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  short int yyssa[YYINITDEPTH];
  short int *yyss = yyssa;
  register short int *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  register YYSTYPE *yyvsp;



#define YYPOPSTACK   (yyvsp--, yyssp--)

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* When reducing, the number of symbols on the RHS of the reduced
     rule.  */
  int yylen;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;


  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed. so pushing a state here evens the stacks.
     */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack. Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	short int *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow ("parser stack overflow",
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyoverflowlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyoverflowlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	short int *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyoverflowlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

/* Do appropriate processing given the current state.  */
/* Read a lookahead token if we need one and don't already have one.  */
/* yyresume: */

  /* First try to decide what to do without reference to lookahead token.  */

  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YYDSYMPRINTF ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Shift the lookahead token.  */
  YYDPRINTF ((stderr, "Shifting token %s, ", yytname[yytoken]));

  /* Discard the token being shifted unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  *++yyvsp = yylval;


  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  yystate = yyn;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 240 "pddl+.yacc"
    {top_thing= yyvsp[0].t_domain; current_analysis->the_domain= yyvsp[0].t_domain;;}
    break;

  case 3:
#line 241 "pddl+.yacc"
    {top_thing= yyvsp[0].t_problem; current_analysis->the_problem= yyvsp[0].t_problem;;}
    break;

  case 4:
#line 242 "pddl+.yacc"
    {top_thing= yyvsp[0].t_plan; ;}
    break;

  case 5:
#line 247 "pddl+.yacc"
    {yyval.t_domain= yyvsp[-1].t_domain; yyval.t_domain->name= yyvsp[-2].cp;delete [] yyvsp[-2].cp;;}
    break;

  case 6:
#line 249 "pddl+.yacc"
    {yyerrok; yyval.t_domain=static_cast<domain*>(NULL);
       	log_error(E_FATAL,"Syntax error in domain"); ;}
    break;

  case 7:
#line 255 "pddl+.yacc"
    {yyval.t_domain= yyvsp[0].t_domain; yyval.t_domain->req= yyvsp[-1].t_pddl_req_flag;;}
    break;

  case 8:
#line 256 "pddl+.yacc"
    {yyval.t_domain= yyvsp[0].t_domain; yyval.t_domain->types= yyvsp[-1].t_type_list;;}
    break;

  case 9:
#line 257 "pddl+.yacc"
    {yyval.t_domain= yyvsp[0].t_domain; yyval.t_domain->constants= yyvsp[-1].t_const_symbol_list;;}
    break;

  case 10:
#line 258 "pddl+.yacc"
    {yyval.t_domain= yyvsp[0].t_domain; 
                                       yyval.t_domain->predicates= yyvsp[-1].t_pred_decl_list; ;}
    break;

  case 11:
#line 260 "pddl+.yacc"
    {yyval.t_domain= yyvsp[0].t_domain; 
                                       yyval.t_domain->functions= yyvsp[-1].t_func_decl_list; ;}
    break;

  case 12:
#line 262 "pddl+.yacc"
    {yyval.t_domain= yyvsp[0].t_domain;
   										yyval.t_domain->constraints = yyvsp[-1].t_con_goal;;}
    break;

  case 13:
#line 264 "pddl+.yacc"
    {yyval.t_domain= new domain(yyvsp[0].t_structure_store); ;}
    break;

  case 14:
#line 267 "pddl+.yacc"
    {yyval.cp=yyvsp[-1].cp;;}
    break;

  case 15:
#line 273 "pddl+.yacc"
    {
	// Stash in analysis object --- we need to refer to it during parse
	//   but domain object is not created yet,
	current_analysis->req |= yyvsp[-1].t_pddl_req_flag;
	yyval.t_pddl_req_flag=yyvsp[-1].t_pddl_req_flag;
    ;}
    break;

  case 16:
#line 280 "pddl+.yacc"
    {yyerrok; 
       log_error(E_FATAL,"Syntax error in requirements declaration.");
       yyval.t_pddl_req_flag= 0; ;}
    break;

  case 17:
#line 286 "pddl+.yacc"
    { yyval.t_pddl_req_flag= yyvsp[-1].t_pddl_req_flag | yyvsp[0].t_pddl_req_flag; ;}
    break;

  case 18:
#line 287 "pddl+.yacc"
    { yyval.t_pddl_req_flag= 0; ;}
    break;

  case 19:
#line 293 "pddl+.yacc"
    {yyval.t_pred_decl_list=yyvsp[0].t_pred_decl_list; yyval.t_pred_decl_list->push_front(yyvsp[-1].t_pred_decl);;}
    break;

  case 20:
#line 295 "pddl+.yacc"
    {  yyval.t_pred_decl_list=new pred_decl_list;
           yyval.t_pred_decl_list->push_front(yyvsp[0].t_pred_decl); ;}
    break;

  case 21:
#line 300 "pddl+.yacc"
    {yyval.t_pred_decl= new pred_decl(yyvsp[-2].t_pred_symbol,yyvsp[-1].t_var_symbol_list,current_analysis->var_tab_stack.pop());;}
    break;

  case 22:
#line 302 "pddl+.yacc"
    {yyerrok; 
        // hope someone makes this error someday
        log_error(E_FATAL,"Syntax error in predicate declaration.");
	yyval.t_pred_decl= NULL; ;}
    break;

  case 23:
#line 310 "pddl+.yacc"
    { yyval.t_pred_symbol=current_analysis->pred_tab.symbol_put(yyvsp[0].cp);
           current_analysis->var_tab_stack.push(
           				current_analysis->buildPredTab());
           delete [] yyvsp[0].cp; ;}
    break;

  case 24:
#line 317 "pddl+.yacc"
    { yyval.t_pred_symbol=current_analysis->pred_tab.symbol_ref("="); 
	      requires(E_EQUALITY); ;}
    break;

  case 25:
#line 319 "pddl+.yacc"
    { yyval.t_pred_symbol=current_analysis->pred_tab.symbol_get(yyvsp[0].cp); delete [] yyvsp[0].cp; ;}
    break;

  case 26:
#line 327 "pddl+.yacc"
    { yyval.t_pred_symbol=current_analysis->pred_tab.symbol_get(yyvsp[0].cp); delete [] yyvsp[0].cp;;}
    break;

  case 27:
#line 333 "pddl+.yacc"
    {yyval.t_func_decl_list=yyvsp[-1].t_func_decl_list; yyval.t_func_decl_list->push_back(yyvsp[0].t_func_decl);;}
    break;

  case 28:
#line 334 "pddl+.yacc"
    { yyval.t_func_decl_list=new func_decl_list; ;}
    break;

  case 29:
#line 339 "pddl+.yacc"
    {yyval.t_func_decl= new func_decl(yyvsp[-2].t_func_symbol,yyvsp[-1].t_var_symbol_list,current_analysis->var_tab_stack.pop());;}
    break;

  case 30:
#line 341 "pddl+.yacc"
    {yyerrok; 
	 log_error(E_FATAL,"Syntax error in functor declaration.");
	 yyval.t_func_decl= NULL; ;}
    break;

  case 31:
#line 348 "pddl+.yacc"
    { yyval.t_func_symbol=current_analysis->func_tab.symbol_put(yyvsp[0].cp);
           current_analysis->var_tab_stack.push(
           		current_analysis->buildFuncTab()); 
           delete [] yyvsp[0].cp; ;}
    break;

  case 32:
#line 361 "pddl+.yacc"
    {  
      yyval.t_var_symbol_list= yyvsp[-3].t_var_symbol_list;
      yyval.t_var_symbol_list->set_types(yyvsp[-1].t_type);           /* Set types for variables */
      yyval.t_var_symbol_list->splice(yyval.t_var_symbol_list->end(),*yyvsp[0].t_var_symbol_list);   /* Join lists */ 
      delete yyvsp[0].t_var_symbol_list;                   /* Delete (now empty) list */
      requires(E_TYPING);
   ;}
    break;

  case 33:
#line 369 "pddl+.yacc"
    {  
      yyval.t_var_symbol_list= yyvsp[-3].t_var_symbol_list;
      yyval.t_var_symbol_list->set_either_types(yyvsp[-1].t_type_list);    /* Set types for variables */
      yyval.t_var_symbol_list->splice(yyval.t_var_symbol_list->end(),*yyvsp[0].t_var_symbol_list);   /* Join lists */ 
      delete yyvsp[0].t_var_symbol_list;                   /* Delete (now empty) list */
      requires(E_TYPING);
   ;}
    break;

  case 34:
#line 377 "pddl+.yacc"
    {
       yyval.t_var_symbol_list= yyvsp[0].t_var_symbol_list;
   ;}
    break;

  case 35:
#line 389 "pddl+.yacc"
    {yyval.t_var_symbol_list=yyvsp[0].t_var_symbol_list; yyvsp[0].t_var_symbol_list->push_front(yyvsp[-1].t_var_symbol); ;}
    break;

  case 36:
#line 390 "pddl+.yacc"
    {yyval.t_var_symbol_list= new var_symbol_list; ;}
    break;

  case 37:
#line 397 "pddl+.yacc"
    {  
      yyval.t_const_symbol_list= yyvsp[-3].t_const_symbol_list;
      yyvsp[-3].t_const_symbol_list->set_types(yyvsp[-1].t_type);           /* Set types for constants */
      yyvsp[-3].t_const_symbol_list->splice(yyvsp[-3].t_const_symbol_list->end(),*yyvsp[0].t_const_symbol_list); /* Join lists */ 
      delete yyvsp[0].t_const_symbol_list;                   /* Delete (now empty) list */
      requires(E_TYPING);
   ;}
    break;

  case 38:
#line 405 "pddl+.yacc"
    {  
      yyval.t_const_symbol_list= yyvsp[-3].t_const_symbol_list;
      yyvsp[-3].t_const_symbol_list->set_either_types(yyvsp[-1].t_type_list);
      yyvsp[-3].t_const_symbol_list->splice(yyvsp[-3].t_const_symbol_list->end(),*yyvsp[0].t_const_symbol_list);
      delete yyvsp[0].t_const_symbol_list;
      requires(E_TYPING);
   ;}
    break;

  case 39:
#line 413 "pddl+.yacc"
    {yyval.t_const_symbol_list= yyvsp[0].t_const_symbol_list;;}
    break;

  case 40:
#line 418 "pddl+.yacc"
    {yyval.t_const_symbol_list=yyvsp[0].t_const_symbol_list; yyvsp[0].t_const_symbol_list->push_front(yyvsp[-1].t_const_symbol);;}
    break;

  case 41:
#line 419 "pddl+.yacc"
    {yyval.t_const_symbol_list=new const_symbol_list;;}
    break;

  case 42:
#line 423 "pddl+.yacc"
    {yyval.t_const_symbol_list=yyvsp[0].t_const_symbol_list; yyvsp[0].t_const_symbol_list->push_front(yyvsp[-1].t_const_symbol);;}
    break;

  case 43:
#line 424 "pddl+.yacc"
    {yyval.t_const_symbol_list=new const_symbol_list;;}
    break;

  case 44:
#line 433 "pddl+.yacc"
    {  
       yyval.t_type_list= yyvsp[-3].t_type_list;
       yyval.t_type_list->set_types(yyvsp[-1].t_type);           /* Set types for constants */
       yyval.t_type_list->splice(yyval.t_type_list->end(),*yyvsp[0].t_type_list); /* Join lists */ 
       delete yyvsp[0].t_type_list;                   /* Delete (now empty) list */
   ;}
    break;

  case 45:
#line 440 "pddl+.yacc"
    {  
   // This parse needs to be excluded, we think (DPL&MF: 6/9/01)
       yyval.t_type_list= yyvsp[-3].t_type_list;
       yyval.t_type_list->set_either_types(yyvsp[-1].t_type_list);
       yyval.t_type_list->splice(yyvsp[-3].t_type_list->end(),*yyvsp[0].t_type_list);
       delete yyvsp[0].t_type_list;
   ;}
    break;

  case 46:
#line 449 "pddl+.yacc"
    { yyval.t_type_list= yyvsp[0].t_type_list; ;}
    break;

  case 47:
#line 455 "pddl+.yacc"
    {yyval.t_parameter_symbol_list=yyvsp[-1].t_parameter_symbol_list; yyval.t_parameter_symbol_list->push_back(yyvsp[0].t_const_symbol); ;}
    break;

  case 48:
#line 457 "pddl+.yacc"
    {yyval.t_parameter_symbol_list=yyvsp[-2].t_parameter_symbol_list; yyval.t_parameter_symbol_list->push_back(yyvsp[0].t_var_symbol); ;}
    break;

  case 49:
#line 458 "pddl+.yacc"
    {yyval.t_parameter_symbol_list= new parameter_symbol_list;;}
    break;

  case 50:
#line 465 "pddl+.yacc"
    { yyval.t_var_symbol= current_analysis->var_tab_stack.top()->symbol_put(yyvsp[0].cp); delete [] yyvsp[0].cp; ;}
    break;

  case 51:
#line 471 "pddl+.yacc"
    { yyval.t_var_symbol= current_analysis->var_tab_stack.symbol_get(yyvsp[0].cp); delete [] yyvsp[0].cp; ;}
    break;

  case 52:
#line 475 "pddl+.yacc"
    { yyval.t_const_symbol= current_analysis->const_tab.symbol_get(yyvsp[0].cp); delete [] yyvsp[0].cp; ;}
    break;

  case 53:
#line 479 "pddl+.yacc"
    { yyval.t_const_symbol= current_analysis->const_tab.symbol_put(yyvsp[0].cp); delete [] yyvsp[0].cp;;}
    break;

  case 54:
#line 484 "pddl+.yacc"
    { yyval.t_type_list= yyvsp[-1].t_type_list; ;}
    break;

  case 55:
#line 489 "pddl+.yacc"
    { yyval.t_type= current_analysis->pddl_type_tab.symbol_ref(yyvsp[0].cp); delete [] yyvsp[0].cp;;}
    break;

  case 56:
#line 496 "pddl+.yacc"
    { yyval.t_type= current_analysis->pddl_type_tab.symbol_ref(yyvsp[0].cp); delete [] yyvsp[0].cp;;}
    break;

  case 57:
#line 501 "pddl+.yacc"
    {yyval.t_type_list= yyvsp[-1].t_type_list; yyval.t_type_list->push_back(yyvsp[0].t_type);;}
    break;

  case 58:
#line 502 "pddl+.yacc"
    {yyval.t_type_list= new pddl_type_list;;}
    break;

  case 59:
#line 507 "pddl+.yacc"
    {yyval.t_type_list= yyvsp[-1].t_type_list; yyval.t_type_list->push_back(yyvsp[0].t_type);;}
    break;

  case 60:
#line 508 "pddl+.yacc"
    {yyval.t_type_list= new pddl_type_list;;}
    break;

  case 61:
#line 513 "pddl+.yacc"
    { yyval.t_effect_lists=yyvsp[-5].t_effect_lists;
	  yyval.t_effect_lists->assign_effects.push_back(new assignment(yyvsp[-2].t_func_term,E_ASSIGN,yyvsp[-1].t_num_expression));  
          requires(E_FLUENTS); 
	;}
    break;

  case 62:
#line 518 "pddl+.yacc"
    { yyval.t_effect_lists=yyvsp[-1].t_effect_lists; yyval.t_effect_lists->add_effects.push_back(yyvsp[0].t_simple_effect); ;}
    break;

  case 63:
#line 520 "pddl+.yacc"
    { yyval.t_effect_lists=yyvsp[-1].t_effect_lists; yyval.t_effect_lists->del_effects.push_back(yyvsp[0].t_simple_effect); ;}
    break;

  case 64:
#line 522 "pddl+.yacc"
    { yyval.t_effect_lists=yyvsp[-1].t_effect_lists; yyval.t_effect_lists->timed_effects.push_back(yyvsp[0].t_timed_effect); ;}
    break;

  case 65:
#line 524 "pddl+.yacc"
    { yyval.t_effect_lists= new effect_lists;;}
    break;

  case 66:
#line 529 "pddl+.yacc"
    { requires(E_TIMED_INITIAL_LITERALS); 
   		yyval.t_timed_effect=new timed_initial_literal(yyvsp[-1].t_effect_lists,yyvsp[-2].fval);;}
    break;

  case 67:
#line 534 "pddl+.yacc"
    {yyval.t_effect_lists=yyvsp[0].t_effect_lists; yyval.t_effect_lists->append_effects(yyvsp[-1].t_effect_lists); delete yyvsp[-1].t_effect_lists;;}
    break;

  case 68:
#line 535 "pddl+.yacc"
    {yyval.t_effect_lists=yyvsp[0].t_effect_lists; yyval.t_effect_lists->cond_effects.push_front(yyvsp[-1].t_cond_effect); 
                                      requires(E_COND_EFFS);;}
    break;

  case 69:
#line 537 "pddl+.yacc"
    {yyval.t_effect_lists=yyvsp[0].t_effect_lists; yyval.t_effect_lists->forall_effects.push_front(yyvsp[-1].t_forall_effect);
                                      requires(E_COND_EFFS);;}
    break;

  case 70:
#line 539 "pddl+.yacc"
    {yyval.t_effect_lists=new effect_lists(); ;}
    break;

  case 71:
#line 548 "pddl+.yacc"
    {yyval.t_effect_lists= yyvsp[0].t_effect_lists;;}
    break;

  case 72:
#line 549 "pddl+.yacc"
    {yyval.t_effect_lists=new effect_lists; yyval.t_effect_lists->add_effects.push_front(yyvsp[0].t_simple_effect);;}
    break;

  case 73:
#line 550 "pddl+.yacc"
    {yyval.t_effect_lists=new effect_lists; yyval.t_effect_lists->del_effects.push_front(yyvsp[0].t_simple_effect);;}
    break;

  case 74:
#line 551 "pddl+.yacc"
    {yyval.t_effect_lists=new effect_lists; yyval.t_effect_lists->cond_effects.push_front(yyvsp[0].t_cond_effect);;}
    break;

  case 75:
#line 552 "pddl+.yacc"
    {yyval.t_effect_lists=new effect_lists; yyval.t_effect_lists->forall_effects.push_front(yyvsp[0].t_forall_effect);;}
    break;

  case 76:
#line 556 "pddl+.yacc"
    {yyval.t_effect_lists= yyvsp[-1].t_effect_lists;;}
    break;

  case 77:
#line 557 "pddl+.yacc"
    {yyval.t_effect_lists= yyvsp[0].t_effect_lists;;}
    break;

  case 78:
#line 562 "pddl+.yacc"
    {yyval.t_effect_lists=new effect_lists; yyval.t_effect_lists->del_effects.push_front(yyvsp[0].t_simple_effect);;}
    break;

  case 79:
#line 564 "pddl+.yacc"
    {yyval.t_effect_lists=new effect_lists; yyval.t_effect_lists->add_effects.push_front(yyvsp[0].t_simple_effect);;}
    break;

  case 80:
#line 566 "pddl+.yacc"
    {yyval.t_effect_lists=new effect_lists; yyval.t_effect_lists->assign_effects.push_front(yyvsp[0].t_assignment);
         requires(E_FLUENTS);;}
    break;

  case 81:
#line 572 "pddl+.yacc"
    {yyval.t_effect_lists= yyvsp[-1].t_effect_lists; yyval.t_effect_lists->del_effects.push_back(yyvsp[0].t_simple_effect);;}
    break;

  case 82:
#line 573 "pddl+.yacc"
    {yyval.t_effect_lists= yyvsp[-1].t_effect_lists; yyval.t_effect_lists->add_effects.push_back(yyvsp[0].t_simple_effect);;}
    break;

  case 83:
#line 574 "pddl+.yacc"
    {yyval.t_effect_lists= yyvsp[-1].t_effect_lists; yyval.t_effect_lists->assign_effects.push_back(yyvsp[0].t_assignment);
                                     requires(E_FLUENTS); ;}
    break;

  case 84:
#line 576 "pddl+.yacc"
    { yyval.t_effect_lists= new effect_lists; ;}
    break;

  case 85:
#line 581 "pddl+.yacc"
    { yyval.t_effect_lists=yyvsp[-1].t_effect_lists; ;}
    break;

  case 86:
#line 583 "pddl+.yacc"
    {yyerrok; yyval.t_effect_lists=NULL;
	 log_error(E_FATAL,"Syntax error in (and ...)");
	;}
    break;

  case 87:
#line 591 "pddl+.yacc"
    { yyval.t_effect_lists=yyvsp[-1].t_effect_lists; ;}
    break;

  case 88:
#line 596 "pddl+.yacc"
    { yyval.t_effect_lists= new effect_lists; 
          yyval.t_effect_lists->forall_effects.push_back(
	       new forall_effect(yyvsp[-1].t_effect_lists, yyvsp[-3].t_var_symbol_list, current_analysis->var_tab_stack.pop())); 
          requires(E_COND_EFFS);;}
    break;

  case 89:
#line 601 "pddl+.yacc"
    { yyval.t_effect_lists= new effect_lists;
	  yyval.t_effect_lists->cond_effects.push_back(
	       new cond_effect(yyvsp[-2].t_goal,yyvsp[-1].t_effect_lists));
          requires(E_COND_EFFS); ;}
    break;

  case 90:
#line 606 "pddl+.yacc"
    { yyval.t_effect_lists=new effect_lists;
          yyval.t_effect_lists->timed_effects.push_back(yyvsp[0].t_timed_effect); ;}
    break;

  case 91:
#line 609 "pddl+.yacc"
    { yyval.t_effect_lists= new effect_lists;
	  yyval.t_effect_lists->assign_effects.push_front(yyvsp[0].t_assignment);
          requires(E_FLUENTS); ;}
    break;

  case 92:
#line 615 "pddl+.yacc"
    { yyval.t_effect_lists=yyvsp[-1].t_effect_lists; yyvsp[-1].t_effect_lists->append_effects(yyvsp[0].t_effect_lists); delete yyvsp[0].t_effect_lists; ;}
    break;

  case 93:
#line 616 "pddl+.yacc"
    { yyval.t_effect_lists= new effect_lists; ;}
    break;

  case 94:
#line 621 "pddl+.yacc"
    {yyval.t_timed_effect=new timed_effect(yyvsp[-1].t_effect_lists,E_AT_START);;}
    break;

  case 95:
#line 623 "pddl+.yacc"
    {yyval.t_timed_effect=new timed_effect(yyvsp[-1].t_effect_lists,E_AT_END);;}
    break;

  case 96:
#line 625 "pddl+.yacc"
    {yyval.t_timed_effect=new timed_effect(new effect_lists,E_CONTINUOUS);
         yyval.t_timed_effect->effs->assign_effects.push_front(
	     new assignment(yyvsp[-2].t_func_term,E_INCREASE,yyvsp[-1].t_expression)); ;}
    break;

  case 97:
#line 629 "pddl+.yacc"
    {yyval.t_timed_effect=new timed_effect(new effect_lists,E_CONTINUOUS);
         yyval.t_timed_effect->effs->assign_effects.push_front(
	     new assignment(yyvsp[-2].t_func_term,E_DECREASE,yyvsp[-1].t_expression)); ;}
    break;

  case 98:
#line 633 "pddl+.yacc"
    {yyerrok; yyval.t_timed_effect=NULL;
	log_error(E_FATAL,"Syntax error in timed effect"); ;}
    break;

  case 99:
#line 639 "pddl+.yacc"
    {yyval.t_effect_lists= yyvsp[-1].t_effect_lists;;}
    break;

  case 100:
#line 640 "pddl+.yacc"
    {yyval.t_effect_lists= yyvsp[0].t_effect_lists;;}
    break;

  case 101:
#line 645 "pddl+.yacc"
    {yyval.t_effect_lists=new effect_lists; yyval.t_effect_lists->del_effects.push_front(yyvsp[0].t_simple_effect);;}
    break;

  case 102:
#line 647 "pddl+.yacc"
    {yyval.t_effect_lists=new effect_lists; yyval.t_effect_lists->add_effects.push_front(yyvsp[0].t_simple_effect);;}
    break;

  case 103:
#line 649 "pddl+.yacc"
    {yyval.t_effect_lists=new effect_lists; yyval.t_effect_lists->assign_effects.push_front(yyvsp[0].t_assignment);
         requires(E_FLUENTS);;}
    break;

  case 104:
#line 655 "pddl+.yacc"
    {yyval.t_effect_lists= yyvsp[-1].t_effect_lists; yyval.t_effect_lists->del_effects.push_back(yyvsp[0].t_simple_effect);;}
    break;

  case 105:
#line 656 "pddl+.yacc"
    {yyval.t_effect_lists= yyvsp[-1].t_effect_lists; yyval.t_effect_lists->add_effects.push_back(yyvsp[0].t_simple_effect);;}
    break;

  case 106:
#line 657 "pddl+.yacc"
    {yyval.t_effect_lists= yyvsp[-1].t_effect_lists; yyval.t_effect_lists->assign_effects.push_back(yyvsp[0].t_assignment);
                                     requires(E_FLUENTS); ;}
    break;

  case 107:
#line 659 "pddl+.yacc"
    { yyval.t_effect_lists= new effect_lists; ;}
    break;

  case 108:
#line 665 "pddl+.yacc"
    { yyval.t_assignment= new assignment(yyvsp[-2].t_func_term,E_ASSIGN,yyvsp[-1].t_expression); ;}
    break;

  case 109:
#line 667 "pddl+.yacc"
    { yyval.t_assignment= new assignment(yyvsp[-2].t_func_term,E_INCREASE,yyvsp[-1].t_expression); ;}
    break;

  case 110:
#line 669 "pddl+.yacc"
    { yyval.t_assignment= new assignment(yyvsp[-2].t_func_term,E_DECREASE,yyvsp[-1].t_expression); ;}
    break;

  case 111:
#line 671 "pddl+.yacc"
    { yyval.t_assignment= new assignment(yyvsp[-2].t_func_term,E_SCALE_UP,yyvsp[-1].t_expression); ;}
    break;

  case 112:
#line 673 "pddl+.yacc"
    { yyval.t_assignment= new assignment(yyvsp[-2].t_func_term,E_SCALE_DOWN,yyvsp[-1].t_expression); ;}
    break;

  case 113:
#line 678 "pddl+.yacc"
    {yyval.t_effect_lists=new effect_lists; 
         timed_effect * te = new timed_effect(new effect_lists,E_CONTINUOUS);
         yyval.t_effect_lists->timed_effects.push_front(te);
         te->effs->assign_effects.push_front(
	     new assignment(yyvsp[-2].t_func_term,E_INCREASE,yyvsp[-1].t_expression)); ;}
    break;

  case 114:
#line 684 "pddl+.yacc"
    {yyval.t_effect_lists=new effect_lists; 
         timed_effect * te = new timed_effect(new effect_lists,E_CONTINUOUS);
         yyval.t_effect_lists->timed_effects.push_front(te);
         te->effs->assign_effects.push_front(
	     new assignment(yyvsp[-2].t_func_term,E_DECREASE,yyvsp[-1].t_expression)); ;}
    break;

  case 115:
#line 690 "pddl+.yacc"
    {yyval.t_effect_lists = yyvsp[-1].t_effect_lists;;}
    break;

  case 116:
#line 694 "pddl+.yacc"
    { yyval.t_effect_lists=yyvsp[-1].t_effect_lists; yyvsp[-1].t_effect_lists->append_effects(yyvsp[0].t_effect_lists); delete yyvsp[0].t_effect_lists; ;}
    break;

  case 117:
#line 695 "pddl+.yacc"
    { yyval.t_effect_lists= new effect_lists; ;}
    break;

  case 118:
#line 699 "pddl+.yacc"
    {yyval.t_expression= yyvsp[0].t_expression;;}
    break;

  case 119:
#line 700 "pddl+.yacc"
    {yyval.t_expression= new special_val_expr(E_DURATION_VAR);
                    requires( E_DURATION_INEQUALITIES );;}
    break;

  case 120:
#line 702 "pddl+.yacc"
    { yyval.t_expression=yyvsp[0].t_num_expression; ;}
    break;

  case 121:
#line 703 "pddl+.yacc"
    { yyval.t_expression= yyvsp[0].t_func_term; ;}
    break;

  case 122:
#line 708 "pddl+.yacc"
    { yyval.t_expression= new plus_expression(yyvsp[-2].t_expression,yyvsp[-1].t_expression); ;}
    break;

  case 123:
#line 710 "pddl+.yacc"
    { yyval.t_expression= new minus_expression(yyvsp[-2].t_expression,yyvsp[-1].t_expression); ;}
    break;

  case 124:
#line 712 "pddl+.yacc"
    { yyval.t_expression= new mul_expression(yyvsp[-2].t_expression,yyvsp[-1].t_expression); ;}
    break;

  case 125:
#line 714 "pddl+.yacc"
    { yyval.t_expression= new div_expression(yyvsp[-2].t_expression,yyvsp[-1].t_expression); ;}
    break;

  case 126:
#line 719 "pddl+.yacc"
    { yyval.t_goal= new conj_goal(yyvsp[-1].t_goal_list); ;}
    break;

  case 127:
#line 721 "pddl+.yacc"
    { yyval.t_goal= new timed_goal(new comparison(yyvsp[-4].t_comparison_op,
        			new special_val_expr(E_DURATION_VAR),yyvsp[-1].t_expression),E_AT_START); ;}
    break;

  case 128:
#line 724 "pddl+.yacc"
    { yyval.t_goal = new timed_goal(new comparison(yyvsp[-5].t_comparison_op,
					new special_val_expr(E_DURATION_VAR),yyvsp[-2].t_expression),E_AT_START);;}
    break;

  case 129:
#line 727 "pddl+.yacc"
    { yyval.t_goal = new timed_goal(new comparison(yyvsp[-5].t_comparison_op,
					new special_val_expr(E_DURATION_VAR),yyvsp[-2].t_expression),E_AT_END);;}
    break;

  case 130:
#line 732 "pddl+.yacc"
    {yyval.t_comparison_op= E_LESSEQ; requires(E_DURATION_INEQUALITIES);;}
    break;

  case 131:
#line 733 "pddl+.yacc"
    {yyval.t_comparison_op= E_GREATEQ; requires(E_DURATION_INEQUALITIES);;}
    break;

  case 132:
#line 734 "pddl+.yacc"
    {yyval.t_comparison_op= E_EQUALS; ;}
    break;

  case 133:
#line 742 "pddl+.yacc"
    {yyval.t_expression= yyvsp[0].t_expression; ;}
    break;

  case 134:
#line 747 "pddl+.yacc"
    { yyval.t_goal_list=yyvsp[-1].t_goal_list; yyval.t_goal_list->push_back(yyvsp[0].t_goal); ;}
    break;

  case 135:
#line 749 "pddl+.yacc"
    { yyval.t_goal_list= new goal_list; ;}
    break;

  case 136:
#line 754 "pddl+.yacc"
    { yyval.t_simple_effect= new simple_effect(yyvsp[-1].t_proposition); ;}
    break;

  case 137:
#line 759 "pddl+.yacc"
    { yyval.t_simple_effect= new simple_effect(yyvsp[0].t_proposition); ;}
    break;

  case 138:
#line 766 "pddl+.yacc"
    { yyval.t_simple_effect= new simple_effect(yyvsp[-1].t_proposition); ;}
    break;

  case 139:
#line 771 "pddl+.yacc"
    { yyval.t_simple_effect= new simple_effect(yyvsp[0].t_proposition); ;}
    break;

  case 140:
#line 776 "pddl+.yacc"
    { yyval.t_forall_effect= new forall_effect(yyvsp[-1].t_effect_lists, yyvsp[-3].t_var_symbol_list, current_analysis->var_tab_stack.pop());;}
    break;

  case 141:
#line 781 "pddl+.yacc"
    { yyval.t_cond_effect= new cond_effect(yyvsp[-2].t_goal,yyvsp[-1].t_effect_lists); ;}
    break;

  case 142:
#line 786 "pddl+.yacc"
    { yyval.t_assignment= new assignment(yyvsp[-2].t_func_term,E_ASSIGN,yyvsp[-1].t_expression); ;}
    break;

  case 143:
#line 788 "pddl+.yacc"
    { yyval.t_assignment= new assignment(yyvsp[-2].t_func_term,E_INCREASE,yyvsp[-1].t_expression); ;}
    break;

  case 144:
#line 790 "pddl+.yacc"
    { yyval.t_assignment= new assignment(yyvsp[-2].t_func_term,E_DECREASE,yyvsp[-1].t_expression); ;}
    break;

  case 145:
#line 792 "pddl+.yacc"
    { yyval.t_assignment= new assignment(yyvsp[-2].t_func_term,E_SCALE_UP,yyvsp[-1].t_expression); ;}
    break;

  case 146:
#line 794 "pddl+.yacc"
    { yyval.t_assignment= new assignment(yyvsp[-2].t_func_term,E_SCALE_DOWN,yyvsp[-1].t_expression); ;}
    break;

  case 147:
#line 799 "pddl+.yacc"
    { yyval.t_expression= new uminus_expression(yyvsp[-1].t_expression); requires(E_FLUENTS); ;}
    break;

  case 148:
#line 801 "pddl+.yacc"
    { yyval.t_expression= new plus_expression(yyvsp[-2].t_expression,yyvsp[-1].t_expression); requires(E_FLUENTS); ;}
    break;

  case 149:
#line 803 "pddl+.yacc"
    { yyval.t_expression= new minus_expression(yyvsp[-2].t_expression,yyvsp[-1].t_expression); requires(E_FLUENTS); ;}
    break;

  case 150:
#line 805 "pddl+.yacc"
    { yyval.t_expression= new mul_expression(yyvsp[-2].t_expression,yyvsp[-1].t_expression); requires(E_FLUENTS); ;}
    break;

  case 151:
#line 807 "pddl+.yacc"
    { yyval.t_expression= new div_expression(yyvsp[-2].t_expression,yyvsp[-1].t_expression); requires(E_FLUENTS); ;}
    break;

  case 152:
#line 808 "pddl+.yacc"
    { yyval.t_expression=yyvsp[0].t_num_expression; ;}
    break;

  case 153:
#line 809 "pddl+.yacc"
    { yyval.t_expression= yyvsp[0].t_func_term; requires(E_FLUENTS); ;}
    break;

  case 154:
#line 814 "pddl+.yacc"
    { yyval.t_expression= new mul_expression(new special_val_expr(E_HASHT),yyvsp[-1].t_expression); ;}
    break;

  case 155:
#line 816 "pddl+.yacc"
    { yyval.t_expression= new mul_expression(yyvsp[-2].t_expression, new special_val_expr(E_HASHT)); ;}
    break;

  case 156:
#line 818 "pddl+.yacc"
    { yyval.t_expression= new special_val_expr(E_HASHT); ;}
    break;

  case 157:
#line 823 "pddl+.yacc"
    { yyval.t_num_expression=new int_expression(yyvsp[0].ival);   ;}
    break;

  case 158:
#line 824 "pddl+.yacc"
    { yyval.t_num_expression=new float_expression(yyvsp[0].fval); ;}
    break;

  case 159:
#line 828 "pddl+.yacc"
    { yyval.t_func_term=new func_term( current_analysis->func_tab.symbol_get(yyvsp[-2].cp), yyvsp[-1].t_parameter_symbol_list); delete [] yyvsp[-2].cp; ;}
    break;

  case 160:
#line 831 "pddl+.yacc"
    { yyval.t_func_term=new func_term( current_analysis->func_tab.symbol_get(yyvsp[-2].cp), yyvsp[-1].t_parameter_symbol_list); delete [] yyvsp[-2].cp; ;}
    break;

  case 161:
#line 833 "pddl+.yacc"
    { yyval.t_func_term=new func_term( current_analysis->func_tab.symbol_get(yyvsp[0].cp),
                            new parameter_symbol_list); delete [] yyvsp[0].cp;;}
    break;

  case 162:
#line 851 "pddl+.yacc"
    { yyval.t_func_term=new func_term( current_analysis->func_tab.symbol_get(yyvsp[-2].cp), yyvsp[-1].t_parameter_symbol_list); delete [] yyvsp[-2].cp; ;}
    break;

  case 163:
#line 853 "pddl+.yacc"
    { yyval.t_func_term=new func_term( current_analysis->func_tab.symbol_get(yyvsp[-2].cp), yyvsp[-1].t_parameter_symbol_list); delete [] yyvsp[-2].cp; ;}
    break;

  case 164:
#line 855 "pddl+.yacc"
    { yyval.t_func_term=new func_term( current_analysis->func_tab.symbol_get(yyvsp[0].cp),
                            new parameter_symbol_list); delete [] yyvsp[0].cp;;}
    break;

  case 165:
#line 860 "pddl+.yacc"
    { yyval.t_comparison_op= E_GREATER; ;}
    break;

  case 166:
#line 861 "pddl+.yacc"
    { yyval.t_comparison_op= E_GREATEQ; ;}
    break;

  case 167:
#line 862 "pddl+.yacc"
    { yyval.t_comparison_op= E_LESS; ;}
    break;

  case 168:
#line 863 "pddl+.yacc"
    { yyval.t_comparison_op= E_LESSEQ; ;}
    break;

  case 169:
#line 864 "pddl+.yacc"
    { yyval.t_comparison_op= E_EQUALS; ;}
    break;

  case 170:
#line 877 "pddl+.yacc"
    {yyval.t_goal= yyvsp[0].t_goal;;}
    break;

  case 171:
#line 879 "pddl+.yacc"
    {yyval.t_goal = new conj_goal(yyvsp[-1].t_goal_list);;}
    break;

  case 172:
#line 882 "pddl+.yacc"
    {yyval.t_goal= new qfied_goal(E_FORALL,yyvsp[-3].t_var_symbol_list,yyvsp[-1].t_goal,current_analysis->var_tab_stack.pop());
        requires(E_UNIV_PRECS);;}
    break;

  case 173:
#line 888 "pddl+.yacc"
    {yyval.t_con_goal = new preference(yyvsp[-1].t_con_goal);requires(E_PREFERENCES);;}
    break;

  case 174:
#line 890 "pddl+.yacc"
    {yyval.t_con_goal = new preference(yyvsp[-2].cp,yyvsp[-1].t_con_goal);requires(E_PREFERENCES);;}
    break;

  case 175:
#line 892 "pddl+.yacc"
    {yyval.t_con_goal = new conj_goal(yyvsp[-1].t_goal_list);;}
    break;

  case 176:
#line 895 "pddl+.yacc"
    {yyval.t_con_goal= new qfied_goal(E_FORALL,yyvsp[-3].t_var_symbol_list,yyvsp[-1].t_con_goal,current_analysis->var_tab_stack.pop());
                requires(E_UNIV_PRECS);;}
    break;

  case 177:
#line 898 "pddl+.yacc"
    {yyval.t_con_goal = yyvsp[0].t_con_goal;;}
    break;

  case 178:
#line 903 "pddl+.yacc"
    {yyval.t_goal_list=yyvsp[-1].t_goal_list; yyvsp[-1].t_goal_list->push_back(yyvsp[0].t_con_goal);;}
    break;

  case 179:
#line 905 "pddl+.yacc"
    {yyval.t_goal_list= new goal_list;;}
    break;

  case 180:
#line 910 "pddl+.yacc"
    {yyval.t_goal= new preference(yyvsp[-1].t_goal); requires(E_PREFERENCES);;}
    break;

  case 181:
#line 912 "pddl+.yacc"
    {yyval.t_goal= new preference(yyvsp[-2].cp,yyvsp[-1].t_goal); requires(E_PREFERENCES);;}
    break;

  case 182:
#line 914 "pddl+.yacc"
    {yyval.t_goal=yyvsp[0].t_goal;;}
    break;

  case 183:
#line 919 "pddl+.yacc"
    {yyval.t_goal_list = yyvsp[-1].t_goal_list; yyval.t_goal_list->push_back(yyvsp[0].t_con_goal);;}
    break;

  case 184:
#line 921 "pddl+.yacc"
    {yyval.t_goal_list = new goal_list;;}
    break;

  case 185:
#line 926 "pddl+.yacc"
    {yyval.t_con_goal= new conj_goal(yyvsp[-1].t_goal_list);;}
    break;

  case 186:
#line 928 "pddl+.yacc"
    {yyval.t_con_goal = new qfied_goal(E_FORALL,yyvsp[-3].t_var_symbol_list,yyvsp[-1].t_con_goal,current_analysis->var_tab_stack.pop());
        requires(E_UNIV_PRECS);;}
    break;

  case 187:
#line 931 "pddl+.yacc"
    {yyval.t_con_goal = new constraint_goal(E_ATEND,yyvsp[-1].t_goal);;}
    break;

  case 188:
#line 933 "pddl+.yacc"
    {yyval.t_con_goal = new constraint_goal(E_ALWAYS,yyvsp[-1].t_goal);;}
    break;

  case 189:
#line 935 "pddl+.yacc"
    {yyval.t_con_goal = new constraint_goal(E_SOMETIME,yyvsp[-1].t_goal);;}
    break;

  case 190:
#line 937 "pddl+.yacc"
    {yyval.t_con_goal = new constraint_goal(E_WITHIN,yyvsp[-1].t_goal,NULL,yyvsp[-2].t_num_expression->double_value(),0.0);delete yyvsp[-2].t_num_expression;;}
    break;

  case 191:
#line 939 "pddl+.yacc"
    {yyval.t_con_goal = new constraint_goal(E_ATMOSTONCE,yyvsp[-1].t_goal);;}
    break;

  case 192:
#line 941 "pddl+.yacc"
    {yyval.t_con_goal = new constraint_goal(E_SOMETIMEAFTER,yyvsp[-1].t_goal,yyvsp[-2].t_goal);;}
    break;

  case 193:
#line 943 "pddl+.yacc"
    {yyval.t_con_goal = new constraint_goal(E_SOMETIMEBEFORE,yyvsp[-1].t_goal,yyvsp[-2].t_goal);;}
    break;

  case 194:
#line 945 "pddl+.yacc"
    {yyval.t_con_goal = new constraint_goal(E_ALWAYSWITHIN,yyvsp[-1].t_goal,yyvsp[-2].t_goal,yyvsp[-3].t_num_expression->double_value(),0.0);delete yyvsp[-3].t_num_expression;;}
    break;

  case 195:
#line 947 "pddl+.yacc"
    {yyval.t_con_goal = new constraint_goal(E_HOLDDURING,yyvsp[-1].t_goal,NULL,yyvsp[-2].t_num_expression->double_value(),yyvsp[-3].t_num_expression->double_value());delete yyvsp[-3].t_num_expression;delete yyvsp[-2].t_num_expression;;}
    break;

  case 196:
#line 949 "pddl+.yacc"
    {yyval.t_con_goal = new constraint_goal(E_HOLDAFTER,yyvsp[-1].t_goal,NULL,0.0,yyvsp[-2].t_num_expression->double_value());delete yyvsp[-2].t_num_expression;;}
    break;

  case 197:
#line 954 "pddl+.yacc"
    {yyval.t_goal= new simple_goal(yyvsp[0].t_proposition,E_POS);;}
    break;

  case 198:
#line 956 "pddl+.yacc"
    {yyval.t_goal= new neg_goal(yyvsp[-1].t_goal);simple_goal * s = dynamic_cast<simple_goal *>(yyvsp[-1].t_goal);
       if(s && s->getProp()->head->getName()=="=") {requires(E_EQUALITY);} 
       else{requires(E_NEGATIVE_PRECONDITIONS);};;}
    break;

  case 199:
#line 960 "pddl+.yacc"
    {yyval.t_goal= new conj_goal(yyvsp[-1].t_goal_list);;}
    break;

  case 200:
#line 962 "pddl+.yacc"
    {yyval.t_goal= new disj_goal(yyvsp[-1].t_goal_list);
        requires(E_DISJUNCTIVE_PRECONDS);;}
    break;

  case 201:
#line 965 "pddl+.yacc"
    {yyval.t_goal= new imply_goal(yyvsp[-2].t_goal,yyvsp[-1].t_goal);
        requires(E_DISJUNCTIVE_PRECONDS);;}
    break;

  case 202:
#line 969 "pddl+.yacc"
    {yyval.t_goal= new qfied_goal(yyvsp[-5].t_quantifier,yyvsp[-3].t_var_symbol_list,yyvsp[-1].t_goal,current_analysis->var_tab_stack.pop());;}
    break;

  case 203:
#line 971 "pddl+.yacc"
    {yyval.t_goal= new comparison(yyvsp[-3].t_comparison_op,yyvsp[-2].t_expression,yyvsp[-1].t_expression); 
        requires(E_FLUENTS);;}
    break;

  case 204:
#line 977 "pddl+.yacc"
    {yyval.t_goal_list=yyvsp[-1].t_goal_list; yyvsp[-1].t_goal_list->push_back(yyvsp[0].t_goal);;}
    break;

  case 205:
#line 979 "pddl+.yacc"
    {yyval.t_goal_list= new goal_list;;}
    break;

  case 206:
#line 984 "pddl+.yacc"
    {yyval.t_goal_list=yyvsp[-1].t_goal_list; yyvsp[-1].t_goal_list->push_back(yyvsp[0].t_goal);;}
    break;

  case 207:
#line 986 "pddl+.yacc"
    {yyval.t_goal_list= new goal_list;;}
    break;

  case 208:
#line 990 "pddl+.yacc"
    {yyval.t_quantifier=yyvsp[0].t_quantifier;;}
    break;

  case 209:
#line 991 "pddl+.yacc"
    {yyval.t_quantifier=yyvsp[0].t_quantifier;;}
    break;

  case 210:
#line 996 "pddl+.yacc"
    {yyval.t_quantifier=E_FORALL; 
        current_analysis->var_tab_stack.push(
        		current_analysis->buildForallTab());;}
    break;

  case 211:
#line 1003 "pddl+.yacc"
    {yyval.t_quantifier=E_EXISTS;
        current_analysis->var_tab_stack.push(
        	current_analysis->buildExistsTab());;}
    break;

  case 212:
#line 1010 "pddl+.yacc"
    {yyval.t_proposition=new proposition(yyvsp[-2].t_pred_symbol,yyvsp[-1].t_parameter_symbol_list);;}
    break;

  case 213:
#line 1015 "pddl+.yacc"
    {yyval.t_proposition = new proposition(yyvsp[-2].t_pred_symbol,yyvsp[-1].t_var_symbol_list);;}
    break;

  case 214:
#line 1020 "pddl+.yacc"
    {yyval.t_proposition=new proposition(yyvsp[-2].t_pred_symbol,yyvsp[-1].t_parameter_symbol_list);;}
    break;

  case 215:
#line 1025 "pddl+.yacc"
    {yyval.t_pred_decl_list= yyvsp[-1].t_pred_decl_list;;}
    break;

  case 216:
#line 1027 "pddl+.yacc"
    {yyerrok; yyval.t_pred_decl_list=NULL;
	 log_error(E_FATAL,"Syntax error in (:predicates ...)");
	;}
    break;

  case 217:
#line 1034 "pddl+.yacc"
    {yyval.t_func_decl_list= yyvsp[-1].t_func_decl_list;;}
    break;

  case 218:
#line 1036 "pddl+.yacc"
    {yyerrok; yyval.t_func_decl_list=NULL;
	 log_error(E_FATAL,"Syntax error in (:functions ...)");
	;}
    break;

  case 219:
#line 1043 "pddl+.yacc"
    {yyval.t_con_goal = yyvsp[-1].t_con_goal;;}
    break;

  case 220:
#line 1045 "pddl+.yacc"
    {yyerrok; yyval.t_con_goal=NULL;
      log_error(E_FATAL,"Syntax error in (:constraints ...)");
      ;}
    break;

  case 221:
#line 1052 "pddl+.yacc"
    {yyval.t_con_goal = yyvsp[-1].t_con_goal;;}
    break;

  case 222:
#line 1054 "pddl+.yacc"
    {yyerrok; yyval.t_con_goal=NULL;
      log_error(E_FATAL,"Syntax error in (:constraints ...)");
      ;}
    break;

  case 223:
#line 1060 "pddl+.yacc"
    { yyval.t_structure_store=yyvsp[-1].t_structure_store; yyval.t_structure_store->push_back(yyvsp[0].t_structure_def); ;}
    break;

  case 224:
#line 1061 "pddl+.yacc"
    { yyval.t_structure_store= new structure_store; yyval.t_structure_store->push_back(yyvsp[0].t_structure_def); ;}
    break;

  case 225:
#line 1065 "pddl+.yacc"
    { yyval.t_structure_def= yyvsp[0].t_action_def; ;}
    break;

  case 226:
#line 1066 "pddl+.yacc"
    { yyval.t_structure_def= yyvsp[0].t_event_def; requires(E_TIME); ;}
    break;

  case 227:
#line 1067 "pddl+.yacc"
    { yyval.t_structure_def= yyvsp[0].t_process_def; requires(E_TIME); ;}
    break;

  case 228:
#line 1068 "pddl+.yacc"
    { yyval.t_structure_def= yyvsp[0].t_durative_action_def; requires(E_DURATIVE_ACTIONS); ;}
    break;

  case 229:
#line 1069 "pddl+.yacc"
    { yyval.t_structure_def= yyvsp[0].t_derivation_rule; requires(E_DERIVED_PREDICATES);;}
    break;

  case 230:
#line 1073 "pddl+.yacc"
    {yyval.t_dummy= 0; 
    	current_analysis->var_tab_stack.push(
    					current_analysis->buildRuleTab());;}
    break;

  case 231:
#line 1084 "pddl+.yacc"
    {yyval.t_derivation_rule = new derivation_rule(yyvsp[-2].t_proposition,yyvsp[-1].t_goal,current_analysis->var_tab_stack.pop());;}
    break;

  case 232:
#line 1096 "pddl+.yacc"
    { yyval.t_action_def= current_analysis->buildAction(current_analysis->op_tab.symbol_put(yyvsp[-9].cp),
			yyvsp[-6].t_var_symbol_list,yyvsp[-3].t_goal,yyvsp[-1].t_effect_lists,
			current_analysis->var_tab_stack.pop()); delete [] yyvsp[-9].cp; ;}
    break;

  case 233:
#line 1100 "pddl+.yacc"
    {yyerrok; 
	 log_error(E_FATAL,"Syntax error in action declaration.");
	 yyval.t_action_def= NULL; ;}
    break;

  case 234:
#line 1113 "pddl+.yacc"
    {yyval.t_event_def= current_analysis->buildEvent(current_analysis->op_tab.symbol_put(yyvsp[-9].cp),
		   yyvsp[-6].t_var_symbol_list,yyvsp[-3].t_goal,yyvsp[-1].t_effect_lists,
		   current_analysis->var_tab_stack.pop()); delete [] yyvsp[-9].cp;;}
    break;

  case 235:
#line 1118 "pddl+.yacc"
    {yyerrok; 
	 log_error(E_FATAL,"Syntax error in event declaration.");
	 yyval.t_event_def= NULL; ;}
    break;

  case 236:
#line 1130 "pddl+.yacc"
    {yyval.t_process_def= current_analysis->buildProcess(current_analysis->op_tab.symbol_put(yyvsp[-9].cp),
		     yyvsp[-6].t_var_symbol_list,yyvsp[-3].t_goal,yyvsp[-1].t_effect_lists,
                     current_analysis->var_tab_stack.pop()); delete [] yyvsp[-9].cp;;}
    break;

  case 237:
#line 1134 "pddl+.yacc"
    {yyerrok; 
	 log_error(E_FATAL,"Syntax error in process declaration.");
	 yyval.t_process_def= NULL; ;}
    break;

  case 238:
#line 1146 "pddl+.yacc"
    { yyval.t_durative_action_def= yyvsp[-1].t_durative_action_def;
      yyval.t_durative_action_def->name= current_analysis->op_tab.symbol_put(yyvsp[-8].cp);
      yyval.t_durative_action_def->symtab= current_analysis->var_tab_stack.pop();
      yyval.t_durative_action_def->parameters= yyvsp[-5].t_var_symbol_list;
      yyval.t_durative_action_def->dur_constraint= yyvsp[-2].t_goal; 
      delete [] yyvsp[-8].cp;
    ;}
    break;

  case 239:
#line 1155 "pddl+.yacc"
    {yyerrok; 
	 log_error(E_FATAL,"Syntax error in durative-action declaration.");
	 yyval.t_durative_action_def= NULL; ;}
    break;

  case 240:
#line 1162 "pddl+.yacc"
    {yyval.t_durative_action_def=yyvsp[-2].t_durative_action_def; yyval.t_durative_action_def->effects=yyvsp[0].t_effect_lists;;}
    break;

  case 241:
#line 1164 "pddl+.yacc"
    {yyval.t_durative_action_def=yyvsp[-2].t_durative_action_def; yyval.t_durative_action_def->precondition=yyvsp[0].t_goal;;}
    break;

  case 242:
#line 1165 "pddl+.yacc"
    {yyval.t_durative_action_def= current_analysis->buildDurativeAction();;}
    break;

  case 243:
#line 1170 "pddl+.yacc"
    { yyval.t_goal=yyvsp[0].t_goal; ;}
    break;

  case 244:
#line 1172 "pddl+.yacc"
    { yyval.t_goal= new conj_goal(yyvsp[-1].t_goal_list); ;}
    break;

  case 245:
#line 1177 "pddl+.yacc"
    { yyval.t_goal_list=yyvsp[-1].t_goal_list; yyval.t_goal_list->push_back(yyvsp[0].t_goal); ;}
    break;

  case 246:
#line 1179 "pddl+.yacc"
    { yyval.t_goal_list= new goal_list; ;}
    break;

  case 247:
#line 1184 "pddl+.yacc"
    {yyval.t_goal= new timed_goal(yyvsp[-1].t_goal,E_AT_START);;}
    break;

  case 248:
#line 1186 "pddl+.yacc"
    {yyval.t_goal= new timed_goal(yyvsp[-1].t_goal,E_AT_END);;}
    break;

  case 249:
#line 1188 "pddl+.yacc"
    {yyval.t_goal= new timed_goal(yyvsp[-1].t_goal,E_OVER_ALL);;}
    break;

  case 250:
#line 1190 "pddl+.yacc"
    {timed_goal * tg = dynamic_cast<timed_goal *>(yyvsp[-1].t_goal);
		yyval.t_goal = new timed_goal(new preference(yyvsp[-2].cp,tg->clearGoal()),tg->getTime());
			delete tg;
			requires(E_PREFERENCES);;}
    break;

  case 251:
#line 1195 "pddl+.yacc"
    {yyval.t_goal = new preference(yyvsp[-1].t_goal);requires(E_PREFERENCES);;}
    break;

  case 252:
#line 1199 "pddl+.yacc"
    {yyval.t_dummy= 0; current_analysis->var_tab_stack.push(
    				current_analysis->buildOpTab());;}
    break;

  case 253:
#line 1204 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_EQUALITY;;}
    break;

  case 254:
#line 1205 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_STRIPS;;}
    break;

  case 255:
#line 1207 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_TYPING;;}
    break;

  case 256:
#line 1209 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_NEGATIVE_PRECONDITIONS;;}
    break;

  case 257:
#line 1211 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_DISJUNCTIVE_PRECONDS;;}
    break;

  case 258:
#line 1212 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_EXT_PRECS;;}
    break;

  case 259:
#line 1213 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_UNIV_PRECS;;}
    break;

  case 260:
#line 1214 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_COND_EFFS;;}
    break;

  case 261:
#line 1215 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_FLUENTS;;}
    break;

  case 262:
#line 1217 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_DURATIVE_ACTIONS;;}
    break;

  case 263:
#line 1218 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_TIME |
                      E_FLUENTS |
                      E_DURATIVE_ACTIONS; ;}
    break;

  case 264:
#line 1222 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_STRIPS |
		      E_TYPING | 
		      E_NEGATIVE_PRECONDITIONS |
		      E_DISJUNCTIVE_PRECONDS |
		      E_EQUALITY |
		      E_EXT_PRECS |
		      E_UNIV_PRECS |
		      E_COND_EFFS;;}
    break;

  case 265:
#line 1231 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_EXT_PRECS |
		      E_UNIV_PRECS;;}
    break;

  case 266:
#line 1235 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_DURATION_INEQUALITIES;;}
    break;

  case 267:
#line 1238 "pddl+.yacc"
    {yyval.t_pddl_req_flag= E_CONTINUOUS_EFFECTS;;}
    break;

  case 268:
#line 1240 "pddl+.yacc"
    {yyval.t_pddl_req_flag = E_DERIVED_PREDICATES;;}
    break;

  case 269:
#line 1242 "pddl+.yacc"
    {yyval.t_pddl_req_flag = E_TIMED_INITIAL_LITERALS;;}
    break;

  case 270:
#line 1244 "pddl+.yacc"
    {yyval.t_pddl_req_flag = E_PREFERENCES;;}
    break;

  case 271:
#line 1246 "pddl+.yacc"
    {yyval.t_pddl_req_flag = E_CONSTRAINTS;;}
    break;

  case 272:
#line 1248 "pddl+.yacc"
    {log_error(E_WARNING,"Unrecognised requirements declaration ");
       yyval.t_pddl_req_flag= 0; delete [] yyvsp[0].cp;;}
    break;

  case 273:
#line 1254 "pddl+.yacc"
    {yyval.t_const_symbol_list=yyvsp[-1].t_const_symbol_list;;}
    break;

  case 274:
#line 1258 "pddl+.yacc"
    {yyval.t_type_list=yyvsp[-1].t_type_list; requires(E_TYPING);;}
    break;

  case 275:
#line 1268 "pddl+.yacc"
    {yyval.t_problem=yyvsp[-1].t_problem; yyval.t_problem->name = yyvsp[-7].cp; yyval.t_problem->domain_name = yyvsp[-3].cp;;}
    break;

  case 276:
#line 1270 "pddl+.yacc"
    {yyerrok; yyval.t_problem=NULL;
       	log_error(E_FATAL,"Syntax error in problem definition."); ;}
    break;

  case 277:
#line 1276 "pddl+.yacc"
    {yyval.t_problem=yyvsp[0].t_problem; yyval.t_problem->req= yyvsp[-1].t_pddl_req_flag;;}
    break;

  case 278:
#line 1277 "pddl+.yacc"
    {yyval.t_problem=yyvsp[0].t_problem; yyval.t_problem->objects= yyvsp[-1].t_const_symbol_list;;}
    break;

  case 279:
#line 1278 "pddl+.yacc"
    {yyval.t_problem=yyvsp[0].t_problem; yyval.t_problem->initial_state= yyvsp[-1].t_effect_lists;;}
    break;

  case 280:
#line 1279 "pddl+.yacc"
    {yyval.t_problem=yyvsp[0].t_problem; yyval.t_problem->the_goal= yyvsp[-1].t_goal;;}
    break;

  case 281:
#line 1281 "pddl+.yacc"
    {yyval.t_problem=yyvsp[0].t_problem; yyval.t_problem->constraints = yyvsp[-1].t_con_goal;;}
    break;

  case 282:
#line 1282 "pddl+.yacc"
    {yyval.t_problem=yyvsp[0].t_problem; yyval.t_problem->metric= yyvsp[-1].t_metric;;}
    break;

  case 283:
#line 1283 "pddl+.yacc"
    {yyval.t_problem=yyvsp[0].t_problem; yyval.t_problem->length= yyvsp[-1].t_length_spec;;}
    break;

  case 284:
#line 1284 "pddl+.yacc"
    {yyval.t_problem=new problem;;}
    break;

  case 285:
#line 1287 "pddl+.yacc"
    {yyval.t_const_symbol_list=yyvsp[-1].t_const_symbol_list;;}
    break;

  case 286:
#line 1290 "pddl+.yacc"
    {yyval.t_effect_lists=yyvsp[-1].t_effect_lists;;}
    break;

  case 287:
#line 1293 "pddl+.yacc"
    {yyval.vtab = current_analysis->buildOpTab();;}
    break;

  case 288:
#line 1296 "pddl+.yacc"
    {yyval.t_goal=yyvsp[-1].t_goal;delete yyvsp[-2].vtab;;}
    break;

  case 289:
#line 1301 "pddl+.yacc"
    { yyval.t_metric= new metric_spec(yyvsp[-2].t_optimization,yyvsp[-1].t_expression); ;}
    break;

  case 290:
#line 1303 "pddl+.yacc"
    {yyerrok; 
        log_error(E_FATAL,"Syntax error in metric declaration.");
        yyval.t_metric= NULL; ;}
    break;

  case 291:
#line 1310 "pddl+.yacc"
    {yyval.t_length_spec= yyvsp[-1].t_length_spec;;}
    break;

  case 292:
#line 1314 "pddl+.yacc"
    {yyval.t_length_spec= new length_spec(E_SERIAL,yyvsp[0].ival);;}
    break;

  case 293:
#line 1315 "pddl+.yacc"
    {yyval.t_length_spec= new length_spec(E_PARALLEL,yyvsp[-3].ival);;}
    break;

  case 294:
#line 1319 "pddl+.yacc"
    {yyval.t_optimization= E_MINIMIZE;;}
    break;

  case 295:
#line 1320 "pddl+.yacc"
    {yyval.t_optimization= E_MAXIMIZE;;}
    break;

  case 296:
#line 1325 "pddl+.yacc"
    {yyval.t_expression= yyvsp[-1].t_expression;;}
    break;

  case 297:
#line 1326 "pddl+.yacc"
    {yyval.t_expression= yyvsp[0].t_func_term;;}
    break;

  case 298:
#line 1327 "pddl+.yacc"
    {yyval.t_expression= yyvsp[0].t_num_expression;;}
    break;

  case 299:
#line 1328 "pddl+.yacc"
    { yyval.t_expression= new special_val_expr(E_TOTAL_TIME); ;}
    break;

  case 300:
#line 1330 "pddl+.yacc"
    {yyval.t_expression = new violation_term(yyvsp[-1].cp);;}
    break;

  case 301:
#line 1331 "pddl+.yacc"
    { yyval.t_expression= new special_val_expr(E_TOTAL_TIME); ;}
    break;

  case 302:
#line 1335 "pddl+.yacc"
    { yyval.t_expression= new plus_expression(yyvsp[-1].t_expression,yyvsp[0].t_expression); ;}
    break;

  case 303:
#line 1336 "pddl+.yacc"
    { yyval.t_expression= new minus_expression(yyvsp[-1].t_expression,yyvsp[0].t_expression); ;}
    break;

  case 304:
#line 1337 "pddl+.yacc"
    { yyval.t_expression= new mul_expression(yyvsp[-1].t_expression,yyvsp[0].t_expression); ;}
    break;

  case 305:
#line 1338 "pddl+.yacc"
    { yyval.t_expression= new div_expression(yyvsp[-1].t_expression,yyvsp[0].t_expression); ;}
    break;

  case 306:
#line 1342 "pddl+.yacc"
    {yyval.t_expression = yyvsp[0].t_expression;;}
    break;

  case 307:
#line 1344 "pddl+.yacc"
    {yyval.t_expression = new plus_expression(yyvsp[-1].t_expression,yyvsp[0].t_expression);;}
    break;

  case 308:
#line 1348 "pddl+.yacc"
    {yyval.t_expression = yyvsp[0].t_expression;;}
    break;

  case 309:
#line 1350 "pddl+.yacc"
    {yyval.t_expression = new mul_expression(yyvsp[-1].t_expression,yyvsp[0].t_expression);;}
    break;

  case 310:
#line 1356 "pddl+.yacc"
    {yyval.t_plan= yyvsp[0].t_plan; 
         yyval.t_plan->push_front(yyvsp[-1].t_step); ;}
    break;

  case 311:
#line 1359 "pddl+.yacc"
    {yyval.t_plan = yyvsp[0].t_plan;yyval.t_plan->insertTime(yyvsp[-1].fval);;}
    break;

  case 312:
#line 1361 "pddl+.yacc"
    {yyval.t_plan = yyvsp[0].t_plan;yyval.t_plan->insertTime(yyvsp[-1].ival);;}
    break;

  case 313:
#line 1363 "pddl+.yacc"
    {yyval.t_plan= new plan;;}
    break;

  case 314:
#line 1368 "pddl+.yacc"
    {yyval.t_step=yyvsp[0].t_step; 
         yyval.t_step->start_time_given=1; 
         yyval.t_step->start_time=yyvsp[-2].fval;;}
    break;

  case 315:
#line 1372 "pddl+.yacc"
    {yyval.t_step=yyvsp[0].t_step;
	 yyval.t_step->start_time_given=0;;}
    break;

  case 316:
#line 1378 "pddl+.yacc"
    {yyval.t_step= yyvsp[-3].t_step; 
	 yyval.t_step->duration_given=1;
         yyval.t_step->duration= yyvsp[-1].fval;;}
    break;

  case 317:
#line 1382 "pddl+.yacc"
    {yyval.t_step= yyvsp[0].t_step;
         yyval.t_step->duration_given=0;;}
    break;

  case 318:
#line 1388 "pddl+.yacc"
    {yyval.t_step= new plan_step( 
              current_analysis->op_tab.symbol_get(yyvsp[-2].cp), 
	      yyvsp[-1].t_const_symbol_list); delete [] yyvsp[-2].cp;
      ;}
    break;

  case 319:
#line 1395 "pddl+.yacc"
    {yyval.fval= yyvsp[0].fval;;}
    break;

  case 320:
#line 1396 "pddl+.yacc"
    {yyval.fval= (float) yyvsp[0].ival;;}
    break;


    }

/* Line 1010 of yacc.c.  */
#line 3786 "pddl+.cpp"

  yyvsp -= yylen;
  yyssp -= yylen;


  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if YYERROR_VERBOSE
      yyn = yypact[yystate];

      if (YYPACT_NINF < yyn && yyn < YYLAST)
	{
	  YYSIZE_T yysize = 0;
	  int yytype = YYTRANSLATE (yychar);
	  const char* yyprefix;
	  char *yymsg;
	  int yyx;

	  /* Start YYX at -YYN if negative to avoid negative indexes in
	     YYCHECK.  */
	  int yyxbegin = yyn < 0 ? -yyn : 0;

	  /* Stay within bounds of both yycheck and yytname.  */
	  int yychecklim = YYLAST - yyn;
	  int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
	  int yycount = 0;

	  yyprefix = ", expecting ";
	  for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	      {
		yysize += yystrlen (yyprefix) + yystrlen (yytname [yyx]);
		yycount += 1;
		if (yycount == 5)
		  {
		    yysize = 0;
		    break;
		  }
	      }
	  yysize += (sizeof ("syntax error, unexpected ")
		     + yystrlen (yytname[yytype]));
	  yymsg = (char *) YYSTACK_ALLOC (yysize);
	  if (yymsg != 0)
	    {
	      char *yyp = yystpcpy (yymsg, "syntax error, unexpected ");
	      yyp = yystpcpy (yyp, yytname[yytype]);

	      if (yycount < 5)
		{
		  yyprefix = ", expecting ";
		  for (yyx = yyxbegin; yyx < yyxend; ++yyx)
		    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
		      {
			yyp = yystpcpy (yyp, yyprefix);
			yyp = yystpcpy (yyp, yytname[yyx]);
			yyprefix = " or ";
		      }
		}
	      yyerror (yymsg);
	      YYSTACK_FREE (yymsg);
	    }
	  else
	    yyerror ("syntax error; also virtual memory exhausted");
	}
      else
#endif /* YYERROR_VERBOSE */
	yyerror ("syntax error");
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* If at end of input, pop the error token,
	     then the rest of the stack, then return failure.  */
	  if (yychar == YYEOF)
	     for (;;)
	       {
		 YYPOPSTACK;
		 if (yyssp == yyss)
		   YYABORT;
		 YYDSYMPRINTF ("Error: popping", yystos[*yyssp], yyvsp, yylsp);
		 yydestruct (yystos[*yyssp], yyvsp);
	       }
        }
      else
	{
	  YYDSYMPRINTF ("Error: discarding", yytoken, &yylval, &yylloc);
	  yydestruct (yytoken, &yylval);
	  yychar = YYEMPTY;

	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

#ifdef __GNUC__
  /* Pacify GCC when the user code never invokes YYERROR and the label
     yyerrorlab therefore never appears in user code.  */
  if (0)
     goto yyerrorlab;
#endif

  yyvsp -= yylen;
  yyssp -= yylen;
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;

      YYDSYMPRINTF ("Error: popping", yystos[*yyssp], yyvsp, yylsp);
      yydestruct (yystos[yystate], yyvsp);
      YYPOPSTACK;
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  YYDPRINTF ((stderr, "Shifting error token, "));

  *++yyvsp = yylval;


  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*----------------------------------------------.
| yyoverflowlab -- parser overflow comes here.  |
`----------------------------------------------*/
yyoverflowlab:
  yyerror ("parser stack overflow");
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  return yyresult;
}


#line 1399 "pddl+.yacc"


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


