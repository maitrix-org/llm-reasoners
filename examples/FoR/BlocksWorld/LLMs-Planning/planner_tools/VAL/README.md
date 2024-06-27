VAL
===

This repository contains the current version of the VAL plan validation code. 

Compilation under Linux using g++ should be straightforward: use "make validate", "make parser" etc. The repository also contains a ".cbp" file which is a CodeBlocks project file. Using this, the code is set up to offer the targets listed below for compilation using mingw under Windows. We have a version of the VAL code that compiles with Visual Studio, but have not completed the merging with this repository. Windows executables are in bin/validate, bin/parser etc.

The main difficulties we have experienced in the past in compiling tend to be in the flex/bison code. To avoid that, the code in this repository contains pddl+.cpp, which is the generated source, and does not require to be regenerated from the lex and yacc source files in src/Parser. 



There are multiple targets, but the ones intended for general use are:

parser
validate
tan

These are: the PDDL parser, the VALidator and a type-analyser. The use for the first and last of these is straightforward:

parser <domainfile> <?problemfile>

(Problem file is optional).

tan <domainfile> <problemfile>

Note that the parser will find and report errors in PDDL more explicitly than VAL. The type-checking tool, tan, is reasonably robust at finding type errors in your PDDL domain/problem files.

VAL has many command line options, but the most important first few are:

validate -t <number> -v <domainfile> <problemfile> <planfile....>

Multiple plan files can be handled together. The -t flag allows the value of epsilon to be set. The default value is 0.01, but 0.001 is a good value to use for most planners. Actions separated by epsilon or less are treated as simultaneous by VAL. -v is the verbose flag. 

Another useful flag is the -l flag, which causes VAL to generate a LaTeX report, and -f <file> sets the file destination for this (the .tex extension is automatically added, so need not be placed on the command line).

