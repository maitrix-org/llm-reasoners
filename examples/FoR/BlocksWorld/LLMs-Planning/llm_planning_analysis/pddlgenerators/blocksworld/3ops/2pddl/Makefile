#!/bin/sh



####### FLAGS

TYPE	= 
ADDONS	= 

CC      = gcc

CFLAGS	= -O6 -Wall -g -ansi $(TYPE) $(ADDONS) 
# -g -pg

LIBS    = -lm


SOURCES 	= 2pddl.c

OBJECTS 	= $(SOURCES:.c=.o)

####### Implicit rules

.SUFFIXES:

.SUFFIXES: .c .o

.c.o:; $(CC) -c $(CFLAGS) $<

####### Build rules


miconic: $(OBJECTS)
	$(CC) -o 2pddl $(OBJECTS) $(CFLAGS) $(LIBS)

# misc
clean:
	rm -f *.o *.bak *~ *% core *_pure_p9_c0_400.o.warnings \
        \#*\# $(RES_PARSER_SRC) $(PDDL_PARSER_SRC)

veryclean: clean
	rm -f  2pddl H* J* K* L* O* graph.* *.symbex gmon.out \
	$(PDDL_PARSER_SRC) \
	lex.fct_pddl.c lex.ops_pddl.c lex.probname.c \
	*.output

lint:
	lclint -booltype Bool $(SOURCES) 2> output.lint

# DO NOT DELETE
