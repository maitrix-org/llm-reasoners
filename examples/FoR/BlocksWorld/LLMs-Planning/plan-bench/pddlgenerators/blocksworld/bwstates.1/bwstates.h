#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

/*
* The constant SZ is the maximum number of blocks allowed in a state.
* Of course, smaller states are obtained by using the -s option on the
* command line. For states larger than the given value of SZ, increase
* it, but beware that the memory requirements will then increase fast.
*
* SZZ is the length of half an array of dimension SZxSZ.
*
* Note that most implementations of the C language (for instance on
* SUNs) require that the pre-allocated memory fit into some space of
* a specific maximum size. This means that it will be impossible to
* declare an array of type bigarray (see below) if SZ is large. In order
* to allow large states to be generated, we have used an explicit malloc
* in main(). Please adjust the constant SZ to be large enough for the
* problems you desire but no larger.
*/

#ifndef SZ
#define SZ 1000
#endif
#define SZZ (((SZ+2)*(SZ+2)+3)/4)

typedef float bigarray[SZZ];

typedef struct {   /* A tower is distinguished by its top and bottom blocks */
  int top;
  int bottom;
} TOWER, *tower;

typedef struct {            /* A partial state of Blocks World */
  int N;                    /* The number of blocks */
  int S[SZ+1];              /* The ON relation considered as a function */
  TOWER rooted[SZ+1];       /* The towers which are on the table */
  TOWER floating[SZ+1];     /* The towers which are not yet on anything */
  int nrt, nft;             /* The numbers of rooted and floating towers */
} STATE, *state;


void get_options(int argc, char *argv[], int *size, int *states, long *seed);
int pos(int N, int x, int y);
void make_ratio(int N, float ratio[]);
void make_state(state sigma, float ratio[]);
void print_state(state sigma);
double drand48();
