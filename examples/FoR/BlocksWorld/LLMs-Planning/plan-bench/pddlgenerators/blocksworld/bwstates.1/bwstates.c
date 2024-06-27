/*
* This program generates random states of Blocks World (BW) suitable for
* giving a supply of random BW planning problems. Note that the states
* generated are complete (that is, complete specifications of which block
* is on which) and are pseudo-random within the class of such states of
* the same number of blocks. The number of blocks (not including the table)
* is supplied as the argument to this program, along with the number of
* states required.
*
* The output is in the form of a list of intgers. The blocks are to be
* thought of as numbered from 1 to N, with the table as number 0. Each
* state is specified by giving first N and then for each block in sequence
* the number of the block it is on. A notional size of zero terminates the
* output. For example, the state of 6 blocks
*
*     2
*     5    3
*     4    1    6
*    -------------
*
* is represented by the integers
*    6
*    0 5 1 0 4 0
* meaning there are 6 blocks; 1 is on the table, 2 on 5, 3 on 1, 4 on the
* table, 5 on 4 and 6 on the table.
*
* Parameters are given to this program on the command line. The options are:
*
*    -n <integer>     number of blocks in each state (default 0)
*    -s <integer>     number of states required (default 1)
*    -r <integer>     seed for the random number generator (default 3088)
*
* The random numbers are supplied by the C library function drand48(). To
* ensure that every BW state of the given size has the same probability of
* being generated, it is necessary to calculate the probability at every
* step that the next block to be placed should go on the table. The numbers
* required for this are calculated in advance, which is why this program
* pauses for a while before generating any states.
*
* As supplied, the maximum number of blocks is 1000. This figure may be
* changed by redefining the constant SZ in the header. However, as the
* memory requirement is quadratic in SZ, significantly increasing the size
* may be a bad idea unless your machine has adequate memory.
*/


#include "bwstates.h"



int main(argc,argv)
int argc;
char *argv[];
{
    int S;                /* Number of states required */
    state sigma;          /* Structure holding the state */
    float *ratio;         /* Array of ratios. See note in BW.h */
    long seed;            /* Seed for drand48() */
    int x;
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);

    ratio = (float*)malloc(sizeof(bigarray));
    sigma = (state)malloc(sizeof(STATE));
    sigma->N = 0;                                   /* Default */
    S = 1;                                          /* Default */
    seed = tv.tv_usec;                              /* Default */
    get_options(argc,argv,&(sigma->N),&S,&seed);    /* Read command line */
    make_ratio(sigma->N,ratio);                     /* Get probabilities */
    srand48(seed);                                  /* Initialise drand48() */

    while (S--) {                                   /* Loop S times */
	make_state(sigma,ratio);
	print_state(sigma);
    }
    printf("0\n");                                  /* EOF marker */
}


/*
* Read the command line and check that it is sensible
*/

void get_options(int argc, char *argv[], int *size, int *nstates, long *seed)
{
    int option;

    while ((option = getopt (argc, argv, "n:r:s:")) != -1)
	switch( option ) {
	  case 'n': sscanf(optarg,"%d",size);
	    break;
	  case 'r': sscanf(optarg,"%ld",seed);
	    break;
	  case 's': sscanf(optarg,"%d",nstates);
	    break;
	}
    if (size<0 || *size >SZ) {
	fprintf(stderr, "Bad -n option: size %d out of range\n\n", *size);
	exit(2);
    }
    if (*nstates<0) {
	fprintf(stderr, "Bad -s option: negative number of states\n\n");
	exit(2);
    }
}


/*
* The 2-dimensional array of ratios is represented in one dimension, so here
* is an index function such that ratio[pos(x,y)] is essentially ratio[2x][y].
*/

int pos(int N, int x, int y)
{
    return ((x*(N+2-x)) + y);
}


/*
* Let g(n,k) be the number of states that extend a part-state with k
* towers already on the table and n floating towers not yet on anything.
* We work with Ratio(...n,k) which is g(n,k+1)/g(n,k).

* The ratio is stored in the case of an even-numbered row, and calculated
* in the case of an odd-numbered row. This is simply to halve the space
* required to store ratios. Note that N is the number of blocks.
*/

float Ratio(float ratio[], int N, int x, int y)
{
    int z;

    z = pos(N,x/2,y);
    if (x%2) return (ratio[z+1]+x+y) / (((1/ratio[z])*(x+y-1))+1);
    else return ratio[z];
}



/*
* This function is called during initialisation.
*
* The function g is easily defined recursively:
*   g(0,k) = 1
*   g(n+1,k) = g(n,k)(n + k) + g(n,k+1)
*
* This determines the required ratio.
* Let g(n-1,k) = a. Let g(n-1,k+1)/a = R. Let g(n-1,k+2)/g(n-1,k+1) = S.
* Then we have:
*   g(n,k+1) / g(n,k)
* = (Ra(n+k) + SRa) / (a(n+k-1) + Ra)
* = R(n+k+Sa) / (n+k-1+R)
* = (n+k+Sa) / ((n+k-1)/R + 1)
* Either of the last two expressions may be used conveniently to calculate
* the ratio for (n,k) given those for (n-1,k) and (n-1,k+1).
*/

void make_ratio(int N, float ratio[])
{
  int n,k;
  double temp[SZ+1];

  for (k=0; k<=N; k++) temp[k] = 1.0;

  for (n=0; n<=N; n++)
      for (k=0; k+n<=N; k++) {
	  if ( !n ) ratio[pos(N,n,k)] = 1.0;
	  else {
	      temp[k] = (temp[k] * (temp[k+1]+n+k)) / (temp[k]+n+k-1.0);
	      if (!(n%2)) ratio[pos(N,n/2,k)] = temp[k];
	  }
      }
}



/*
* To make the state, begin by regarding the blocks as short floating towers,
* and repeatedly take the last one and put ikt on something. It may go on
* the table, in which case the array of grounded or rooted towers is
* extended by one, or it may go on another (floating or rooted) tower. All
* destinations except for the table have equal probability.
*/

void make_state(state sigma, float ratio[])
{
    int x;
    float r;         /* The randomly generated number */
    float rat;       /* The relevant ratio from the array */
    float p;         /* The probability that the block goes on the table */
    int choice;      /* Abbreviates (n + k) */
    int b;           /* The destination block */

    for (x=0; x<sigma->N; x++) {
	sigma->rooted[x].top = sigma->rooted[x].bottom = -1;
	sigma->floating[x].top = sigma->floating[x].bottom = x;
	sigma->S[x] = -1;
    }                 /* Initially, each block is a floating tower */
    sigma->nrt = 0;
    sigma->nft = sigma->N;

    while (sigma->nft--) {
	r = drand48();
	choice = sigma->nft + sigma->nrt;
	rat = Ratio(ratio,sigma->N,sigma->nft,sigma->nrt);
	p = rat / (rat + choice);
	if (r <= p) {                 /* Put the next block on the table */
	    sigma->rooted[sigma->nrt].top = sigma->floating[sigma->nft].top;
	    sigma->rooted[sigma->nrt].bottom =
		sigma->floating[sigma->nft].bottom;
	    sigma->nrt++;
	}
	else {                        /* Put the next block on some b */
	    b = (int) (floor((r - p) / ((1.0 - p) / choice)));
	    if (b < sigma->nrt) {     /* Destination is a rooted tower */
		sigma->S[sigma->floating[sigma->nft].bottom] =
		    sigma->rooted[b].top;
		sigma->rooted[b].top = sigma->floating[sigma->nft].top;
	    }
	    else {                    /* Destination is a floating tower */
		b -= sigma->nrt;
		sigma->S[sigma->floating[sigma->nft].bottom] =
		    sigma->floating[b].top;
		sigma->floating[b].top = sigma->floating[sigma->nft].top;
      }
    }
  }
}


/*
* Print the size (number of blocks) and list the S function. Note that for
* output purposes, the first block will be #1 although the array actually
* starts at 0, so we have to add one to everything.
*/

void print_state(state sigma)
{
  int x;

  printf(" %d\n",sigma->N);
  for (x=0; x<sigma->N; x++)
    printf(" %d", sigma->S[x]+1);
  printf("\n");
}

