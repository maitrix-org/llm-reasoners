.nr PO 1.0i
.SH
Welcome to BWSTATES 1.0
.PP
You should find the following files in this directory:
.DS
Makefile
README.ms
bwstates.c
bwstates.h
bwstates.man
doc/ (a directory)
.DE
.PP
The directory ./doc contains the .dvi and .ps files of the full
documentation:
.DS
bwstates.dvi
bwstates.ps
.DE
.PP
BWSTATES is a program which generates random sequences of
states of the classical planning domain Blocks World. It was
written by John Slaney of the Automated Reasoning Project,
The Research School of Information Sciences and Engineering,
The Australian National University.
.SH
Installation
.PP
Before compilation and installation edit the Makefile in
the main directory of bwstates. Generally, only the top
few lines of the Makefile require editing. BIN
defines where bwstates binaries will go. The manual will be
installed in $(MANROOT)/man$(MANSECT) as bwstates.$(MANSECT).
The constant SZ gives the size
(number of blocks) of the largest problem that bwstates can
generate. If it is set too high, your machine may not have
enough memory to cope, so experiment with it.
.PP
To generate random states, bwstates uses the math library
functions drand48(), srand48, nint and floor. Under Solaris
this means that /opt/SUNWspro/SC2.0.1/libm.a has to be linked.
Under BSD Unix, a simple \-lm should suffice. Uncomment as
required.
.PP
Once you are happy with the Makefile, type
.DS
make 
.DE
in order to compile and link bwstates. If everything goes
without glitches, type
.DS
make install
.DE
to install bwstates, the help files and the manual pages.
.PP
To clean up the source directory type
.DS
make clean
.DE
.PP
If at some stage you would like to remove BWSTATES from your system
change back to this directory and type
.DS
make deinstall
.DE
This will automatically remove BWSTATES and all its works.
.PP
.SH
Author
.PP
BWSTATES was written by John Slaney. It can be obtained via anonymous
ftp from arp.anu.oz.au or via snail-mail from 
.DS
Automated Reasoning Project,
The Research School of Information Sciences and Engineering,
The Australian National University,
Canberra, A.C.T. 0200,
Australia
.DE
BWSTATES problems and queries should be referred to 
.DS
gustav@arp.anu.oz.au, or
jks@arp.anu.oz.au
.DE
.SH

