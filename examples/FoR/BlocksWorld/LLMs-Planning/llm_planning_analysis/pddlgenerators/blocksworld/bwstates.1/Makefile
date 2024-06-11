# Uncomment WARNINGS to suppress warning messages during compilation.

# WARNINGS	= -w

##########################################################################
# DESTINATION DIRECTORIES, GROUPS AND UIDS                               #
##########################################################################

# BEWARE! SVR4 install works quite differently from Berkeley install!
# in particular the destination directory is given by the "-f" option.
# Using the form "install object directory" will do something quite
# different from what you are used to!!!
#
INSTALL		= /usr/sbin/install 
# where the binaries go
BIN		= ./
# the root of man directories
MANROOT		= /opt/local/man
# manual section
MANSECT		= 1

##########################################################################
# Set the constant SIZE to the maximum number of blocks that you will    #
# need in any state generated. The memory requirement is quadratic in    #
# SIZE, so it is not a good idea to make this very large in the site-    #
# wide version of BWSTATES. If a particular user wants to generate very  #
# large problems and has a large enough machine to cope, it is best to   #
# make a special BWSTATES just for that purpose.                         #
##########################################################################

SIZE		= 200

##########################################################################
# If you have no gcc or if you wish to compile BWSTATES with another     #
# compiler, change the next line to suit yourself.                       #
##########################################################################

CC		= gcc

##########################################################################
# You should not have to change the stuff below                          #
##########################################################################

MAXBLOCKS	= -DSZ=$(SIZE)
CFLAGS		= -O2 $(WARNINGS) $(MAXBLOCKS)
MANDIR		= $(MANROOT)/man$(MANSECT)
CATDIR		= $(MANROOT)/cat$(MANSECT)
CC = gcc

##########################################################################
# Under Solaris the following math library should work. Under BSD Unix   #
# it is probably sufficient just to use -lm instead.                     #
##########################################################################

# MATHLIB = /opt/SUNWspro/SC2.0.1/libm.a
MATHLIB = -lm


all: bwstates

bwstates: bwstates.o
	$(CC) $(CFLAGS) bwstates.o -o bwstates $(MATHLIB)

README: README.ms
	nroff -ms README.ms > README

install_bwstates: bwstates
	$(INSTALL) -f $(BIN) -m 755 bwstates

install_man:
	$(INSTALL) -d $(MANDIR)
	cp bwstates.man $(MANDIR)/bwstates.$(MANSECT)
	chmod 644 $(MANDIR)/bwstates.$(MANSECT)
	
install:
	make install_bwstates
	make install_man

clean:
	rm -f *.o core *.o bwstates README

deinstall:
	rm -f $(BIN)/bwstates
	rm -f $(MANDIR)/bwstates.$(MANSECT)*
	rm -f $(CATDIR)/bwstates.$(MANSECT)*


bwstates.o: bwstates.h

