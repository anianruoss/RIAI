#
#
#  This source file is part of ELINA (ETH LIbrary for Numerical Analysis).
#  ELINA is Copyright © 2018 Department of Computer Science, ETH Zurich
#  This software is distributed under GNU Lesser General Public License Version 3.0.
#  For more information, see the ELINA project website at:
#  http://elina.ethz.ch
#
#  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER
#  EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO ANY WARRANTY
#  THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS OR BE ERROR-FREE AND ANY
#  IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
#  TITLE, OR NON-INFRINGEMENT.  IN NO EVENT SHALL ETH ZURICH BE LIABLE FOR ANY     
#  DAMAGES, INCLUDING BUT NOT LIMITED TO DIRECT, INDIRECT,
#  SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN
#  ANY WAY CONNECTED WITH THIS SOFTWARE (WHETHER OR NOT BASED UPON WARRANTY,
#  CONTRACT, TORT OR OTHERWISE).
#
#



include ../Makefile.config

LIBDIR = $(ELINA_PREFIX)/lib

INCLDIR = $(ELINA_PREFIX)/include 

INSTALL = install
INSTALLd = install -d

OBJS = elina_box_internal.o elina_box_representation.o elina_box_constructor.o elina_box_assign.o elina_box_meetjoin.o elina_box_resize.o 

ifeq ($(IS_APRON),)
LIBS = -L../partitions_api -lpartitions -L../elina_auxiliary -lelinaux -L../elina_linearize -lelinalinearize $(MPFR_LIB_FLAG) -lmpfr $(GMP_LIB_FLAG) -lgmp -lm
INCLUDES = -I../elina_auxiliary -I../elina_linearize -I../partitions_api $(MPFR_INCLUDE_FLAG) $(GMP_INCLUDE_FLAG)
else
LIBS = -L../partitions_api -lpartitions -L$(APRON_PREFIX)/lib -lapron  -L../elina_linearize -lelinalinearize $(MPFR_LIB_FLAG) -lmpfr $(GMP_LIB_FLAG) -lgmp -lm
INCLUDES = -I../apron_interface -I$(APRON_PREFIX)/include -I../elina_linearize -I../partitions_api $(MPFR_INCLUDE_FLAG) $(GMP_INCLUDE_FLAG)
endif

INSTALL = install
INSTALLd = install -d

SOINST = libelinabox.so

ELINABOXH = elina_box.h

all : libelinabox.so

libelinabox.so : $(OBJS) $(ELINABOXH)
	$(CC) -shared $(CC_ELINA_DYLIB) $(CFLAGS) $(DFLAGS) $(INCLUDES) -o $(SOINST) $(OBJS) $(LIBS)

elina_box_internal.o : elina_box_internal.h elina_box_internal.c
	$(CC) -c $(CFLAGS) $(DFLAGS) $(INCLUDES) -o elina_box_internal.o elina_box_internal.c $(LIBS)

elina_box_representation.o : elina_box_representation.h elina_box_representation.c
	$(CC) -c $(CFLAGS) $(DFLAGS) $(INCLUDES) -o elina_box_representation.o elina_box_representation.c $(LIBS)

elina_box_constructor.o : elina_box_constructor.h elina_box_constructor.c
	$(CC) -c $(CFLAGS) $(DFLAGS) $(INCLUDES) -o elina_box_constructor.o elina_box_constructor.c $(LIBS)

elina_box_assign.o : elina_box_assign.h elina_box_assign.c
	$(CC) -c $(CFLAGS) $(DFLAGS) $(INCLUDES) -o elina_box_assign.o elina_box_assign.c $(LIBS)

elina_box_meetjoin.o : elina_box_meetjoin.h elina_box_meetjoin.c
	$(CC) -c $(CFLAGS) $(DFLAGS) $(INCLUDES) -o elina_box_meetjoin.o elina_box_meetjoin.c $(LIBS)

elina_box_resize.o : elina_box_resize.h elina_box_resize.c
	$(CC) -c $(CFLAGS) $(DFLAGS) $(INCLUDES) -o elina_box_resize.o elina_box_resize.c $(LIBS)


install:
	$(INSTALLd) $(LIBDIR); \
	for i in $(SOINST); do \
		$(INSTALL) $$i $(LIBDIR); \
	done;

	$(INSTALLd) $(INCLDIR); \
	for i in $(ELINABOXH); do \
		$(INSTALL) $$i $(INCLDIR); \
	done; 


clean:
	-rm *.o
	-rm *.so

