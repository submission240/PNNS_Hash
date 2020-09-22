SRCS=pri_queue.cc util.cc eh.cc bh.cc mh.cc baseline.cc \
	rqalsh.cc fh.cc fh_sampling.cc mfh.cc mfh_sampling.cc \
	qalsh.cc nh.cc nh_sampling.cc nh_lccs.cc nh_lccs_sampling.cc \
	nns.cc main.cc
OBJS=${SRCS:.cc=.o}

CXX=g++-8 -std=c++17
CPPFLAGS=-w -O3

.PHONY: clean

all: ${OBJS}
	${CXX} ${CPPFLAGS} -o plane ${OBJS}

pri_queue.o: pri_queue.h

util.o: util.h

eh.o: eh.h  kl_bucketing.h

bh.o: bh.h  kl_bucketing.h 

mh.o: mh.h  kl_bucketing.h

baseline.o: baseline.h

rqalsh.o: rqalsh.h

fh.o: fh.h

fh_sampling.o: fh_sampling.h

mfh.o: mfh.h

mfh_sampling.o: mfh_sampling.h

qalsh.o: qalsh.h

nh.o: nh.h

nh_sampling.o: nh_sampling.h

nh_lccs.o: nh_lccs.h

nh_lccs_sampling.o: nh_lccs_sampling.h

nns.o: nns.h

main.o:

clean:
	-rm ${OBJS} plane
