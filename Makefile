# Makefile

EXE1=d2q9-bgk.exe
EXES=$(EXE1)

#CC=tau_cc.sh
#CFLAGS= -std=c99 -Wall -O3 -funroll-loops -g -fopenmp #-qopt-report

CC=icc
CFLAGS= -std=c99 -Wall -xHost -Ofast -ipo -g -fopenmp #-qopt-report
#CFLAGS= -std=c99 -Wall -Ofast -fopenmp -prof-gen
#CFLAGS= -std=c99 -Wall -Ofast -fopenmp -fprof-use
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat

REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat

REF_FINAL_STATE_FILE_256=check/256x256.final_state.dat
REF_AV_VELS_FILE_256=check/256x256.av_vels.dat

REF_FINAL_STATE_FILE_EMPTY=check/empty.final_state.dat
REF_AV_VELS_FILE_EMPTY=check/empty.av_vels.dat

all: $(EXES)

$(EXES): %.exe : %.c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@ 

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

check256:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE_256) --ref-final-state-file=$(REF_FINAL_STATE_FILE_256) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

checkEmpty:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE_EMPTY) --ref-final-state-file=$(REF_FINAL_STATE_FILE_EMPTY) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXES)
	rm -f *.pomp.*
	rm -f cachegrind.out.*
	rm -f callgrind.out.*
	rm -f profile.0.0.*
	rm -f gmon.out

