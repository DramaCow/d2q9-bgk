# Makefile

EXE=d2q9-bgk

CC=mpicc
CFLAGS= -std=c99 -Wall -Ofast -xHost -ipo -par-affinity=compact # -fopenmp
#CFLAGS= -std=c99 -Wall -O3 -g
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat

REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat
REF_FINAL_STATE_FILE_1=check/128x256.final_state.dat
REF_AV_VELS_FILE_1=check/128x256.av_vels.dat
REF_FINAL_STATE_FILE_2=check/256x256.final_state.dat
REF_AV_VELS_FILE_2=check/256x256.av_vels.dat
REF_FINAL_STATE_FILE_3=check/1024x1024.final_state.dat
REF_AV_VELS_FILE_3=check/1024x1024.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@ 

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

check1:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE_1) --ref-final-state-file=$(REF_FINAL_STATE_FILE_1) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

check2:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE_2) --ref-final-state-file=$(REF_FINAL_STATE_FILE_2) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

check3:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE_3) --ref-final-state-file=$(REF_FINAL_STATE_FILE_3) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)
	rm -f d2q9-bgk.80s*
	rm -f machine.file.*
