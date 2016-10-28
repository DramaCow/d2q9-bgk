# Makefile

EXE=d2q9-bgk

CC=icc
CFLAGS= -std=c99 -Wall -Ofast -xHost -ipo -fopenmp #-prof-use
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat

REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat
REF_FINAL_STATE_FILE_1=check/128x256.final_state.dat
REF_AV_VELS_FILE_1=check/128x256.av_vels.dat
REF_FINAL_STATE_FILE_2=check/256x256.final_state.dat
REF_AV_VELS_FILE_2=check/256x256.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@ 

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

check1:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE_1) --ref-final-state-file=$(REF_FINAL_STATE_FILE_1) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

check2:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE_2) --ref-final-state-file=$(REF_FINAL_STATE_FILE_2) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)


.PHONY: all check clean

clean:
	rm -f $(EXE)
