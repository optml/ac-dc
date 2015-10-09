#!/bin/sh

cd cpp
make clean 
cd ..

RSYNC=/usr/bin/rsync
SSH=/usr/bin/ssh
KEY=/home/taki/.cron/cron
RUSER=mat614
RHOST=polyps.ie.lehigh.edu

RHOST=coral.ie.lehigh.edu
LPATH=/work/software/ac-dc
 
 
 
RPATH=/home/mat614/

$RSYNC -avr  $LPATH  --exclude data $RUSER@$RHOST:$RPATH 


LPATH=/work/software/ac-dc/cpp/data/rcv1_train.binary.8

 
RPATH=/home/mat614/ac-dc/cpp/data

$RSYNC -avr  $LPATH    $RUSER@$RHOST:$RPATH 

