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

$RSYNC -avr  $LPATH -e  "$SSH -i $KEY" $RUSER@$RHOST:$RPATH 

