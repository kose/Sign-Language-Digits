#!/bin/sh

#
cd `dirname $0`
CDIR=`pwd`

FILENAME=train_master.csv

rm -f $FILENAME

for D in 0 1 2 3 4 5 6 7 8 9; do
    for F in Sign-Language-Digits-Dataset/Dataset/$D/*.JPG; do
	echo "$F,$D" >> $FILENAME
    done
done

# data N: 2062

sort -R $FILENAME > /tmp/$FILENAME

head -1650 /tmp/$FILENAME | sort > train.csv
tail -412 /tmp/$FILENAME | sort > test.csv

# end
