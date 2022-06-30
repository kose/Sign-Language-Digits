#!/bin/sh

export PATH=$HOME/miniforge3/bin:$PATH
export LANG=C

cd `dirname $0`
CDIR=`pwd`

mkdir -p result

if test "$1" = "--clean"; then
    echo "rm result/optuna.db"
    rm result/optuna.db
fi

if test ! -s parameters.py; then

    cat <<EOF > parameters.py
dropout = 0.2565810500965313
dim1=30
dim2=30
dim3=30
dim4=30
dim5=30
EOF

fi


proc()
{
    SCRIPT=$1
    N=$2
    LOG=result/result_$SCRIPT.txt

    cp -p result/optuna.db result/optuna_0.db
    
    /bin/echo -n "# $SCRIPT: `date` -> " >> $LOG
    python $SCRIPT.py --trial $N

    echo "`date +%H:%M:%S`" >> $LOG
    python $SCRIPT.py --trial 0 >> $LOG

    echo "" >> $LOG

    python $SCRIPT.py --trial 0 > /tmp/parameters.py
    mv /tmp/parameters.py .
}

proc optuna-hand_sign 30

DATE=`date +%Y-%m%d%M`
python train.py --log $DATE
python test.py --model result/$DATE.pt

cp result/$DATE.pt result/result.pt

# end
