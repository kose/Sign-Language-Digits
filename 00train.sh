#!/bin/sh

export PATH=$HOME/miniforge3/bin:$PATH
export LANG=C

cd `dirname $0`
CDIR=`pwd`

mkdir -p result


##
## analize options
##
while [ $# -gt 0 ]
do
    case $1 in
	--clean)
	    FLAGS_clean=true
	    ;;
	    # exit 0;;

	-optuna | --optuna)
	    FLAGS_optuna=true
	    ;;

	-test | --test)
	    FLAGS_test=true
	    ;;

	-train | --train)
	    FLAGS_train=true
	    ;;

	-*)
	    echo "unknown: $1"
	    exit -1
	    ;;

	*)
	    echo "unknown: $1"
	    exit -1
	    ;;
    esac
    shift
done

##
##
##
if test ! -s parameters.py; then

    cat <<EOF > parameters.py
lr = 0.001
dropout = 0.3
dim1=20
dim2=20
dim3=20
dim4=20
dim5=50
EOF
fi

##
##
##
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

if test $FLAGS_clean; then
    echo "rm result/optuna.db"
    rm result/optuna.db
    proc optuna-hand_sign 50
    exit 0
fi

if test $FLAGS_test; then

    DATE=`date +%Y-%m%d%H%M`
    python train.py --log $DATE --epoch 20
    cp result/$DATE.pt result/result.pt

    # python test.py --model result/$DATE.pt
    exit 0
fi

if test $FLAGS_optuna; then
    optuna --storage sqlite:///result/optuna.db best-trials --study-name hand-sign
    exit 0
fi

if test $FLAGS_train; then
    proc optuna-hand_sign 50
    exit 0
fi

echo "do nothing: Sign-Language-Digits/00train.sh"

# end
