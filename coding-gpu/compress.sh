FILE=$1
BASE=${FILE##*/}
BASE=${BASE%.*}
JOINT=_
EXT=bstrap
mode=$3
OUTPUT=$2

python run.py --file_name $FILE

python train_bootstrap.py --file_name $BASE --epochs 8

if [ "$mode" = com ] ; then
	python compress_adaptive.py --file_name $BASE --output $2
elif [ "$mode" = bs ] ; then
	python compress_bootstrap.py --filename $BASE --bs 64 --timesteps 64 --output $2
fi

