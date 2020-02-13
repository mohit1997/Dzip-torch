FILE=$1
BASE=$FILE
JOINT=_
EXT=bstrap
mode=$3
OUTPUT=$2
MODEL_PATH=$4


if [ "$mode" = com ] ; then
	python decompress_adaptive.py --file_name $BASE --output $2 --model_weight_path $4
elif [ "$mode" = bs ] ; then
	python decompress_bootstrap.py --filename $BASE --output $2 --model_weights_path $4
fi

