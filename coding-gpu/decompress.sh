FILE=$1
BASE=$FILE
JOINT=_
EXT=bstrap
mode=$3
OUTPUT=$2
MODEL_PATH=$4


if [ "$mode" = com ] ; then
	python decompress_adaptive.py --file_name $BASE --output $OUTPUT --model_weight_path $MODEL_PATH
elif [ "$mode" = bs ] ; then
	python decompress_bootstrap.py --filename $BASE --output $OUTPUT --model_weights_path $MODEL_PATH
fi

