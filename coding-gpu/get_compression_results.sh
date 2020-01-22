FILE=$1
BASE=${FILE##*/}
BASE=${BASE%.*}

PRNN=biGRU_jump
ARNN=biGRU_big
JOINT=_

python run.py --file_name $FILE

python train_bootstrap.py --file_name $BASE --model $PRNN --epochs 10
python train_combined.py --file_name $BASE --PRNN $PRNN --ARNN $ARNN
