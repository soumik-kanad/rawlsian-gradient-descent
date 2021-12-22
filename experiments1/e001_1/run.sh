# here I am slowing the learning to see if it can learn better with SGD 
currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --sensitive_attr=sex --epochs=1000 --learning_rate=0.001 --save_dir=experiments/"$currdir" &>> experiments/"$currdir"/out.txt