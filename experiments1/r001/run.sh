currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --sensitive_attr=race --save_dir=experiments/"$currdir" &>> experiments/"$currdir"/out.txt