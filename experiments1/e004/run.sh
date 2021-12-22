currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --rgd --rgd_mode=only1 --sensitive_attr=sex --save_dir=experiments/"$currdir" &>> experiments/"$currdir"/out.txt