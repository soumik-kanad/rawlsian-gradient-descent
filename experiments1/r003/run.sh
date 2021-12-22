currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --rgd --rgd_mode=only2 --sensitive_attr=race --save_dir=experiments/"$currdir" &>> experiments/"$currdir"/out.txt