currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --rgd --rgd_mode=both --rgd_k1=1 --rgd_k2=2 --sensitive_attr=race --save_dir=experiments/"$currdir" &>> experiments/"$currdir"/out.txt