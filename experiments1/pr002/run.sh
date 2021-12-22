currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --pretrain --rgd --rgd_mode=both --sensitive_attr=race --save_dir=experiments/"$currdir" &>> experiments/"$currdir"/out.txt

#Comment - Adding pretraining