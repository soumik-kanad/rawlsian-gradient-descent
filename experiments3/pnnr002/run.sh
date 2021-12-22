currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --rgd --rgd_mode=both --pretrain --rgd_step1_remove_normalisation --sensitive_attr=race --save_dir=experiments3/"$currdir" &>> experiments3/"$currdir"/out.txt

#Comment - Pretrain and Remove normalisation