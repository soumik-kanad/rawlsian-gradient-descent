currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --rgd --rgd_mode=both --pretrain --rgd_step1_remove_normalisation --rgd_k1=5 --rgd_k2=1 --sensitive_attr=sex --save_dir=experiments3/"$currdir" &>> experiments3/"$currdir"/out.txt

#Comment - Pretrain and Remove normalisation