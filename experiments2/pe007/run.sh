currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --rgd --rgd_mode=both --pretrain --rgd_k1=1 --rgd_k2=10 --sensitive_attr=sex --save_dir=experiments2/"$currdir" &>> experiments2/"$currdir"/out.txt

#Comment - Pretrain