currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --rgd --rgd_mode=both --pretrain --rgd_k1=5 --rgd_k2=1 --sensitive_attr=race --save_dir=experiments2r/"$currdir" &>> experiments2r/"$currdir"/out.txt

#Comment - Pretrain