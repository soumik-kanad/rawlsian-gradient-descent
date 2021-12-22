currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --rgd --rgd_mode=both --pretrain --sensitive_attr=race --save_dir=experiments2r/"$currdir" &>> experiments2r/"$currdir"/out.txt

#Comment - Pretrain