currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --rgd --rgd_mode=both --rgd_step1_remove_normalisation --rgd_k1=5 --rgd_k2=1 --sensitive_attr=race --save_dir=experiments/"$currdir" &>> experiments/"$currdir"/out.txt

#Comment - Removing Normalisation