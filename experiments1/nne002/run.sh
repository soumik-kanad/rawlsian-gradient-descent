currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --rgd --rgd_mode=both --rgd_step1_remove_normalisation --sensitive_attr=sex --save_dir=experiments/"$currdir" &>> experiments/"$currdir"/out.txt

#Comment - Removing Normalisation