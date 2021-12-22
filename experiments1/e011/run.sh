currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --rgd --rgd_mode=only1 --rgd_lr1=0.001 --sensitive_attr=sex --save_dir=experiments/"$currdir" &>> experiments/"$currdir"/out.txt

#Comment - The loss for equation 1 is quite high. Plus it slightly goes down and then goes back up
# so i am trying a different learning rate for equation 1.