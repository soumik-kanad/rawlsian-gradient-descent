currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py --train_data_balance --sensitive_attr=sex --save_dir=experiments/"$currdir" &>> experiments/"$currdir"/out.txt

#Comment: Balance the training data based on the taget value
#Training is SGD