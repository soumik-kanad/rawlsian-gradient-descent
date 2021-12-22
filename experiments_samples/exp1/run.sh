currdir="$(basename "$(pwd)")"
cd ../../
python -u train.py \
--batch_size=256 \
--epochs=10 \
--learning_rate=0.1 \
--rgd \ #To run RGD or not (if not set then code runs SGD)
--rgd_mode='both' \ #['both', 'only1', 'only2']
--rgd_k1=1 \ #how many time to run step1 in an iteration
--rgd_k2=5 \ #how many times to run step2 in an iteration
--sensitive_attr='sex' \ #['race','sex']
--rgd_step1_remove_normalisation \ #this removes the original normalisation proposed from step1
--save_dir=experiments/"$currdir" &>> experiments/"$currdir"/out.txt

