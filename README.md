# Rawlsian Gradient Descent
Using Rawlsian theory of fairness (which has two principles - equal opportunity and compensation to least advantaged group) into gradient based optimisation methods. 
(see `report/828Z_PROJECT.pdf` for more details)

## Dataset
UCI Adults income dataset - [link](https://archive.ics.uci.edu/ml/datasets/adult)

## Requirements
`pip install -r requirements.txt`
- Code was tested on 
   - NVIDIA gtx1080ti
   - Cuda 11.1.1
   - Cudnn 8.2.1

## How to use
- Inside the project directory make a `<experiments_folder>`  directory
- Make make all your experiment folders inside that, eg. `<experiments_folder>/<exp_name>`
- `cp experiments_samples/exp1/run.sh <experiments_folder>/<exp_name>`
- Change the parameters according to your requirements (more details in `run.sh`) and then run `run.sh` (give executable permission).
   - `cd <experiments_folder>/<exp_name>`
   - `./run.sh`
- This should output `out.txt` and `logs` into `<experiments_folder>/<exp_name>`.

## Compile results from multiple experiments
`python get_results.py --dir=<experiments_folder>`
- This should compile all the results from `<experiments_folder>` and results in two files `results_race_<experiments_folder>.csv` and `results_sex_<experiments_folder>.csv`.


## Note
This was the course project for the course CMSC828Z: Just Machine Learning taught by Dr. Hal Daumé III taught in Fall 2021 at UMD.