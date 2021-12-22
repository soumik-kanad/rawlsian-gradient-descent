import os

dir='/vulcanscratch/smukhopa/CMSC828Z/experiments2r'
folders=[x for x in os.listdir(dir) if os.path.isdir(x)]

for folder in folders:
    print(folder)
    os.chdir(os.path.join(dir,folder))
    cmd="./run.sh"
    os.system(cmd)
