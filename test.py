from saimese_seperate.data_processer import *
R = read_files

train_data = R.read_csv('input_data/train100.csv')
for line in train_data:
    print(line)