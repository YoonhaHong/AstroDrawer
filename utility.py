import pandas as pd
import glob
import os
import yaml

def yaml_reader(yaml_file):
    disablepix=[]
    file = open(yaml_file, 'r')
    config = yaml.safe_load(file)

    for col in range(0, 35, 1):
        value = config['astropix3']['config']['recconfig'][f'col{col}'][1]
        for row in range(0, 35, 1): 
                disable = (value & (2 << row)) >> (row+1)
                disablepix.append([col, row, disable])
    pixs=pd.DataFrame(disablepix, columns=['col','row','disable'])
    return pixs

def yaml_reader_astep(yaml_file):
    disablepix=[]
    file = open(yaml_file, 'r')
    config = yaml.safe_load(file)

    for col in range(0, 35, 1):
        value = config['Receiver'][f'col{col}']
        for row in range(0, 35, 1): 
                disable = (value & (2 << row)) >> (row+1)
                disablepix.append([col, row, disable])
    pixs=pd.DataFrame(disablepix, columns=['col','row','disable'])
    return pixs