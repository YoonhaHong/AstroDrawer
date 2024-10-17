"""
02/2023 Jihee Kim added number of events from csv file of beam measurements
06/2024 Bobae Kim updated
09/19/2024 Yoonha Hong updataed
"""

import re
import argparse
import matplotlib.pyplot as plt
from matplotlib.table import Table
import matplotlib.colors as mcolors
import matplotlib
import pandas as pd
import numpy as np
import glob
import os
from matplotlib.colors import Normalize
from utility import yaml_reader_astep
plt.style.use('classic')


def main(args):
    
    output = yaml_reader_astep(args.inputfile)

    fig, ax = plt.subplots(figsize=(9, 8)) 

    bounds = [0, 1]  # 경계값 설정: 0, 1, 2 (2는 두 조건 모두 만족)
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    pn = ax.hist2d(x=output['col'], y=output['row'], 
        bins=35, range=[[0,35],[0,35]], 
        weights= (output['disable']).astype(int), 
        cmin = 1.0,
        norm=norm,
        cmap='gray_r')
    #cbar = fig.colorbar(pn[3], ax=ax, fraction=0.046, pad=0.04)  # fraction, pad로 크기 조절
    #cbar.set_label(label='Hit Counts', weight='bold', size=14)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Col', fontweight='bold', fontsize=14)
    ax.set_ylabel('Row', fontweight='bold', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid()

    plt.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.07)
    #fig.suptitle(f'chip ID = {args.name}, Threhold = {args.threshold} mV', fontsize=16, fontweight='bold')

    plt.savefig(f"./fig_astep/{args.name}_THR{args.threshold}.pdf")
    plt.show()






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Astropix Driver Code')


    parser.add_argument('-if', '--inputfile', required=True, default =None,
                    help = 'input file')

    parser.add_argument('-n', '--name', required=False, default='APSw08s03',
                    help='chip ID that can be used in name of output file (default=APSw08s03)')

    parser.add_argument('-t', '--threshold', required=True, default =None,
                    help = 'input directory for beam data file')

    
    parser.add_argument
    args = parser.parse_args()

    main(args)

