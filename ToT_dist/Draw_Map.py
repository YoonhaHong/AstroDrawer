"""
02/2023 Jihee Kim added number of events from csv file of beam measurements
06/2024 Bobae Kim updated
09/09/2024 Yoonha Hong updataed
"""
import os
import re
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import glob
from matplotlib.colors import Normalize
import matplotlib as mpl
plt.style.use('classic')

def plot_hist2d(df, x_col, y_col, z_col, color, file_name):
    # Pivot the data for 2D histogram (heatmap)
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    p = ax.hist2d(x=df[x_col], y=df[y_col], bins=35, range=[[0,35],[0,35]], weights=df[z_col], cmap=color, cmin=1e-5)
    cbar = fig.colorbar(p[3], ax=ax, fraction=0.046, pad=0.04)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(z_col, fontweight='bold', fontsize=26)
    ax.set_xlabel('Col', fontweight='bold', fontsize=14)
    ax.set_ylabel('Row', fontweight='bold', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid()

    plt.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.07)
    dirname = "./fig"
    filename =file_name[0:-4]+'_'+z_col.replace('/', '')
    plt.savefig(f"{dirname}/{filename}.png")
    plt.show()


def main(args):

    csv_file = args.inputfile  # Update with your file path
    df = pd.read_csv(csv_file)
    
    # Check the contents of the dataframe
    print(df.head())

    file_name = os.path.basename(args.inputfile)
    # Plot 2D histograms for nhits, MPV, and chi2/ndf
    plot_hist2d(df, x_col='col', y_col='row', z_col='nhits',color='YlOrRd', file_name=file_name)
    #plot_hist2d(df, x_col='col', y_col='row', z_col='MPV', color='Blues', file_name=file_name)
    #plot_hist2d(df, x_col='col', y_col='row', z_col='chi2/ndf', color='gray_r', file_name=file_name)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Astropix Driver Code')

    parser.add_argument('-if', '--inputfile', required=True, default =None,
                    help = 'input file')

    parser.add_argument('-n', '--name', default='APSw08s03', required=False,
                    help='chip ID that can be used in name of output file (default=APSw08s03)')

    parser.add_argument('-o', '--outdir', default=None, required=False,
                    help='output directory for all png files')

    parser.add_argument('-t', '--threshold', type = float, action='store', required=False, default=None,
                    help = 'Threshold voltage for digital ToT (in mV). DEFAULT value in yml OR 100mV if voltagecard not in yml')

    parser.add_argument('-rt','--rdothr', type=int, required=False, default=9,
                    help = 'threshold for nReadouts')
   
    parser.add_argument('-ht','--hitthr', type=int, required=False, default=9,
                    help = 'threshold for nHits')

    parser.add_argument('-td','--timestampdiff', type=float, required=False, default=2,
                    help = 'difference in timestamp in pixel matching (default:col.ts-row.ts<2)')
   
    parser.add_argument('-tot','--totdiff', type=float, required=False, default=10,
                    help = 'error in ToT[us] in pixel matching (default:(col.tot-row.tot)/col.tot<10%)')
    
    parser.add_argument('-b', '--beaminfo', default='Sr90', required=False,
                    help='beam information ex) proton120GeV')

    parser.add_argument('-ns', '--noisescandir', action='store', required=False, type=str, default ='../astropix-python/noisescan',
                    help = 'filepath noise scan summary file containing chip noise infomation.')

    parser.add_argument
    args = parser.parse_args()

    main(args)
