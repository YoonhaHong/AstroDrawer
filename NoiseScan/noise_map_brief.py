#09/2024 Yoonha Hong 

from utility import makesummary
import argparse
import csv
import matplotlib.pyplot as plt
from matplotlib.table import Table
import matplotlib
import pandas as pd
import numpy as np
import glob
import os
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import matplotlib as mpl
import asyncio
plt.style.use('classic')

def main(args):

    datadir = "../../data" if os.path.exists("../../data") else "/Users/yoonha/cernbox/AstroPix"
    noise_scan_summary_path=f"{datadir}/NoiseScan/{args.name}_{args.threshold}_summary.csv"
    #print (noise_scan_summary_path)

    try:
        output = pd.read_csv(noise_scan_summary_path)

    except:
        makesummary(threshold=args.threshold, output_path=noise_scan_summary_path, timestamp_diff=args.timestampdiff, tot_time_limit=args.totdiff)
        output = pd.read_csv(noise_scan_summary_path)

    if output.shape[0] <= 1:
            print(f"{noise_scan_summary_path} is empty")
            #os.remove(noise_scan_summary_path)
            return
    else:
        print(output.head(5))

    if args.drawOption == 'nReadouts': nReadouts_or_Hits(output, _option='nReadouts')
    elif args.drawOption == 'nHits': nReadouts_or_Hits(output, _option='nHits')
    elif args.drawOption == 'Mask' : Maskmap(output)

def nReadouts_or_Hits(output, _option='nReadouts'):
    fig, ax = plt.subplots(figsize=(8, 8))  
    p1 = ax.hist2d(x=output['col'], y=output['row'], 
               bins=35, range=[[0, 35], [0, 35]], 
               weights=output[_option],  
               cmap='YlOrRd',
               cmin=1,
               norm=matplotlib.colors.LogNorm()
               )


    top_n = 10
    count = 0

    for index, row in output.iterrows():
        if row[_option] == 0:
            break
        if count < top_n:
            ax.text(row['col']+0.5, row['row']+0.5, row[_option],
                     va="center", ha="center", color="w", fontsize=7)
            count += 1

    cbar = fig.colorbar(p1[3], ax=ax, fraction=0.046, pad=0.04)  # fraction, pad로 크기 조절
    #cbar.set_label(label='Hit Counts', weight='bold', size=14)

    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('Col', fontweight='bold', fontsize=14)
    ax.set_ylabel('Row', fontweight='bold', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid()

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    #fig.suptitle(f'chip ID = {args.name}, Threhold = {args.threshold} mV', fontsize=16, fontweight='bold')

    plt.savefig(f"./fig/{args.name}_THR{args.threshold}_{args.drawOption}.pdf")
    plt.show()

def Maskmap(output):

    nReadouts_threshold = args.rdothr
    nHits_threshold = args.hitthr
    fig, ax = plt.subplots(figsize=(8, 8)) 

    bounds = [0, 1, 2, 3]  # 경계값 설정: 0, 1, 2 (2는 두 조건 모두 만족)
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    pn = ax.hist2d(x=output['col'], y=output['row'], 
        bins=35, range=[[0,35],[0,35]], 
        weights= ((output['nReadouts'] > nReadouts_threshold).astype(int) + (output['nHits'] > nHits_threshold).astype(int)), 
        cmin = 1.0,
        norm=norm,
        cmap='gray_r')
    cbar = fig.colorbar(pn[3], ax=ax, fraction=0.046, pad=0.04)  # fraction, pad로 크기 조절
    #cbar.set_label(label='Hit Counts', weight='bold', size=14)

    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('Col', fontweight='bold', fontsize=14)
    ax.set_ylabel('Row', fontweight='bold', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid()
    ax.text(2.5, 32.5, f'nHits > {nHits_threshold}', fontsize=12)
    ax.text(2.5, 30.5, f'nReadouts > {nReadouts_threshold}', fontsize=12)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    #fig.suptitle(f'chip ID = {args.name}, Threhold = {args.threshold} mV', fontsize=16, fontweight='bold')

    plt.savefig(f"./fig/{args.name}_THR{args.threshold}_rdo>{nReadouts_threshold}_hit>{nHits_threshold}_{args.drawOption}.pdf")
    plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Astropix Driver Code')

    parser.add_argument('-n', '--name', required=False, default='APSw08s03',
                    help='chip ID that can be used in name of output file (default=APSw08s03)')

    parser.add_argument('-t', '--threshold', required=True, default =None,
                    help = 'input directory for beam data file')
    
    parser.add_argument('-draw', '--drawOption', type=str, default='nHits', 
                        choices=['nHits', 'nReadouts', 'Mask'],
                        help='Choose the type of plot')

    parser.add_argument('-rt','--rdothr', type=int, required=False, default=9,
                    help = 'threshold for nReadouts')
   
    parser.add_argument('-ht','--hitthr', type=int, required=False, default=9,
                    help = 'threshold for nHits')

    parser.add_argument('-td','--timestampdiff', type=float, required=False, default=2,
                    help = 'difference in timestamp in pixel matching (default:col.ts-row.ts<2)')
   
    parser.add_argument('-tot','--totdiff', type=float, required=False, default=10,
                    help = 'error in ToT[us] in pixel matching (default:(col.tot-row.tot)/col.tot<10%)')
    
    parser.add_argument
    args = parser.parse_args()

    main(args)
