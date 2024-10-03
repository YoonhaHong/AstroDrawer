"""
02/2023 Jihee Kim added number of events from csv file of beam measurements
06/2024 Bobae Kim updated
09/19/2024 Yoonha Hong updataed
"""

from utility import makesummary
import argparse
import csv
import matplotlib.pyplot as plt
from matplotlib.table import Table
import matplotlib.colors as mcolors
import matplotlib
import pandas as pd
import numpy as np
import glob
import os
from matplotlib.colors import Normalize
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
    
    nhits_sorted_output = output.sort_values(by='nHits', ascending=False)

    print(nhits_sorted_output.head(5))
    print(output.head(5))
    #print(output)
    #output.to_csv(f"{args.datadir}/{args.name}_{args.threshold}_summary.csv", index=False)

    row = 2
    col = 3

    fig, ax = plt.subplots(row, col, figsize=(col*6, row*5))
    #fig, ax = plt.subplots(row, col)
    
    for irow in range(0, row):
        for icol in range(0, col):
            for axis in ['top','bottom','left','right']:
                ax[irow, icol].spines[axis].set_linewidth(1.5)

    #### Hit Map ####
    h2_nHits = ax[0, 0].hist2d(x=output['col'], y=output['row'], 
        bins=35, range=[[0,35],[0,35]], 
        weights=output['nHits'],  
        cmap='YlOrRd',
        cmin=1.0, 
        norm=matplotlib.colors.LogNorm())  

    cbar_nHits = fig.colorbar(h2_nHits[3], ax=ax[0,0])
    cbar_nHits.set_label(label='Hit Counts', weight='bold', size=14)  # Colorbar 레이블 추가
    cbar_nHits.ax.yaxis.set_label_position('left')  # 레이블 위치를 왼쪽으로
    cbar_nHits.ax.yaxis.label.set_verticalalignment('center')  # 수직 정렬

    ax[0,0].grid()
    ax[0, 0].set_xlabel('Col', fontweight = 'bold', fontsize=14)
    ax[0, 0].set_ylabel('Row', fontweight = 'bold', fontsize=14)
    ax[0, 0].xaxis.set_tick_params(labelsize = 14)
    ax[0, 0].yaxis.set_tick_params(labelsize = 14)



    #### Readout Map ####
    h2_nReadouts = ax[0, 1].hist2d(x=output['col'], y=output['row'], 
        bins=35, range=[[0,35],[0,35]], 
        weights=output['nReadouts'], 
        cmap='YlOrRd',
        cmin=1.0, 
        norm=matplotlib.colors.LogNorm()) 
    
    cbar_nReadouts = fig.colorbar(h2_nReadouts[3], ax=ax[0,1])
    cbar_nReadouts.set_label(label='Readout Counts', weight='bold', size=14)  # Colorbar 레이블 추가
    cbar_nReadouts.ax.yaxis.set_label_position('left')  # 레이블 위치를 왼쪽으로
    cbar_nReadouts.ax.yaxis.label.set_verticalalignment('center')  # 수직 정렬

    ax[0, 1].grid()
    ax[0, 1].set_xlabel('Col', fontweight = 'bold', fontsize=14)
    ax[0, 1].set_ylabel('Row', fontweight = 'bold', fontsize=14)
    ax[0, 1].xaxis.set_tick_params(labelsize = 14)
    ax[0, 1].yaxis.set_tick_params(labelsize = 14)



    # 열 제목 추가
    columns = ['col', 'row', 'nHits', 'nReadouts']

    ax[1, 0].set_axis_off()
    ax[1, 1].set_axis_off()
    table_hit = Table(ax[1,0], bbox=[0, 0, 1, 1]) 
    table_rdo = Table(ax[1,1], bbox=[0, 0, 1, 1])
    for i, col_name in enumerate(columns):
        table_hit.add_cell(0, i, width=0.2, height=0.1, text=col_name, loc='center', facecolor='lightgrey')
        table_rdo.add_cell(0, i, width=0.2, height=0.1, text=col_name, loc='center', facecolor='lightgrey')
    rank = 10
    num_rows = min(rank, len(nhits_sorted_output))
    for i in range(num_rows):
        table_hit.add_cell(i+1, 0, width=0.2, height=0.1, text=str(nhits_sorted_output['col'].iloc[i]), loc='center')
        table_hit.add_cell(i+1, 1, width=0.2, height=0.1, text=str(nhits_sorted_output['row'].iloc[i]), loc='center')
        table_hit.add_cell(i+1, 2, width=0.2, height=0.1, text=str(nhits_sorted_output['nHits'].iloc[i]), loc='center')
        table_hit.add_cell(i+1, 3, width=0.2, height=0.1, text=str(nhits_sorted_output['nReadouts'].iloc[i]), loc='center')

        table_rdo.add_cell(i+1, 0, width=0.2, height=0.1, text=str(output['col'].iloc[i]), loc='center')
        table_rdo.add_cell(i+1, 1, width=0.2, height=0.1, text=str(output['row'].iloc[i]), loc='center')
        table_rdo.add_cell(i+1, 2, width=0.2, height=0.1, text=str(output['nHits'].iloc[i]), loc='center')
        table_rdo.add_cell(i+1, 3, width=0.2, height=0.1, text=str(output['nReadouts'].iloc[i]), loc='center')

    ax[1, 0].add_table(table_hit)  
    ax[1, 0].set_title(f"TOP{rank} noisy pixels (by nHits)", fontweight = 'bold', fontsize=14) 
    ax[1, 1].add_table(table_rdo)  
    ax[1, 1].set_title(f"TOP{rank} noisy pixels (by nReadouts)", fontweight = 'bold', fontsize=14) 

    # Text
    ax[1, 2].set_axis_off()
    ax[1, 2].text(0.05, 0.85, f"ChipID: {args.name}", fontsize=18, fontweight = 'bold');
    ax[1, 2].text(0.05, 0.75, f"Threshold: {args.threshold} mV", fontsize=18, fontweight = 'bold');
    ax[1, 2].text(0.05, 0.65, f"Time for each pixel: {0.084} min.", fontsize=18, fontweight = 'bold');
    ax[1, 2].text(0.05, 0.45, "Hits are processed below", fontsize=15, fontweight = 'bold');
    ax[1, 2].text(0.05, 0.35, f"conditions: <{args.timestampdiff} timestamp and <{args.totdiff}% in ToT", fontsize=15, fontweight = 'bold');



    # For making disable pixel
    nReadouts_threshold = args.rdothr
    nHits_threshold = args.hitthr

    bounds = [0, 1, 2, 3]  # 경계값 설정: 0, 1, 2 (2는 두 조건 모두 만족)
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    h2_Mask = ax[0, 2].hist2d(x=output['col'], y=output['row'], 
        bins=35, range=[[0,35],[0,35]], 
        weights= ((output['nReadouts'] > nReadouts_threshold).astype(int) + (output['nHits'] > nHits_threshold).astype(int)), 
        cmin = 1.0,
        norm=norm,
        cmap='gray_r')

    cbar_Mask = fig.colorbar(h2_Mask[3], ax=ax[0,2])
    cbar_Mask.set_label(label='# of satisfying conditions', weight='bold', size=14)  # Colorbar 레이블 추가
    cbar_Mask.ax.yaxis.set_label_position('left')  # 레이블 위치를 왼쪽으로
    cbar_Mask.ax.yaxis.label.set_verticalalignment('center')  # 수직 정렬

    ax[0, 2].grid()
    ax[0, 2].set_xlabel('Col', fontweight = 'bold', fontsize=14)
    ax[0, 2].set_ylabel('Row', fontweight = 'bold', fontsize=14)
    ax[0, 2].xaxis.set_tick_params(labelsize = 14)
    ax[0, 2].yaxis.set_tick_params(labelsize = 14)
    ax[0, 2].text(2.5, 32.5, f'nHits > {nHits_threshold}', fontsize=16)
    ax[0, 2].text(2.5, 30.5, f'nReadouts > {nReadouts_threshold}', fontsize=16)

    
    plt.savefig(f"./fig/{args.name}_THR{args.threshold}_All.pdf")
    plt.show()





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Astropix Driver Code')

    parser.add_argument('-n', '--name', required=False, default='APSw08s03',
                    help='chip ID that can be used in name of output file (default=APSw08s03)')

    parser.add_argument('-t', '--threshold', required=True, default =None,
                    help = 'input directory for beam data file')

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

