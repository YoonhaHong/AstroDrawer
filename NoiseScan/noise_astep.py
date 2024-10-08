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
plt.style.use('classic')


def main(args):
    datadir = args.directory 
    #print (noise_scan_summary_path)

    csv_files = glob.glob(os.path.join(args.directory, "noise_scan_summary*.csv"))
    thr_match = re.search(r'THR(\d+)', csv_files[0])
    split_parts = re.split(r'THR\d+\.?\d*_', csv_files[0])[-1].split('_')

    threshold = thr_match.group(1) if thr_match else args.threshold  
    name = split_parts[0]
    date = split_parts[1]

    

    # csv 파일을 하나씩 읽어오고 리스트로 저장
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, sep='\t')
            
            # 데이터가 있는 경우만 리스트에 추가 (빈 데이터는 제외)
            if not df.empty:
                df_list.append(df)
            else:
                print(f"Skipping empty file: {file}")

        except pd.errors.EmptyDataError:
            # 파일이 비어 있으면 스킵
            print(f"Skipping empty file: {file}")

    # 모든 데이터를 합침 (index를 무시하고, 같은 컬럼은 수평적으로 이어붙임)
    if df_list:

        merged_df = pd.concat(df_list, ignore_index=True)

        # Col과 Row가 중복될 경우 Count가 높은 값을 선택
        merged_df = merged_df.sort_values(by='Count', ascending=False)
        output = merged_df.drop_duplicates(subset=['Col', 'Row'], keep='first')

        output_path = os.path.join(args.directory, f"{args.name}_THR{args.threshold}.csv")
        merged_df.to_csv(output_path, index=False)
    else:
        print("No valid files to merge.")

    if args.drawOption == 'Count': Count(output)
    elif args.drawOption == 'Mask' : Maskmap(output)

def Count(output):
    fig, ax = plt.subplots(figsize=(8, 8))  
    p1 = ax.hist2d(x=output['Col'], y=output['Row'], 
               bins=35, range=[[0, 35], [0, 35]], 
               weights=output['Count'],  
               norm=matplotlib.colors.LogNorm(),
               cmap='YlOrRd',
               cmin=1
               )


    top_n = 10
    count = 0

    for index, Row in output.iterrows():
        if(Row['Count']==0): continue
        ax.text(Row['Col']+0.5, Row['Row']+0.5, Row['Count'],
                    va="center", ha="center", color="blue", fontsize=7)
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

    plt.savefig(f"./fig_astep/{args.name}_THR{args.threshold}_{args.drawOption}.pdf")
    plt.show()

def Maskmap(output):

    Count_threshold = args.Count
    fig, ax = plt.subplots(figsize=(8, 8)) 

    bounds = [0, 1, 2, 3]  # 경계값 설정: 0, 1, 2 (2는 두 조건 모두 만족)
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    pn = ax.hist2d(x=output['Col'], y=output['Row'], 
        bins=35, range=[[0,35],[0,35]], 
        weights= (output['Count'] > Count_threshold).astype(int), 
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

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    #fig.suptitle(f'chip ID = {args.name}, Threhold = {args.threshold} mV', fontsize=16, fontweight='bold')

    plt.savefig(f"./fig_astep/{args.name}_THR{args.threshold}_count>{args.Count}_{args.drawOption}.pdf")
    plt.show()




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Astropix Driver Code')

    parser.add_argument('-d', '--directory', type=str, required=True, 
                    help='Directory path containing .txt files')

    parser.add_argument('-n', '--name', required=False, default='APSw08s03',
                    help='chip ID that can be used in name of output file (default=APSw08s03)')

    parser.add_argument('-ct','--Count', type=int, required=False, default=9,
                    help = 'threshold for Count')

    parser.add_argument('-draw', '--drawOption', type=str, default='nHits', 
                        choices=['Count', 'Mask'],
                        help='Choose the type of plot')

    parser.add_argument('-t', '--threshold', required=True, default =None,
                    help = 'input directory for beam data file')

    
    parser.add_argument
    args = parser.parse_args()

    main(args)

