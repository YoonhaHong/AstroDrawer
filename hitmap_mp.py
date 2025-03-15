import os
import re
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import glob
from matplotlib.colors import Normalize
import matplotlib as mpl
from utility import yaml_reader
from Multiprocess import parallel_process 
plt.style.use('classic')

NCOL=35
NROW=35

def main(args):

    t_start = time.time()

    pair = [] 
    f = args.inputfile
    file_name = os.path.basename(f)
    dir_name = os.path.dirname(f)
    if "_offline.csv" in file_name: 
        file_name = file_name.rstrip('_offline.csv')
    elif ".csv" in file_name:
        file_name = file_name.rstrip('.csv')

    split_parts = file_name.split('_')
    date = split_parts[-1]  # 파일 이름에서 날짜 부분 추출
    name = split_parts[0:-1]

    pair = parallel_process(f, args.timestampdiff, args.totdiff) 
    print("... Matching is done!")

    ##### Find masked pixel and save it as pixs###########################################################
    findyaml = f"{dir_name}/*{date}*.yml"
    yamlpath = glob.glob(findyaml)
    
    if not yamlpath:
        print(f"No YAML file found for date: {date}")
        return
    
    disablepix = yaml_reader(yamlpath[0])  # YAML 파일 읽기
    pixs = pd.DataFrame(disablepix, columns=['col', 'row', 'disable'])  # 마스킹된 픽셀 정보를 DataFrame으로 변환
    navailpixs = pixs[pixs['disable'] == 0].shape[0]
    npixel = '%.2f' % ( (navailpixs/NCOL/NROW) * 100.)
    print(f"{navailpixs}, {npixel}% active")
     
    ##### Create hit pixel dataframes #######################################################
    # Hit pixel information for all events
    dffpair = pd.DataFrame(pair, columns=['col', 'row', 
                                          'timestamp_col', 'timestamp_row', 
                                          'tot_us_col', 'tot_us_row', 'avg_tot_us'])
    # Create dataframe for number of hits 
    dfpair = dffpair[['col','row']].copy()
    dfpairc = dfpair[['col','row']].value_counts().reset_index(name='hits')


    t_end = time.time()
    print(f"{t_end-t_start} Elapsed")

    fig, ax = plt.subplots(figsize=(8, 8))  
    p3 = ax.hist2d(x=pixs['col'], y=pixs['row'], bins=[NCOL, NROW], range=[[0, NCOL],[0, NROW]], 
                        weights=pixs['disable'], 
                        norm=Normalize(vmin=0,vmax=1),cmap='Greys')
    p1 = ax.hist2d(x=dfpairc['col'], y=dfpairc['row'], bins=[NCOL, NROW], range=[[0, NCOL], [0, NROW]], 
               weights=dfpairc['hits'],  
               cmap='YlOrRd',
               norm=matplotlib.colors.LogNorm(),
               cmin=1
               )
    cbar = fig.colorbar(p1[3], ax=ax, fraction=0.046, pad=0.04)  # fraction, pad로 크기 조절
    #cbar.set_label(label='Hit Counts', weight='bold', size=14)


    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Col', fontweight='bold', fontsize=14)
    ax.set_ylabel('Row', fontweight='bold', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid()

    plt.subplots_adjust(left=0.10, right=0.90, top=0.95, bottom=0.05)
    figdir=args.outdir if args.outdir else dir_name
    #plt.savefig(f"{figdir}/{file_name}_{args.beaminfo}_diffTS{args.timestampdiff}_diffToT{args.totdiff}_mp.png")
    #print(f"Saved at {figdir}")
    plt.show()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot Hit Map with Masked Pixels from CSV file - MultiProcessing version')
    parser.add_argument("inputfile", type=str, help='Path to the input CSV file')
    parser.add_argument('-o', '--outdir', default="./fig", help='Output directory for the plot')


    parser.add_argument('-td','--timestampdiff', type=float, required=False, default=2,
                    help = 'difference in timestamp in pixel matching (default:col.ts-row.ts<2)')
   
    parser.add_argument('-tot','--totdiff', type=float, required=False, default=10,
                    help = 'error in ToT[us] in pixel matching (default:(col.tot-row.tot)/col.tot<10%)')
    
    args = parser.parse_args()
    main(args)




