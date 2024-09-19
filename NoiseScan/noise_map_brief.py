#09/2024 Yoonha Hong 

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
import matplotlib as mpl
import asyncio
plt.style.use('classic')

def main(args):

    #outfilename=f"./{args.name}_{args.threshold}_summary.csv"
    outfilename=f"{args.datadir}/{args.name}_{args.threshold}_summary.csv"
    if not os.path.exists(outfilename): makesummary(outpath=outfilename)
    output= pd.read_csv(outfilename)

    print(output.head(5))

    fig, ax = plt.subplots(figsize=(8, 8))  # 사이즈 설정, 정사각형으로

    p1 = ax.hist2d(x=output['col'], y=output['row'], 
               bins=35, range=[[0, 35], [0, 35]], 
               weights=output['nReadouts'],  
               cmap='YlOrRd',
               norm=matplotlib.colors.LogNorm())


    top_n = 10
    count = 0

    for index, row in output.iterrows():
        if row['nReadouts'] == 0:
            break
        if count < top_n:
            ax.text(row['col']+0.5, row['row']+0.5, row['nReadouts'],
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

    plt.savefig(f"./fig/{args.name}_THR{args.threshold}_readout.pdf")
    plt.show()


def makesummary(outpath):
    output = pd.DataFrame( columns=['row','col','nReadouts','nHits'] )
    #noisescan_col13_row18_20240911-181123
    for r in range(0,35,1):
        for c in range(0,35,1):
            findfile = f"{args.datadir}/noisescan_col{c}_row{r}_*.csv"
            filename = glob.glob(findfile)
            if filename:
                df = pd.read_csv(filename[0],sep=',')
            else:
                print(f"r{r} c{c}: FILE NOT FOUND")
                continue
            

            n_all_rows = df.shape[0] #num of rows in .csv file
            n_non_nan_rows = df['readout'].count() #valid number of row of 'readout' in .csv file
            n_nan_evts = n_all_rows - n_non_nan_rows
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            df['readout'] = df['readout'].astype('Int64')

            if df.empty: continue
            max_readout_n = df['readout'].iloc[-1] #value from last row of 'readout'
            nreadouts = max_readout_n+1
            #print(f"n_all_rows={n_all_rows}")
            #print(f"n_non_nan_rows={n_non_nan_rows}")
            #print(f"max_readout_n={max_readout_n}")
    
            nhits = 0

            for ievt in range(0, nreadouts, 1):
                dff = df.loc[(df['readout'] == ievt) & (df['payload'] == 4) & (df['Chip ID'] == 0)]
                if dff.empty:
                    continue
        # Match col and row to find hit pixel
                else:
                    dffcol = dff.loc[dff['isCol'] == 1]
                    dffrow = dff.loc[dff['isCol'] == 0]
                    timestamp_diff = args.timestampdiff
                    tot_time_limit = args.totdiff

                    for indc in dffcol.index:
                       for indr in dffrow.index:
                            if (abs(dffcol['timestamp'][indc] - dffrow['timestamp'][indr]) > timestamp_diff): continue
                            if (dffcol['tot_us'][indc] == 0): continue
                            if (abs(dffcol['tot_us'][indc] - dffrow['tot_us'][indr])/dffcol['tot_us'][indc]*100 > tot_time_limit): continue
                            if (dffcol['location'][indc] != c or dffrow['location'][indr] != r):
                                #print(f"[Matching but Continue] col.location, row.location = {dffcol['location'][indc]},{dffrow['location'][indr]}")
                                continue
                            else:
                                #average_tot = ((dffcol['tot_us'][indc] + dffrow['tot_us'][indr])/2)
                                #pair.append([ dffcol['location'][indc], dffrow['location'][indr], dffcol['timestamp'][indc], dffrow['timestamp'][indr], dffcol['tot_us'][indc], dffrow['tot_us'][indr], ((dffcol['tot_us'][indc] + dffrow['tot_us'][indr])/2)])
                                nhits += 1

            print(f"r{r} c{c}: {nhits} hits of {nreadouts} readouts")# Summary of how many events being used
            output.loc[len(output)] = [r, c, nreadouts, nhits]

    output = output.sort_values(by='nReadouts',ascending=False)
    output.to_csv(outpath, index=False)
    print("Made .csv file")





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Astropix Driver Code')

    parser.add_argument('-n', '--name', required=False, default='APSw08s03',
                    help='chip ID that can be used in name of output file (default=APSw08s03)')

    parser.add_argument('-d', '--datadir', required=True, default =None,
                    help = 'input directory for beam data file')

    parser.add_argument('-t', '--threshold', required=True, default =None,
                    help = 'input directory for beam data file')

    parser.add_argument('-td','--timestampdiff', type=float, required=False, default=2,
                    help = 'difference in timestamp in pixel matching (default:col.ts-row.ts<2)')
   
    parser.add_argument('-tot','--totdiff', type=float, required=False, default=10,
                    help = 'error in ToT[us] in pixel matching (default:(col.tot-row.tot)/col.tot<10%)')
    
    parser.add_argument
    args = parser.parse_args()

    main(args)
