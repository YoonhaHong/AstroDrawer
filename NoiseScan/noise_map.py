"""
02/2023 Jihee Kim added number of events from csv file of beam measurements
06/2024 Bobae Kim updated
09/19/2024 Yoonha Hong updataed
"""
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

    nhits_sorted_output = output.sort_values(by='nHits', ascending=False)

    print(nhits_sorted_output.head(5))
    print(output.head(5))
    #print(output)
    #output.to_csv(f"{args.datadir}/{args.name}_{args.threshold}_summary.csv", index=False)

    row = 2
    col = 2
    fig, ax = plt.subplots(row, col, figsize=(row*6, row*5))
    for irow in range(0, row):
        for icol in range(0, col):
            for axis in ['top','bottom','left','right']:
                ax[irow, icol].spines[axis].set_linewidth(1.5)

    #### Hit Map ####
    p1 = ax[0, 0].hist2d(x=output['col'], y=output['row'], 
           bins=35, range=[[0,35],[0,35]], 
           weights=output['nHits'],  
           cmap='YlOrRd',
           #cmin=1.0, 
           norm=matplotlib.colors.LogNorm())  
    fig.colorbar(p1[3], ax=ax[0, 0]).set_label(label='Hit Counts', weight='bold', size=14)
    ax[0,0].grid()
    ax[0, 0].set_xlabel('Col', fontweight = 'bold', fontsize=14)
    ax[0, 0].set_ylabel('Row', fontweight = 'bold', fontsize=14)
    ax[0, 0].xaxis.set_tick_params(labelsize = 14)
    ax[0, 0].yaxis.set_tick_params(labelsize = 14)

    #### Readout Map ####
    p2 = ax[0, 1].hist2d(x=output['col'], y=output['row'], 
           bins=35, range=[[0,35],[0,35]], 
           weights=output['nReadouts'], 
           cmap='YlOrRd',
           #cmin=1.0, 
           norm=matplotlib.colors.LogNorm()) 
    fig.colorbar(p2[3], ax=ax[0, 1]).set_label(label='Readout Counts', weight='bold', size=14)
    ax[0,1].grid()
    ax[0, 1].set_xlabel('Col', fontweight = 'bold', fontsize=14)
    ax[0, 1].set_ylabel('Row', fontweight = 'bold', fontsize=14)
    ax[0, 1].xaxis.set_tick_params(labelsize = 14)
    ax[0, 1].yaxis.set_tick_params(labelsize = 14)

    ax[1, 0].set_axis_off()
    table = Table(ax[1,0], bbox=[0, 0, 1, 1])  # ax[1,0] 안에 표 위치 조정

    # 열 제목 추가
    columns = ['col', 'row', 'nHits', 'nReadouts']
    for i, col_name in enumerate(columns):
        table.add_cell(0, i, width=0.2, height=0.1, text=col_name, loc='center', facecolor='lightgrey')

    rank = 10
    num_rows = min(rank, len(nhits_sorted_output))
    for i in range(num_rows):
        table.add_cell(i+1, 0, width=0.2, height=0.1, text=str(nhits_sorted_output['col'].iloc[i]), loc='center')
        table.add_cell(i+1, 1, width=0.2, height=0.1, text=str(nhits_sorted_output['row'].iloc[i]), loc='center')
        table.add_cell(i+1, 2, width=0.2, height=0.1, text=str(nhits_sorted_output['nHits'].iloc[i]), loc='center')
        table.add_cell(i+1, 3, width=0.2, height=0.1, text=str(nhits_sorted_output['nReadouts'].iloc[i]), loc='center')

    # Add table to ax[1,0]
    ax[1,0].add_table(table)   

    # Text
    ax[1, 1].set_axis_off()
    ax[1, 1].text(0.05, 0.85, f"ChipID: {args.name}", fontsize=18, fontweight = 'bold');
    ax[1, 1].text(0.05, 0.75, f"Threshold: {args.threshold} mV", fontsize=18, fontweight = 'bold');
    ax[1, 1].text(0.05, 0.65, f"Time for each pixel: {0.084} min.", fontsize=18, fontweight = 'bold');
    ax[1, 1].text(0.05, 0.45, "Hits are processed below", fontsize=15, fontweight = 'bold');
    ax[1, 1].text(0.05, 0.35, f"conditions: <{args.timestampdiff} timestamp and <{args.totdiff}% in ToT", fontsize=15, fontweight = 'bold');

    ax[1, 0].set_title(f"TOP{rank} noisy pixels", fontweight = 'bold', fontsize=14)
    plt.savefig(f"./fig/{args.name}_THR{args.threshold}_noisemap.pdf")
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
