"""
02/2023 Jihee Kim added number of events from csv file of beam measurements
06/2024 Bobae Kim updated
09/09/2024 Yoonha Hong updataed
"""
import argparse
import csv
import matplotlib.pyplot as plt
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

    pair = [] 
    output = pd.DataFrame( columns=['row','col','nReadouts','nHits'] )
    #noisescan_col13_row18_20240911-181123
    for r in range(18,19,1):
        for c in range(13,15,1):
            findfile = f"{args.datadir}/noisescan_col{c}_row{r}_*.csv"
            filename = glob.glob(findfile)
            if filename:
                df = pd.read_csv(filename[0],sep=',')
            else:
                print(f"r{r} c{c}: FILE NOT FOUND")
                continue
            tot_n_nans = 0
            tot_n_evts = 0
            n_evt_excluded = 0
            n_evt_used = 0
            

            n_all_rows = df.shape[0] #num of rows in .csv file
            n_non_nan_rows = df['readout'].count() #valid number of row of 'readout' in .csv file
            n_nan_evts = n_all_rows - n_non_nan_rows
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            df['readout'] = df['readout'].astype('Int64')

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
                            if dffcol['tot_us'][indc] == 0 or dffrow['tot_us'][indr] ==0: continue
                            if (abs(dffcol['tot_us'][indc] - dffrow['tot_us'][indr])/dffcol['tot_us'][indc]*100 > tot_time_limit): continue
                            if (dffcol['location'][indc] != c or dffrow['location'][indr] != r):
                                print(f"[Matching but Continue] col.location, row.location = {dffcol['location'][indc]},{dffrow['location'][indr]}")
                                continue
                            else:
                                average_tot = ((dffcol['tot_us'][indc] + dffrow['tot_us'][indr])/2)
                                pair.append([ dffcol['location'][indc], dffrow['location'][indr], dffcol['timestamp'][indc], dffrow['timestamp'][indr], dffcol['tot_us'][indc], dffrow['tot_us'][indr], ((dffcol['tot_us'][indc] + dffrow['tot_us'][indr])/2)])
                                nhits += 1

            print(f"r{r} c{c}: {nhits} hits of {nreadouts} readouts")# Summary of how many events being used
            output.loc[len(output)] = [r, c, nreadouts, nhits]

    print(output)
    output.to_csv(f"{args.datadir}/summary.csv", index=False)

    row = 2
    col = 2
    fig, ax = plt.subplots(row, col, figsize=(12, 10))
    for irow in range(0, row):
        for icol in range(0, col):
            for axis in ['top','bottom','left','right']:
                ax[irow, icol].spines[axis].set_linewidth(1.5)

    p1 = ax[0, 0].hist2d(x=output['col'], y=output['row'], 
           bins=35, range=[[0,35],[0,35]],  # col, row의 범위로 빈 지정
           weights=output['nHits'],  # nReadouts로 가중치 부여
           cmap='YlOrRd',
           cmin=1.0, norm=matplotlib.colors.LogNorm())  # 색상
    ax[0,0].grid()
    ax[0, 0].set_xlabel('Col', fontweight = 'bold', fontsize=14)
    ax[0, 0].set_ylabel('Row', fontweight = 'bold', fontsize=14)
    ax[0, 0].xaxis.set_tick_params(labelsize = 14)
    ax[0, 0].yaxis.set_tick_params(labelsize = 14)

    p2 = ax[0, 1].hist2d(x=output['col'], y=output['row'], 
           bins=35, range=[[0,35],[0,35]],  # col, row의 범위로 빈 지정
           weights=output['nHits'],  # nReadouts로 가중치 부여
           cmap='YlOrRd',
           cmin=1.0, norm=matplotlib.colors.LogNorm())  # 색상
    fig.colorbar(p2[3], ax=ax[0, 1]).set_label(label='Masked', weight='bold', size=14)
    ax[0,1].grid()
    ax[0, 1].set_xlabel('Col', fontweight = 'bold', fontsize=14)
    ax[0, 1].set_ylabel('Row', fontweight = 'bold', fontsize=14)
    ax[0, 1].xaxis.set_tick_params(labelsize = 14)
    ax[0, 1].yaxis.set_tick_params(labelsize = 14)

    plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Astropix Driver Code')

    parser.add_argument('-n', '--name', default='APSw08s03', required=False,
                    help='chip ID that can be used in name of output file (default=APSw08s03)')

#    parser.add_argument('-l','--runnolist', nargs='+', required=True,
#                    help = 'List run number(s) you would like to see')

    parser.add_argument('-o', '--outdir', default='.', required=False,
                    help='output directory for all png files')

    parser.add_argument('-d', '--datadir', required=True, default =None,
                    help = 'input directory for beam data file')

    parser.add_argument('-td','--timestampdiff', type=float, required=False, default=2,
                    help = 'difference in timestamp in pixel matching (default:col.ts-row.ts<2)')
   
    parser.add_argument('-tot','--totdiff', type=float, required=False, default=10,
                    help = 'error in ToT[us] in pixel matching (default:(col.tot-row.tot)/col.tot<10%)')
    
    parser.add_argument('-b', '--beaminfo', default='Sr90', required=False,
                    help='beam information ex) proton120GeV')

    parser.add_argument('-ns', '--noisescaninfo', action='store', required=False, type=str, default ='.',
                    help = 'filepath noise scan summary file containing chip noise infomation.')

    parser.add_argument
    args = parser.parse_args()

    main(args)
