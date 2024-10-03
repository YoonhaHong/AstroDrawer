#09/2024 Yoonha Hong 

import argparse
import re
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
    file_pattern = 'injectionscan_c0r0_*.csv'

    file_list = glob.glob(os.path.join(args.datadir, file_pattern))

    mVinjections = []
    mVthresholds = []

    # 파일명에서 inj 값과 thr 값 추출
    for file in file_list:
        filename = os.path.basename(file)  # 파일명만 추출
        match = re.search(r'_([0-9.]+)inj_([0-9.]+)thr', filename)
        mVinjection = float(match.group(1))
        mVthreshold = float(match.group(2))
        mVinjections.append(mVinjection)
        mVthresholds.append(mVthreshold)
        print(f"File: {filename}, Inj: {mVinjection}, Thr: {mVthreshold}")

        df = pd.read_csv(args.datadir+'/'+filename, sep=',')

        n_all_rows = df.shape[0] #num of rows in .csv file
        n_non_nan_rows = df['readout'].count() #valid number of row of 'readout' in .csv file
        n_nan_evts = n_all_rows - n_non_nan_rows
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        df['readout'] = df['readout'].astype('Int64')

        if df.empty: continue
        max_readout_n = df['readout'].iloc[-1] #value from last row of 'readout'
        nreadouts = max_readout_n+1

    
        nhits = 0 
        totcsv = pd.DataFrame( columns=['tot_us_col','tot_us_row','avg_tot_us'] )
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
                        #if (dffcol['location'][indc] != c or dffrow['location'][indr] != r):
                        if (dffcol['location'][indc] > 34 or dffrow['location'][indr] > 34):
                            continue
                        else:
                            average_tot = ((dffcol['tot_us'][indc] + dffrow['tot_us'][indr])/2)
                            new_data = {'tot_us_col': dffcol['tot_us'][indc], 'tot_us_row':  dffrow['tot_us'][indr], 'avg_tot_us': average_tot}
                            totcsv.loc[len(totcsv)] = new_data
                            nhits += 1

        fname = "{0}_{1}_t{2}_v{3}".format(args.name, args.datadir, mVinjection, mVthreshold)
        path = args.datadir + '/' + fname + ".csv"
        #totcsv.to_csv( path, index=False)

        # 히스토그램 그리기
        plt.figure(figsize=(8, 6))
        bins = np.arange(0, 21, 1)
        plt.hist(totcsv['avg_tot_us'], bins=bins, color='skyblue', edgecolor='black')

        # 그래프 꾸미기
        plt.title(fname, fontsize=16)
        plt.xlabel('average TOT [us]', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)

        # 그래프 출력
        plt.grid(True)
        figpath = "./" + fname + ".pdf"
        plt.savefig(figpath)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Astropix Driver Code')

    parser.add_argument('-n', '--name', required=False, default='APSw08s03',
                    help='chip ID that can be used in name of output file (default=APSw08s03)')

    parser.add_argument('-d', '--datadir', required=True, default =None,
                    help = 'input directory for beam data file')
    
    parser.add_argument('-td','--timestampdiff', type=float, required=False, default=2,
                    help = 'difference in timestamp in pixel matching (default:col.ts-row.ts<2)')
   
    parser.add_argument('-tot','--totdiff', type=float, required=False, default=10,
                    help = 'error in ToT[us] in pixel matching (default:(col.tot-row.tot)/col.tot<10%)')
    
    
    parser.add_argument
    args = parser.parse_args()

    main(args)
