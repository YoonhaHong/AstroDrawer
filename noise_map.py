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

    #noisescan_col13_row18_20240911-181123
    for r in range(0,35,1):
        for c in range(0,35,1):
            filename = f"{args.datadir}/noisescan_col{c}_row{r}_*.csv"

    
            pair = [] 
            tot_n_nans = 0
            tot_n_evts = 0
            n_evt_excluded = 0
            n_evt_used = 0
            
            df = pd.read_csv(filename,sep=',')

            n_all_rows = df.shape[0]
            #print(f"n_all_rows={n_all_rows}")
            n_non_nan_rows = df['readout'].count() 
            n_nan_evts = n_all_rows - n_non_nan_rows
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            df['readout'] = df['readout'].astype('Int64')

            max_readout_n = df['readout'].iloc[-1]
    
            ni = 0
            for ievt in range(0, max_readout_n+1, 1):
                dff = df.loc[(df['readout'] == ievt)] 
                if dff.empty:
                    continue
                else:
                    ni += 1
            n_evts = ni + n_nan_evts
            tot_n_evts += n_evts
            tot_n_nans += n_nan_evts

            for ievt in range(0, max_readout_n+1, 1):
                dff = df.loc[(df['readout'] == ievt) & (df['payload'] == 4) & (df['Chip ID'] == 0)]
                if dff.empty:
                    continue
        # Match col and row to find hit pixel
                else:
                    n_evt_used += 1
                    dffcol = dff.loc[dff['isCol'] == True]
                    dffrow = dff.loc[dff['isCol'] == False]
                    timestamp_diff = args.timestampdiff
                    tot_time_limit = args.totdiff

                    for indc in dffcol.index:
                       for indr in dffrow.index:
                            if dffcol['tot_us'][indc] == 0 or dffrow['tot_us'][indr] ==0:
                                continue
                            if (abs(dffcol['timestamp'][indc] - dffrow['timestamp'][indr]) < timestamp_diff) & (abs(dffcol['tot_us'][indc] - dffrow['tot_us'][indr])/dffcol['tot_us'][indc]*100 < tot_time_limit):
                                if (dffcol['location'][indc] > 34 or dffrow['location'][indr] > 34):
                                    print(f"[Matching but Continue] col.location, row.location = {dffcol['location'][indc]},{dffrow['location'][indr]}")
                                    continue
                                average_tot = ((dffcol['tot_us'][indc] + dffrow['tot_us'][indr])/2)
                                pair.append([ dffcol['location'][indc], dffrow['location'][indr], dffcol['timestamp'][indc], dffrow['timestamp'][indr], dffcol['tot_us'][indc], dffrow['tot_us'][indr], ((dffcol['tot_us'][indc] + dffrow['tot_us'][indr])/2)])
            print("... Matching is done!")

    ##### Summary of how many events being used ###################################################
            nevents = '%.2f' % ((n_evt_used/(tot_n_evts)) * 100.)
            nnanevents = '%.2f' % ((tot_n_nans/(tot_n_evts)) * 100.)
            n_empty = tot_n_evts - n_evt_used - tot_n_nans
            nemptyevents = '%.2f' % ((n_empty/(tot_n_evts)) * 100.)
            print("Summary:")
            print(f"{tot_n_nans} of {tot_n_evts} events were found as NaN...")
            print(f"{n_empty} of {tot_n_evts} events were found as empty...")
            print(f"{n_evt_used} of {tot_n_evts} events were processed...")

    ###############################################################################################
    print(f"{len(pixs)}, {npixel}% active")

    pixs=pd.DataFrame(disablepix, columns=['col','row','disable'])
    print(pixs)
    npixel = '%.2f' % ( (1-(len(pixs)/1225)) * 100.)
    print(f"{len(pixs)}, {npixel}% active")
     
    ##### Create hit pixel dataframes #######################################################
    # Hit pixel information for all events
    dffpair = pd.DataFrame(pair, columns=['col', 'row', 
                                          'timestamp_col', 'timestamp_row', 
                                          'tot_us_col', 'tot_us_row', 'avg_tot_us'])
    # Create dataframe for number of hits 
    dfpair = dffpair[['col','row']].copy()
    dfpairc = dfpair[['col','row']].value_counts().reset_index(name='hits')
    # How many hits are collected and shown in a plot
    nhits = dfpairc['hits'].sum()
    # mean of avg_tot_us, each col, row
    grouped_avg = dffpair.groupby(['col', 'row'])['avg_tot_us'].mean().reset_index(name='avg')
    print(grouped_avg)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Astropix Driver Code')

    parser.add_argument('-if', '--inputfile', required=True, default =None,
                    help = 'input file')

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
