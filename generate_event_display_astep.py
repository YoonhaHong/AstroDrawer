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
import asyncio
from utility import yaml_reader
plt.style.use('classic')

def main(args):

    ##### Loop over data files and Find hit pixels #######################################################
    # List for hit pixels
    pair = [] 
    # How many events are remained in one dataset
    tot_n_nans = 0
    tot_n_evts = 0
    n_evt_excluded = 0
    n_evt_used = 0
    # Loop over file
    #for f in all_files:
     # Read csv file
    f = args.inputfile
    file_name = os.path.basename(f)
    dir_name = os.path.dirname(f)
    file_name = file_name.rstrip('.csv')



    threshold = args.threshold 
    name = args.name

    df = pd.read_csv(f,sep='\t')
    #print(df.head())
    print(f"Reading is done")

    # Count per run
    # Total number of rows
    n_all_rows = df.shape[0]
    print(f"n_all_rows={n_all_rows}")
    # Non-NaN rows
    n_non_nan_rows = df['readout'].count() 
    # NaN events
    n_nan_evts = n_all_rows - n_non_nan_rows
    # Skip rows with NAN
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    # Change float to int for readout col
    df['readout'] = df['readout'].astype('Int64')

    #add
    print(df.head())
    if 'readout' in df.columns:
        if len(df['readout']) > 0:
            max_n_readouts = df['readout'].iloc[-1]
            print(f"Max readout value: {max_n_readouts}")
        else:
            print("The 'readout' column is empty.")
    else:
        print("The 'readout' column does not exist.")

    # Get last number of readouts/events per run
    max_readout_n = df['readout'].iloc[-1]
    
    # Count for summary if multiple runs are read in
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

    # Loop over readouts/events
    for ievt in range(0, max_readout_n+1, 1):
        dff = df.loc[(df['readout'] == ievt) & (df['payload'] == 4) & (df['ChipID'] == 0)]
        if dff.empty:
            continue
        # Match col and row to find hit pixel
        else:
            n_evt_used += 1
            # List column info of pixel within one event
            dffcol = dff.loc[dff['isCol'] == True]
            # List row info of pixel within one event
            dffrow = dff.loc[dff['isCol'] == False]
            # Matching conditions: timestamp and time-over-threshold (ToT)
            timestamp_diff = args.timestampdiff
            tot_time_limit = args.totdiff
            # Loop over col and row info to find a pair to define a pixel
            for indc in dffcol.index:
                for indr in dffrow.index:
                    if dffcol['tot_us'][indc] == 0 or dffrow['tot_us'][indr] ==0:
                        continue
                    if (abs(dffcol['timestamp'][indc] - dffrow['timestamp'][indr]) < timestamp_diff) & (abs(dffcol['tot_us'][indc] - dffrow['tot_us'][indr])/dffcol['tot_us'][indc]*100 < tot_time_limit):
                        if (dffcol['location'][indc] > 34 or dffrow['location'][indr] > 34):
                            print(f"[Matching but Continue] col.location, row.location = {dffcol['location'][indc]},{dffrow['location'][indr]}")
                            continue
                        # Record hit pixels per event
                        average_tot = ((dffcol['tot_us'][indc] + dffrow['tot_us'][indr])/2)
                        pair.append([ dffcol['location'][indc], dffrow['location'][indr], dffcol['timestamp'][indc], dffrow['timestamp'][indr], dffcol['tot_us'][indc], dffrow['tot_us'][indr], ((dffcol['tot_us'][indc] + dffrow['tot_us'][indr])/2)])
    print("... Matching is done!")
    ######################################################################################################

    ##### Summary of how many events being used ###################################################
    nevents = '%.2f' % ((n_evt_used/(tot_n_evts)) * 100.)
    nnanevents = '%.2f' % ((tot_n_nans/(tot_n_evts)) * 100.)
    n_empty = tot_n_evts - n_evt_used - tot_n_nans
    nemptyevents = '%.2f' % ((n_empty/(tot_n_evts)) * 100.)
    print("Summary:")
    print(f"{tot_n_nans} of {tot_n_evts} events were found as NaN...")
    print(f"{n_empty} of {tot_n_evts} events were found as empty...")
    print(f"{n_evt_used} of {tot_n_evts} events were processed...")
#        print(f"{n_evt_excluded} of {tot_n_evts} events were excluded because of bad payload...")
#        print(f"{nevents}[%] are used in exclusively mode...")
#        print(f"{nnanevents}[%] are trashed...")
#        print(f"{nemptyevents}[%] are emptied...")
#        print(f"{nevents}[%] are used...")
#        print(f"{nnanevents}[%] are trashed...")
#        print(f"{nemptyevents}[%] are emptied...")

    ###############################################################################################
    # Masking pixels
    # Read noise scan summary file
    #datadir = "../../data" if os.path.exists("../../data") else "/Users/yoonha/cernbox/AstroPix"
    #findyaml = f"{dir_name}/*_{date}.yml"
    #yamlpath = glob.glob(findyaml)
    #print(yamlpath[0])

    #pixs=yaml_reader(yamlpath[0])
    pixs=[]
    #navailpixs = pixs[pixs['disable'] == 0].shape[0]
    #npixel = '%.2f' % ( (navailpixs/1225) * 100.)
    #print(f"{navailpixs}, {npixel}% active")
     
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

    # Generate Plot - Pixel hits
    #fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))
    row = 2
    col = 3
    fig, ax = plt.subplots(row, col, figsize=(20, 10))
    for irow in range(0, row):
        for icol in range(0, col):
            for axis in ['top','bottom','left','right']:
                ax[irow, icol].spines[axis].set_linewidth(1.5)

    p1 = ax[0, 0].hist2d(x=dfpairc['col'], y=dfpairc['row'], bins=35, range=[[0,35],[0,35]], weights=dfpairc['hits'], cmap='YlOrRd', cmin=1.0, norm=matplotlib.colors.LogNorm())
    fig.colorbar(p1[3], ax=ax[0, 0]).set_label(label='Hit Counts', weight='bold', size=14)
    ax[0,0].grid()
    ax[0, 0].set_xlabel('Col', fontweight = 'bold', fontsize=14)
    ax[0, 0].set_ylabel('Row', fontweight = 'bold', fontsize=14)
    ax[0, 0].xaxis.set_tick_params(labelsize = 14)
    ax[0, 0].yaxis.set_tick_params(labelsize = 14)

    #p2 = ax[0, 1].hist2d(x=pixs['col'], y=pixs['row'], bins=35, range=[[0.,35],[0,35]], 
    #                     weights=pixs['disable'], 
    #                     norm=Normalize(vmin=0,vmax=1),cmap='Greys')
    #fig.colorbar(p2[3], ax=ax[0, 1]).set_label(label='Masked', weight='bold', size=14)
    ax[0,1].grid()
    ax[0, 1].set_xlabel('Col', fontweight = 'bold', fontsize=14)
    ax[0, 1].set_ylabel('Row', fontweight = 'bold', fontsize=14)
    ax[0, 1].xaxis.set_tick_params(labelsize = 14)
    ax[0, 1].yaxis.set_tick_params(labelsize = 14)

#p1+p2 overlay plot
    #p3 = ax[0,2].hist2d(x=pixs['col'], y=pixs['row'], bins=35, range=[[0.,35],[0,35]], 
    #                    weights=pixs['disable'], 
    #                    norm=Normalize(vmin=0,vmax=1),cmap='Greys')
    # p3 = ax[0,2].hist2d(x=dfpairc['col'], y=dfpairc['row'], bins=35, range=[[0,35],[0,35]], weights=dfpairc['hits'], cmap='YlOrRd', cmin=1.0, norm=matplotlib.colors.LogNorm())
    p3 = ax[0,2].hist2d(x=dfpairc['col'], y=dfpairc['row'], bins=35, range=[[0,35],[0,35]], weights=dfpairc['hits'], cmap='YlOrRd', cmin=1.0)
#    p3 = ax[1, 0].hist2d(x=dfpixel['col'], y=dfpixel['row'], bins=35, range=[[-0.5,34.5],[-0,35]], weights=dfpixel['norm_sum_avg_tot_us'], cmap='Blues',cmin=1.0, norm=matplotlib.colors.LogNorm())
    fig.colorbar(p3[3], ax=ax[0, 2]).set_label(label='Hit Counts', weight='bold', size=14)
    ax[0,2].grid()
    ax[0,2].set_xlabel('Col', fontweight = 'bold', fontsize=14)
    ax[0,2].set_ylabel('Row', fontweight = 'bold', fontsize=14)
    ax[0,2].xaxis.set_tick_params(labelsize = 14)
    ax[0,2].yaxis.set_tick_params(labelsize = 14)

    p4 = ax[1, 0].hist2d(x=grouped_avg['col'], y=grouped_avg['row'], bins=35, range=[[0,35],[0,35]], weights=grouped_avg['avg'], cmap='Blues',vmin=0.0)
    fig.colorbar(p4[3], ax=ax[1, 0]).set_label(label='Avg.ToT [us]', weight='bold', size=14)
    ax[1, 0].grid()
    ax[1, 0].set_xlabel('Col', fontweight = 'bold', fontsize=14)
    ax[1, 0].set_ylabel('Row', fontweight = 'bold', fontsize=14)
    ax[1, 0].xaxis.set_tick_params(labelsize = 14)
    ax[1, 0].yaxis.set_tick_params(labelsize = 14)

    p5 = ax[1, 1].hist(x=dffpair['avg_tot_us'], bins=22, range=(0, 22), color='blue', edgecolor='black')
#    fig.colorbar(p5[3], ax=ax[1, 2]).set_label(label='Average Normalized Time-over-Threshold[us]', weight='bold', size=18)
    ax[1, 1].grid()
    ax[1, 1].set_xlabel('ToT [us]', fontweight = 'bold', fontsize=14)
    ax[1, 1].set_ylabel('Counts', fontweight = 'bold', fontsize=14)
    ax[1, 1].xaxis.set_tick_params(labelsize = 14)
    ax[1, 1].yaxis.set_tick_params(labelsize = 14)

    # Text
    ax[1, 2].set_axis_off()
    ax[1, 2].text(0.1, 0.85, f"Beam: {args.beaminfo}", fontsize=15, fontweight = 'bold');
    ax[1, 2].text(0.1, 0.80, f"ChipID: {name}", fontsize=15, fontweight = 'bold');
    #ax[1, 2].text(0.1, 0.40, f"Available Pixels: {npixel}%", fontsize=15, fontweight = 'bold');
#    ax[0, 2].text(0.1, 0.75, f"Runs: {runnum}", fontsize=15, fontweight = 'bold');
    ax[1, 2].text(0.1, 0.70, f"Events: {tot_n_evts}", fontsize=15, fontweight = 'bold');
    ax[1, 2].text(0.1, 0.60, "Processed below", fontsize=15, fontweight = 'bold');
    ax[1, 2].text(0.1, 0.55, f"conditions: <{args.timestampdiff} timestamp and <{args.totdiff}% in ToT", fontsize=15, fontweight = 'bold');
    ax[1, 2].text(0.1, 0.50, f"nevents: {nevents}%", fontsize=15, fontweight = 'bold');
    ax[1, 2].text(0.1, 0.45, f"nhits: {nhits}", fontsize=15, fontweight = 'bold');

    ax[0, 0].set_title(f"Hit Map", fontweight = 'bold', fontsize=14)
    ax[0, 1].set_title(f"Masked pixel", fontweight = 'bold', fontsize=14)
    ax[0, 2].set_title(f"Hit Map with Masked pixels", fontweight = 'bold', fontsize=14)
    ax[1, 0].set_title(f"Avg.ToT per pixel", fontweight = 'bold', fontsize=14)
    ax[1, 1].set_title(f"Avg.ToT for all pixels", fontweight = 'bold', fontsize=14)
    #plt.savefig(f"{args.outdir}/{args.beaminfo}_{args.name}_run_{runnum}_evtdisplay.png")
    #print(f"{args.outdir}/{args.beaminfo}_{args.name}_run_{runnum}_evtdisplay.png was created...")

    figdir=args.outdir if args.outdir else os.path.dirname(f)
    plt.savefig(f"{figdir}/{file_name}_{args.beaminfo}_diffTS{args.timestampdiff}_diffToT{args.totdiff}.png")
    print(f"Saved at {figdir}")
    plt.show()
    
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
    
    parser.add_argument('-b', '--beaminfo', default='None', required=False,
                    help='beam information ex) proton120GeV')

    parser.add_argument('-ns', '--noisescandir', action='store', required=False, type=str, default ='../astropix-python/noisescan',
                    help = 'filepath noise scan summary file containing chip noise infomation.')

    parser.add_argument
    args = parser.parse_args()

    main(args)
