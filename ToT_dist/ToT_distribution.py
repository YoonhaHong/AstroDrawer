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

    thr_match = re.search(r'THR(\d+)', file_name)

    # THR 뒤에 _로 구분된 문자열들을 추출 (csv 확장자는 제외)
    split_parts = re.split(r'THR\d+\.?\d*_', file_name)[-1].split('_')

    threshold = thr_match.group(1) if thr_match else args.threshold  
    name = split_parts[0]
    date = split_parts[1]

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
    #findyaml = f"{dir_name}/*_{date}.yml"
    #yamlpath = glob.glob(findyaml)
    #print(yamlpath[0])

    #pixs=yaml_reader(yamlpath[0])
    pixs=[]
    #navailpixs = pixs[pixs['disable'] == 0].shape[0]
    #npixel = '%.2f' % ( (navailpixs/1225) * 100.)
    #print(f"{navailpixs}, {npixel}% active")
    disablepix=[]
    for r in range(0,35,1):
        for c in range(0,3,1): # 0-4 col
                disablepix.append([c, r, 1])
    pixs=pd.DataFrame(disablepix, columns=['col','row','disable'])
    print(pixs)
    npixel = '%.2f' % ( (1-(len(pixs)/1225)) * 100.)
    print(f"{len(pixs)}, {npixel}% active")
     
    ##### Create hit pixel dataframes #######################################################
    # Hit pixel information for all events
    dffpair = pd.DataFrame(pair, columns=['col', 'row', 
                                          'timestamp_col', 'timestamp_row', 
                                          'tot_us_col', 'tot_us_row', 'avg_tot_us'])


    outdir = dir_name+"/ToT_distributions_"+file_name
    os.makedirs(outdir, exist_ok=True)

# 각 col, row의 tot_us 분포 저장
    for col, row in dffpair[['col', 'row']].drop_duplicates().values:
        subset = dffpair[(dffpair['col'] == col) & (dffpair['row'] == row)]
        
        #npyfile = os.path.join(outdir, f"ToT_distribution_col{col}_row{row}.npy")
        #np.save(npyfile, subset['avg_tot_us'].values)

        txtfile = os.path.join(outdir, f"ToT_distribution_col{col}_row{row}.txt")
        np.savetxt(txtfile, subset['avg_tot_us'].values, fmt='%f')
    
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
