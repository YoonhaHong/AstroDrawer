import multiprocessing as mp
import os
import re
import time
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
from utility import yaml_reader_astep
plt.style.use('classic')

CHUNKSIZE = 10000

# 청크 단위로 데이터를 처리하는 함수
def process_chunk(dff, ts, tot):
    n_evt_used = 0
    pair = []
    
    for ievt in dff['readout'].unique():
        df_event = dff.loc[(dff['readout'] == ievt)]
        if df_event.empty:
            continue
        
        n_evt_used += 1
        df_col = df_event[df_event['isCol'] == True]
        df_row = df_event[df_event['isCol'] == False]

        timestamp_diff = ts
        tot_time_limit = tot

        for indc in df_col.index:
            for indr in df_row.index:
                if df_col['tot_us'][indc] == 0 or df_row['tot_us'][indr] == 0:
                    continue
                if (abs(df_col['timestamp'][indc] - df_row['timestamp'][indr]) < timestamp_diff) & \
                   (abs(df_col['tot_us'][indc] - df_row['tot_us'][indr]) / df_col['tot_us'][indc] * 100 < tot_time_limit):
                    
                    average_tot = (df_col['tot_us'][indc] + df_row['tot_us'][indr]) / 2
                    pair.append([df_col['location'][indc], df_row['location'][indr], 
                                 df_col['timestamp'][indc], df_row['timestamp'][indr], 
                                 df_col['tot_us'][indc], df_row['tot_us'][indr], average_tot])
    
    return pair

# 병렬 처리 메인 함수
def parallel_process(f, ts, tot):
    cpu_count = mp.cpu_count()
    
    pool = mp.Pool(cpu_count)
    results = []
    
    # 헤더를 먼저 읽어 저장
    df_header = pd.read_csv(f, sep='\t', nrows=1)
    print( df_header.columns)
    
    # 첫 번째 청크는 헤더 포함, 이후 청크는 헤더 없이 처리
    first_chunk = True
    for chunk in pd.read_csv(f, sep='\t', chunksize=CHUNKSIZE):
        if first_chunk:
            result = pool.apply_async(process_chunk, args=(chunk, ts, tot))
            first_chunk = False
        else:
            # 헤더를 없애고 청크를 읽기
            chunk.columns = df_header.columns
            result = pool.apply_async(process_chunk, args=(chunk, ts, tot))
        
        results.append(result)

    pool.close()
    pool.join()

    final_pairs = []
    for result in results:
        final_pairs.extend(result.get())

    return final_pairs

def main(args):

    t_start = time.time()

    pair = [] 
    tot_n_nans = 0
    tot_n_evts = 0
    n_evt_used = 0
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

    pair = parallel_process(f, args.timestampdiff, args.totdiff) 
    print("... Matching is done!")

    ##### Find masked pixel and save it as pixs###########################################################
    findyaml = f"{dir_name}/{file_name}*.yml"
    yamlpath = glob.glob(findyaml)
    print(yamlpath[0])

    disablepix=yaml_reader_astep(yamlpath[0])
    navailpixs = disablepix[disablepix['disable'] == 0].shape[0]
    npixel = '%.2f' % ( (navailpixs/1225) * 100.)
    print(f"{navailpixs}, {npixel}% active")
    pixs=pd.DataFrame(disablepix, columns=['col','row','disable'])
    #print(pixs)
     
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

    p2 = ax[0, 1].hist2d(x=pixs['col'], y=pixs['row'], bins=35, range=[[0.,35],[0,35]], 
                         weights=pixs['disable'], 
                         norm=Normalize(vmin=0,vmax=1),cmap='Greys')
    fig.colorbar(p2[3], ax=ax[0, 1]).set_label(label='Masked', weight='bold', size=14)
    ax[0,1].grid()
    ax[0, 1].set_xlabel('Col', fontweight = 'bold', fontsize=14)
    ax[0, 1].set_ylabel('Row', fontweight = 'bold', fontsize=14)
    ax[0, 1].xaxis.set_tick_params(labelsize = 14)
    ax[0, 1].yaxis.set_tick_params(labelsize = 14)

#p1+p2 overlay plot
    p3 = ax[0,2].hist2d(x=pixs['col'], y=pixs['row'], bins=35, range=[[0.,35],[0,35]], 
                        weights=pixs['disable'], 
                        norm=Normalize(vmin=0,vmax=1),cmap='Greys')
    p3 = ax[0,2].hist2d(x=dfpairc['col'], y=dfpairc['row'], bins=35, range=[[0,35],[0,35]], weights=dfpairc['hits'], cmap='YlOrRd', cmin=1.0, norm=matplotlib.colors.LogNorm())
    p3 = ax[0,2].hist2d(x=dfpairc['col'], y=dfpairc['row'], bins=35, range=[[0,35],[0,35]], weights=dfpairc['hits'], cmap='YlOrRd', cmin=1.0)
    #p3 = ax[1, 0].hist2d(x=dfpixel['col'], y=dfpixel['row'], bins=35, range=[[-0.5,34.5],[-0,35]], weights=dfpixel['norm_sum_avg_tot_us'], cmap='Blues',cmin=1.0, norm=matplotlib.colors.LogNorm())
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
    ax[1, 2].text(0.1, 0.40, f"Available Pixels: {npixel}%", fontsize=15, fontweight = 'bold');
#    ax[0, 2].text(0.1, 0.75, f"Runs: {runnum}", fontsize=15, fontweight = 'bold');
    #ax[1, 2].text(0.1, 0.70, f"Events: {tot_n_evts}", fontsize=15, fontweight = 'bold');
    ax[1, 2].text(0.1, 0.60, "Processed below", fontsize=15, fontweight = 'bold');
    ax[1, 2].text(0.1, 0.55, f"conditions: <{args.timestampdiff} timestamp and <{args.totdiff}% in ToT", fontsize=15, fontweight = 'bold');
    #ax[1, 2].text(0.1, 0.50, f"nevents: {nevents}%", fontsize=15, fontweight = 'bold');
    ax[1, 2].text(0.1, 0.45, f"nhits: {nhits}", fontsize=15, fontweight = 'bold');

    ax[0, 0].set_title(f"Hit Map", fontweight = 'bold', fontsize=14)
    ax[0, 1].set_title(f"Masked pixel", fontweight = 'bold', fontsize=14)
    ax[0, 2].set_title(f"Hit Map with Masked pixels", fontweight = 'bold', fontsize=14)
    ax[1, 0].set_title(f"Avg.ToT per pixel", fontweight = 'bold', fontsize=14)
    ax[1, 1].set_title(f"Avg.ToT for all pixels", fontweight = 'bold', fontsize=14)
    #plt.savefig(f"{args.outdir}/{args.beaminfo}_{args.name}_run_{runnum}_evtdisplay.png")
    #print(f"{args.outdir}/{args.beaminfo}_{args.name}_run_{runnum}_evtdisplay.png was created...")

    figdir=args.outdir if args.outdir else dir_name
    plt.savefig(f"{figdir}/{file_name}_{args.beaminfo}_diffTS{args.timestampdiff}_diffToT{args.totdiff}_mp.png")
    print(f"Saved at {figdir}")
    plt.show()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Astropix Driver Code')

    parser.add_argument('-if', '--inputfile', required=False, default =None,
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
    
    parser.add_argument('-b', '--beaminfo', default='Sr90', required=False,
                    help='beam information ex) proton120GeV')

    parser.add_argument('-ns', '--noisescandir', action='store', required=False, type=str, default ='../astropix-python/noisescan',
                    help = 'filepath noise scan summary file containing chip noise infomation.')

    parser.add_argument
    args = parser.parse_args()

    main(args)


