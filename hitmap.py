import os, sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob
from utility import yaml_reader  # 상위 폴더의 utility 모듈에서 yaml_reader 함수를 가져옴

NCOL = 35
NROW = 35
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
    if "_offline.csv" in file_name: 
        file_name = file_name.rstrip('_offline.csv')
    elif ".csv" in file_name:
        file_name = file_name.rstrip('.csv')

    split_parts = file_name.split('_')
    date = split_parts[-1]  # 파일 이름에서 날짜 부분 추출
    name = split_parts[0:-1]

    df = pd.read_csv(f,sep=',')
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
        dff = df.loc[(df['readout'] == ievt) & (df['payload'] == 4) & (df['Chip ID'] == 0)]
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
    # YAML 파일에서 마스킹된 픽셀 정보 가져오기
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
    # How many hits are collected and shown in a plot
    nhits = dfpairc['hits'].sum()
    # mean of avg_tot_us, each col, row
    grouped_avg = dffpair.groupby(['col', 'row'])['avg_tot_us'].mean().reset_index(name='avg')
    print(grouped_avg)
    

    
    # 히트맵과 마스킹 맵을 겹쳐서 그리기
    fig, ax = plt.subplots(figsize=(8, 6))
      
    # 마스킹 맵 그리기 (히트맵 위에 겹쳐서 그리기)
    p2 = ax.hist2d(
        x=pixs['col'], 
        y=pixs['row'], 
        bins=[NCOL, NROW], 
        range=[[0, NCOL], [0, NROW]], 
        weights=pixs['disable'], 
        norm=matplotlib.colors.Normalize(vmin=0, vmax=1), 
        cmap='Greys'
    )  
    # 히트맵 그리기
    p1 = ax.hist2d(
        x=dfpairc['col'], 
        y=dfpairc['row'], 
        bins=[NCOL, NROW], 
        range=[[0, NCOL], [0, NROW]], 
        weights=dfpairc['hits'], 
        cmap='YlOrRd', 
        cmin=1.0, 
        norm=matplotlib.colors.LogNorm()
    )

    
    # 그래프 설정
    ax.set_xlabel('Col', fontweight='bold', fontsize=13)
    ax.set_ylabel('Row', fontweight='bold', fontsize=13)
    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)
    ax.set_title('Hit Map with Masked Pixels', fontweight='bold', fontsize=16)
    ax.set_aspect('equal')
    ax.grid()
    
    fig.tight_layout()  # 서브플롯 자동 정렬
    fig.subplots_adjust(right=0.85)  # colorbar 공간 확보
    cbar = fig.colorbar(p1[3], ax=ax, fraction=0.046, pad=0.04)  
    cbar.set_label(label='Hit Counts', weight='bold', size=13)
    # 결과 저장
    figdir = args.outdir if args.outdir else dir_name
    os.makedirs(figdir, exist_ok=True)
    plt.savefig(f"{figdir}/{file_name}_hitmap_with_masked.png")
    print(f"Saved at {figdir}/{file_name}_hitmap_with_masked.png")
    # 창 유지
    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Hit Map with Masked Pixels from CSV file.')
    parser.add_argument("inputfile", type=str, help='Path to the input CSV file')
    parser.add_argument('-o', '--outdir', default="./fig", help='Output directory for the plot')


    parser.add_argument('-td','--timestampdiff', type=float, required=False, default=2,
                    help = 'difference in timestamp in pixel matching (default:col.ts-row.ts<2)')
   
    parser.add_argument('-tot','--totdiff', type=float, required=False, default=10,
                    help = 'error in ToT[us] in pixel matching (default:(col.tot-row.tot)/col.tot<10%)')
    
    args = parser.parse_args()
    main(args)