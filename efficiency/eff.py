import matplotlib.pyplot as plt
import os
import sys
import re
import glob
import pandas as pd
import argparse
from extract import extract_params_from_log

def main(args):
    files = []
    graph = []

    for file in os.listdir(args.directory):
        if not file.endswith(".csv"): continue
        if file.endswith("_offline.csv"):
            log_file=file.replace("_offline.csv", ".log")
        elif file.endswith(".csv"):
            log_file=file.replace(".csv", ".log")
        
        info = extract_params_from_log( os.path.join(args.directory, log_file ))
        nevent, nhit, ninjhit = matching(os.path.join(args.directory, file))
        graph.append(info|{"nevent":nevent, "nhit": nhit, "ninjhit": ninjhit})

    df_graph = pd.DataFrame(graph)

    per_to_frequency = {1: 928,
                        2: 541, 
                        3: 382,
                        4: 295,
                        5: 240, 
                        10: 124, 
                        100: 13}

    df_graph["time"] = df_graph["maxtime"]
    df_graph["frequency"] = df_graph["inject_period"].map(per_to_frequency)
    df_graph["injection"] = df_graph["frequency"] * df_graph["maxtime"]
    df_graph["efficiency"] = df_graph["nhit"]/df_graph["injection"]*100.
    df_graph["purity"] = df_graph["ninjhit"]/df_graph["nhit"]*100.
    df_graph = df_graph.sort_values('frequency').sort_values('maxtime')

    dirname = os.path.dirname(args.directory)
    print(dirname)
    df_graph.to_csv(f"./{dirname}.csv")
    print(df_graph)
    return
    
    df_by_group_label = df_graph.reset_index()
    df_pivot = df_by_group_label.pivot(index='frequency',columns='maxtime',values='efficiency')
    df_pivot.sort_values('frequency')

    print(df_pivot)
    


    fig, ax = plt.subplots(figsize=(10, 6))  # 가로 10인치, 세로 6인치 크기 설정
    df_pivot.plot.bar(ax=ax, rot=0) 
    #color=['skyblue', 'salmon', 'limegreen'])
    ax.set_ylabel("hit reco. efficiency(%)")
    ax.legend(["12 s", "60 s"], title="time of DAQ")
    #ax.set_title("Efficiency Comparison")
    ax.set_xlabel("Injection Frequency(Hz)")
    ax.set_ylim(0, 100)  # Y축을 0에서 100으로 제한
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    dirname = os.path.basename(args.directory)
    output_file = dirname+"_efficiency.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")  
    
    time_markers = {60: '-', 12: '--'}
    markers = [time_markers[time] for time in df_graph['time']]
    
    # Figure와 두 축 생성
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # 첫 번째 y축 (nhit)
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('# of hit', color='b', fontsize=12)
    for time in time_markers.keys():
        subset = df_graph[df_graph['time'] == time]
        ax1.plot(subset['frequency'], subset['nhit'], marker='o', linestyle=time_markers[time], label=f'nhit (time={time})', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='center right', fontsize=10)
    
    # 두 번째 y축 (nevent)
    ax2 = ax1.twinx()
    ax2.set_ylabel('# of event', color='r', fontsize=12)
    for time in time_markers.keys():
        subset = df_graph[df_graph['time'] == time]
        ax2.plot(subset['frequency'], subset['nevent'], marker='s', linestyle=time_markers[time], label=f'nevent (time={time})', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='center right', fontsize=10)
    
    # 제목과 격자 추가
    #plt.title('nhit and nevent vs Frequency (Color by Time)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 그래프 출력
    output_file = dirname+"_rdo+hit.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")  

def matching(file_path, timestamp_diff=2, tot_time_limit=10, inj_pixel=[10, 10]):
    dff = pd.read_csv(file_path,sep=',')
    n_evt = 0
    n_hit = 0
    n_injhit = 0 #hit at inj_pixel
    #pair = []

    
    for ievt in dff['readout'].unique():
        df_event = dff.loc[(dff['readout'] == ievt)]
        if df_event.empty:
            continue
        
        n_evt += 1
        df_col = df_event[df_event['isCol'] == True]
        df_row = df_event[df_event['isCol'] == False]
    
        ihit = 0
    
        for indc in df_col.index:
            for indr in df_row.index:
                col = df_col['location'][indc]
                row = df_row['location'][indr]
    
                col_ts = df_col['timestamp'][indc]
                row_ts = df_row['timestamp'][indr]
    
                col_tot = df_col['tot_us'][indc]
                row_tot = df_row['tot_us'][indr]
    
                if abs(col_ts - row_ts) >= timestamp_diff : continue
                if col_tot == 0 : continue
                if abs(col_tot - row_tot)/col_tot * 100 > tot_time_limit: continue


                ihit +=1
                n_hit += 1
                if col==inj_pixel[1] and inj_pixel[0]:
                    n_injhit += 1
                                
                #average_tot = (df_col['tot_us'][indc] + df_row['tot_us'][indr]) / 2
                #pair.append([ievt, ihit, df_col['location'][indc], df_row['location'][indr], average_tot])
        

    return n_evt, n_hit, n_injhit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drawing efficiecy from injection data')
    
    parser.add_argument('directory', default = "/Users/yoonha/cernbox/AstroPixv3_ANL/data_astropix-python/MultiPulse", help = 'directory path for input files')
    args = parser.parse_args()
    main(args)

    