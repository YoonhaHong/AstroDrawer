import os
import re
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import glob
from matplotlib.colors import Normalize
import matplotlib as mpl
from utility import yaml_reader
from Multiprocess import parallel_process

plt.style.use('classic')

NCOL = 35
NROW = 35

def process_file(csv_file, outdir, timestampdiff, totdiff):
    t_start = time.time()

    pair = []
    file_name = os.path.basename(csv_file)
    dir_name = os.path.dirname(csv_file)
    if "_offline.csv" in file_name:
        file_name = file_name.rstrip('_offline.csv')
    elif ".csv" in file_name:
        file_name = file_name.rstrip('.csv')

    split_parts = file_name.split('_')
    date = split_parts[-1]  # 파일 이름에서 날짜 부분 추출
    name = split_parts[0:-1]

    pair = parallel_process(csv_file, timestampdiff, totdiff)
    print("... Matching is done!")

    ##### Find masked pixel and save it as pixs###########################################################
    findyaml = f"{dir_name}/*{date}*.yml"
    yamlpath = glob.glob(findyaml)

    if not yamlpath:
        print(f"No YAML file found for date: {date}")
        return

    disablepix = yaml_reader(yamlpath[0])  # YAML 파일 읽기
    pixs = pd.DataFrame(disablepix, columns=['col', 'row', 'disable'])  # 마스킹된 픽셀 정보를 DataFrame으로 변환
    navailpixs = pixs[pixs['disable'] == 0].shape[0]
    npixel = '%.2f' % ((navailpixs / NCOL / NROW) * 100.)
    print(f"{navailpixs}, {npixel}% active")

    ##### Create hit pixel dataframes #######################################################
    # Hit pixel information for all events
    dffpair = pd.DataFrame(pair, columns=['col', 'row',
                                          'timestamp_col', 'timestamp_row',
                                          'tot_us_col', 'tot_us_row', 'avg_tot_us'])
    # Create dataframe for number of hits
    dfpair = dffpair[['col', 'row']].copy()
    dfpairc = dfpair[['col', 'row']].value_counts().reset_index(name='hits')
    # How many hits are collected and shown in a plot
    nhits = dfpairc['hits'].sum()
    # mean of avg_tot_us, each col, row
    grouped_avg = dffpair.groupby(['col', 'row'])['avg_tot_us'].mean().reset_index(name='avg')
    print(grouped_avg)

    t_end = time.time()
    print(f"{t_end - t_start} Elapsed")

    row = 2
    col = 3
    fig, ax = plt.subplots(row, col, figsize=(20, 10))
    for irow in range(0, row):
        for icol in range(0, col):
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[irow, icol].spines[axis].set_linewidth(1.5)

    # 히트맵 그리기 (bins를 NCOL, NROW로 설정)
    p1 = ax[0, 0].hist2d(x=dfpairc['col'], y=dfpairc['row'], bins=[NCOL, NROW], range=[[0, NCOL], [0, NROW]],
                         weights=dfpairc['hits'], cmap='YlOrRd', cmin=1.0, norm=matplotlib.colors.LogNorm())
    fig.colorbar(p1[3], ax=ax[0, 0]).set_label(label='Hit Counts', weight='bold', size=14)
    ax[0, 0].grid()
    ax[0, 0].set_xlabel('Col', fontweight='bold', fontsize=14)
    ax[0, 0].set_ylabel('Row', fontweight='bold', fontsize=14)
    ax[0, 0].xaxis.set_tick_params(labelsize=14)
    ax[0, 0].yaxis.set_tick_params(labelsize=14)

    # 마스킹 맵 그리기 (bins를 NCOL, NROW로 설정)
    p2 = ax[0, 1].hist2d(x=pixs['col'], y=pixs['row'], bins=[NCOL, NROW], range=[[0, NCOL], [0, NROW]],
                         weights=pixs['disable'],
                         norm=Normalize(vmin=0, vmax=1), cmap='Greys')
    fig.colorbar(p2[3], ax=ax[0, 1]).set_label(label='Masked', weight='bold', size=14)
    ax[0, 1].grid()
    ax[0, 1].set_xlabel('Col', fontweight='bold', fontsize=14)
    ax[0, 1].set_ylabel('Row', fontweight='bold', fontsize=14)
    ax[0, 1].xaxis.set_tick_params(labelsize=14)
    ax[0, 1].yaxis.set_tick_params(labelsize=14)

    # 히트맵 + 마스킹 맵 겹쳐서 그리기 (bins를 NCOL, NROW로 설정)
    p3 = ax[0, 2].hist2d(x=pixs['col'], y=pixs['row'], bins=[NCOL, NROW], range=[[0, NCOL], [0, NROW]],
                         weights=pixs['disable'],
                         norm=Normalize(vmin=0, vmax=1), cmap='Greys')
    p3 = ax[0, 2].hist2d(x=dfpairc['col'], y=dfpairc['row'], bins=[NCOL, NROW], range=[[0, NCOL], [0, NROW]],
                         weights=dfpairc['hits'], cmap='YlOrRd', cmin=1.0)
    fig.colorbar(p3[3], ax=ax[0, 2]).set_label(label='Hit Counts', weight='bold', size=14)
    ax[0, 2].grid()
    ax[0, 2].set_xlabel('Col', fontweight='bold', fontsize=14)
    ax[0, 2].set_ylabel('Row', fontweight='bold', fontsize=14)
    ax[0, 2].xaxis.set_tick_params(labelsize=14)
    ax[0, 2].yaxis.set_tick_params(labelsize=14)

    # 평균 ToT 맵 그리기 (bins를 NCOL, NROW로 설정)
    p4 = ax[1, 0].hist2d(x=grouped_avg['col'], y=grouped_avg['row'], bins=[NCOL, NROW], range=[[0, NCOL], [0, NROW]],
                         weights=grouped_avg['avg'], cmap='Blues', vmin=0.0)
    fig.colorbar(p4[3], ax=ax[1, 0]).set_label(label='Avg.ToT [us]', weight='bold', size=14)
    ax[1, 0].grid()
    ax[1, 0].set_xlabel('Col', fontweight='bold', fontsize=14)
    ax[1, 0].set_ylabel('Row', fontweight='bold', fontsize=14)
    ax[1, 0].xaxis.set_tick_params(labelsize=14)
    ax[1, 0].yaxis.set_tick_params(labelsize=14)

    # 전체 픽셀의 ToT 분포 그리기
    p5 = ax[1, 1].hist(x=dffpair['avg_tot_us'], bins=60, range=(0, 30), color='blue', edgecolor='black')
    ax[1, 1].grid()
    ax[1, 1].set_xlabel('ToT [us]', fontweight='bold', fontsize=14)
    ax[1, 1].set_ylabel('Counts', fontweight='bold', fontsize=14)
    ax[1, 1].xaxis.set_tick_params(labelsize=14)
    ax[1, 1].yaxis.set_tick_params(labelsize=14)

    # 텍스트 정보 추가
    ax[1, 2].set_axis_off()
    ax[1, 2].text(0.1, 0.85, f"Directory: {dir_name}", fontsize=15, fontweight='bold')
    ax[1, 2].text(0.1, 0.80, f"Filename: {file_name}", fontsize=15, fontweight='bold')
    ax[1, 2].text(0.1, 0.40, f"Available Pixels: {npixel}%", fontsize=15, fontweight='bold')
    ax[1, 2].text(0.1, 0.60, "Processed below", fontsize=15, fontweight='bold')
    ax[1, 2].text(0.1, 0.55, f"conditions: <{timestampdiff} timestamp and <{totdiff}% in ToT", fontsize=15, fontweight='bold')
    ax[1, 2].text(0.1, 0.45, f"nhits: {nhits}", fontsize=15, fontweight='bold')

    ax[0, 0].set_title(f"Hit Map", fontweight='bold', fontsize=14)
    ax[0, 1].set_title(f"Masked pixel", fontweight='bold', fontsize=14)
    ax[0, 2].set_title(f"Hit Map with Masked pixels", fontweight='bold', fontsize=14)
    ax[1, 0].set_title(f"Avg.ToT per pixel", fontweight='bold', fontsize=14)
    ax[1, 1].set_title(f"Avg.ToT for all pixels", fontweight='bold', fontsize=14)

    figdir = outdir if outdir else dir_name
    os.makedirs(figdir, exist_ok=True)
    plt.savefig(f"{figdir}/{file_name}.png")
    print(f"Saved at {figdir}/{file_name}.png")
    plt.close()  # 메모리 누수 방지를 위해 플롯 창 닫기

def main(args):
    input_path = args.inputfile
    outdir = args.outdir
    timestampdiff = args.timestampdiff
    totdiff = args.totdiff

    # 입력 경로가 디렉토리인 경우
    if os.path.isdir(input_path):
        csv_files = glob.glob(os.path.join(input_path, "*.csv"))
        for csv_file in csv_files:
            print(f"Processing file: {csv_file}")
            process_file(csv_file, outdir, timestampdiff, totdiff)
    # 입력 경로가 파일인 경우
    elif os.path.isfile(input_path):
        print(f"Processing file: {input_path}")
        process_file(input_path, outdir, timestampdiff, totdiff)
    else:
        print(f"Invalid input path: {input_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Hit Map with Masked Pixels from CSV file - MultiProcessing version')
    parser.add_argument("inputfile", type=str, help='Path to the input CSV file or directory')
    parser.add_argument('-o', '--outdir', default=None, help='Output directory for the plot')
    parser.add_argument('-td', '--timestampdiff', type=float, required=False, default=2,
                        help='difference in timestamp in pixel matching (default:col.ts-row.ts<2)')
    parser.add_argument('-tot', '--totdiff', type=float, required=False, default=10,
                        help='error in ToT[us] in pixel matching (default:(col.tot-row.tot)/col.tot<10%)')

    args = parser.parse_args()
    main(args)




