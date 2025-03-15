#01/2025 Yoonha Hong at ANL
#change range if mode < 10
import os
import sys
import ROOT
import pandas as pd
import numpy as np
import re
from SingleInjection import make_TH1F
import array
import argparse
import matplotlib.pyplot as plt
from fit_utils import fit_histogram

ROW_MAX = 13  # 0-12
COL_MAX = 16  # 0-15

def create_new_map(directory, outdir):
    
    # 가능한 모든 Vinj 값을 찾기
    all_vinj = set()
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            match = re.search(r'(\d+(\.\d+)?)VInj', file)
            if match:
                vinj = int(float(match.group(1)) * 1000)
                all_vinj.add(vinj)
    
    all_vinj = sorted(list(all_vinj))  # 정렬된 Vinj 리스트
    
    # CSV 데이터를 저장할 리스트
    csv_data = []
    
    # 각 픽셀에 대해 처리
    for row in range(ROW_MAX):
        for col in range(COL_MAX):
            pixel = f"r{row}c{col}"
            print(f"처리 중: {pixel}")
            
            # 결과를 저장할 딕셔너리 초기화
            pixel_data = {
                'row': row,
                'col': col
            }
            # 각 Vinj에 대한 ToT mean을 0으로 초기화
            for vinj in all_vinj:
                pixel_data[f'ToT_mean_{vinj}mV'] = 0
            
            # 해당 픽셀의 모든 파일 찾기
            pixel_files = []
            for file in os.listdir(directory):
                if file.endswith(".csv") and f"r{row}c{col}" in file:
                    pixel_files.append(file)
            
            if not pixel_files:
                print(f"경고: {pixel}에 대한 파일을 찾을 수 없습니다")
                csv_data.append(pixel_data)  # 기본값(0)으로 저장
                continue
                
            # 데이터 수집
            tot_means = {}  # Vinj별 ToT mean 저장
            
            for file in pixel_files:
                match = re.search(r'(\d+(\.\d+)?)VInj', file)
                if match:
                    vinj = int(float(match.group(1)) * 1000)
                    csv_file = os.path.join(directory, file)
                    hist = make_TH1F(csv_file, row, col)
                    
                    if not hist: continue

                    fit_mean, _, _ = fit_histogram(hist, args.tot_thr)
                    tot_means[vinj] = fit_mean
            
            # 각 Vinj에서의 ToT mean 저장
            for vinj in all_vinj:
                pixel_data[f'ToT_mean_{vinj}mV'] = tot_means.get(vinj, 0)
            
            csv_data.append(pixel_data)
    
    # 결과 저장
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # CSV 파일로 저장
    df = pd.DataFrame(csv_data)
    dir_name = os.path.basename(os.path.normpath(directory))
    csv_path = f"{outdir}/InjectionMap_Results_{dir_name}.csv"
    df.to_csv(csv_path, index=False)
    
    return True

def create_qc_map(csv_file, outdir):
    try:
        df = pd.read_csv(csv_file)
        
        # ToT mean 컬럼만 선택하고 900mV 이하만 포함
        tot_columns = [col for col in df.columns if 'ToT_mean_' in col and 
                      int(col.split('_')[-1].replace('mV', '')) <= 1000]
        # Vinj 값 순서대로 정렬
        tot_columns.sort(key=lambda x: int(x.split('_')[-1].replace('mV', '')))
        
        # 모든 픽셀 범위 설정
        ROW_MAX = 13  # 0-12
        COL_MAX = 16  # 0-15
        
        # QC 맵을 위한 2D 배열 생성
        qc_map = np.zeros((ROW_MAX, COL_MAX), dtype=object)
        text_annotations = np.empty((ROW_MAX, COL_MAX), dtype=object)
        
        for _, row in df.iterrows():
            r, c = int(row['row']), int(row['col'])
            tot_values = [row[col] for col in tot_columns]
            
            # QC 조건 체크
            if all(v == 0 for v in tot_values):  # 모든 값이 0 (비어있는 경우)
                qc_map[r, c] = 'black'
                text_annotations[r, c] = ''
            elif all(v < args.tot_thr for v in tot_values):  # 모든 값이 ToT_THR 미만
                qc_map[r, c] = 'red'
                text_annotations[r, c] = ''
            elif any(v < args.tot_thr for v in tot_values):  # 일부 값이 ToT_THR 미만
                qc_map[r, c] = 'orange'
                low_count = sum(1 for v in tot_values if v < args.tot_thr)
                text_annotations[r, c] = str(low_count)
            else:  # 모든 값이 10 이상
                # vinj에 따른 증가 여부 확인
                is_increasing = True
                prev_val = 0
                for v in tot_values:
                    if v > 0:  # 0이 아닌 값에 대해서만 체크
                        if v < prev_val:
                            is_increasing = False
                            break
                        prev_val = v
                
                if is_increasing:
                    qc_map[r, c] = 'green'
                else:
                    qc_map[r, c] = 'white'
                text_annotations[r, c] = ''
        
        # 플롯 생성
        plt.figure(figsize=(12, 10))
        
        # 컬러맵 생성
        plt.imshow(np.zeros((ROW_MAX, COL_MAX)), cmap='binary', aspect='equal')
        
        # 각 픽셀에 대해 색상 및 텍스트 추가
        for r in range(ROW_MAX):
            for c in range(COL_MAX):
                plt.fill_between([c-0.5, c+0.5], [r-0.5, r-0.5], [r+0.5, r+0.5], 
                               color=qc_map[r, c])
                if text_annotations[r, c]:
                    plt.text(c, r, text_annotations[r, c], 
                           ha='center', va='center', color='black')
        
        # 축 레이블 및 그리드 설정
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.title('QC Map')
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        
        # 축 범위 설정
        plt.xlim(-0.5, COL_MAX-0.5)
        plt.ylim(-0.5, ROW_MAX-0.5)  # y축 반전
        
        # 범례 추가
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='black', label=f'Empty ({np.sum(qc_map=="black")/qc_map.size*100:.1f}%)'),
            plt.Rectangle((0,0),1,1, facecolor='red', label=f'All ToT < {args.tot_thr} ({np.sum(qc_map=="red")/qc_map.size*100:.1f}%)'),
            plt.Rectangle((0,0),1,1, facecolor='orange', label=f'Some ToT < {args.tot_thr} ({np.sum(qc_map=="orange")/qc_map.size*100:.1f}%)'),
            plt.Rectangle((0,0),1,1, facecolor='green', label=f'Good (Increasing) ({np.sum(qc_map=="green")/qc_map.size*100:.1f}%)'),
            plt.Rectangle((0,0),1,1, facecolor='white', label=f'Else ({np.sum(qc_map=="white")/qc_map.size*100:.1f}%)')
        ]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # 저장
        dir_name = os.path.basename(args.directory)
        plt.tight_layout()
        plt.savefig(f"{outdir}/QCMap_TU{args.tot_thr}_{dir_name}.pdf", bbox_inches='tight')
        
        return True
        
    except Exception as e:
        print(f"QC Map 생성 중 오류 발생: {str(e)}")
        return False



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Injection Map")
    parser.add_argument("directory", type=str, help="Directory containing CSV files")
    parser.add_argument("-o", "--outdir", type=str, default="./fig/InjectionMap",
                        help="Output directory")
    parser.add_argument("-t", "--tot_thr", type=int, default=4,
                        help="Threshold for ToT")
    args = parser.parse_args()
    
    dir_name = os.path.basename(os.path.normpath(args.directory))
    csv_path = f"{args.outdir}/InjectionMap_Results_{dir_name}.csv"
    
    # CSV 파일이 이미 존재하는지 확인
    if os.path.exists(csv_path):
        print(f"기존 CSV 파일을 불러옵니다: {csv_path}")
        create_qc_map(csv_path, args.outdir)
    
    else:
        print("새로운 CSV 파일을 생성합니다...")
        if create_new_map(args.directory, args.outdir):
            create_qc_map(csv_path, args.outdir)
