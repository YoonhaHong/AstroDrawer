#01/2025 Yoonha Hong at ANL
#csv 파일 한 개를 인자로 받음, row, col도 선택
#해당 파일로 부터 ToT_us의 1D histogram을 그리고, gaussian fitting

import pandas as pd
import ROOT
import argparse
import re
import os

def make_TH1F(csv_file, row, col):
    # CSV 파일 읽기
    try:
        df = pd.read_csv(csv_file, sep=',')
        
        # 빈 파일 체크
        if df.empty:
            print(f"경고: {csv_file}이 비어 있습니다.")
            return None
            
        # 조건 필터링: id==0, payload==7, 특정 row와 col
        filtered_data = df[(df['id'] == 0) & 
                          (df['payload'] == 7) & 
                          (df['row'] == row) & 
                          (df['col'] == col)]
        
        if filtered_data.empty:
            print(f"경고: {csv_file}에서 조건에 맞는 데이터를 찾을 수 없습니다.")
            return None
            
        tot_us_values = filtered_data['tot_us'].to_numpy()
        
        hist_name = os.path.basename(csv_file)
        hist = ROOT.TH1F(hist_name, f" ;ToT (us);Counts", 500, 0, 500)
        
        for value in tot_us_values:
            hist.Fill(value)
        
        return hist
        
    except pd.errors.EmptyDataError:
        print(f"경고: {csv_file}이 비어 있거나 읽을 수 없습니다.")
        return None
    except Exception as e:
        print(f"오류 발생: {csv_file} 처리 중 - {str(e)}")
        return None

def sumToT_TH1F(csv_file, row, col):
    try:
        df = pd.read_csv(csv_file, sep=',')
    
        # 빈 파일 체크
        if df.empty:
            print(f"경고: {csv_file}이 비어 있습니다.")
            return None
            
        # 조건 필터링: id==0, payload==7, 특정 row와 col
        filtered_data = df[(df['id'] == 0) & 
                            (df['payload'] == 7) & 
                            (df['row'] == row) & 
                            (df['col'] == col)]
        
        if filtered_data.empty:
            print(f"경고: {csv_file}에서 조건에 맞는 데이터를 찾을 수 없습니다.")
            return None
            
        
        hist_name = os.path.basename(csv_file)
        hist = ROOT.TH1F(hist_name, f" ;ToT (us);Counts", 500, 0, 500)

        event_starts = filtered_data[filtered_data["dec_order"] == 0].index.tolist()
        event_starts.append(len(filtered_data))
        for i in range(len(event_starts) - 1):
            event_tot_sum = filtered_data["tot_us"].iloc[event_starts[i]:event_starts[i+1]].sum()
            hist.Fill(event_tot_sum)
        
        return hist
        
    except pd.errors.EmptyDataError:
        print(f"경고: {csv_file}이 비어 있거나 읽을 수 없습니다.")
        return None
    except Exception as e:
        print(f"오류 발생: {csv_file} 처리 중 - {str(e)}")
        return None

def recaltot_TH1F(csv_file, row, col):
    try:
        df = pd.read_csv(csv_file, sep=',')

        # 빈 파일 체크
        if df.empty:
            print(f"경고: {csv_file}이 비어 있습니다.")
            return None
            
        # 조건 필터링: id==0, payload==7, 특정 row와 col
        filtered_data = df[(df['id'] == 0) & 
                            (df['payload'] == 7) & 
                            (df['row'] == row) & 
                            (df['col'] == col)]
        
        if filtered_data.empty:
            print(f"경고: {csv_file}에서 조건에 맞는 데이터를 찾을 수 없습니다.")
            return None
            
        
        hist_name = os.path.basename(csv_file)
        hist = ROOT.TH1F(hist_name, f" ;ToT (us);Counts", 500, 0, 500)
        
        # 이벤트 시작 인덱스 찾기 (dec_order == 0)
        event_starts = filtered_data[filtered_data["dec_order"] == 0].index.tolist()
        event_starts.append(len(filtered_data))  # 마지막 이벤트의 끝을 위해 전체 길이를 추가
        
        for i in range(len(event_starts) - 1):
            event_data = filtered_data.iloc[event_starts[i]:event_starts[i+1]]  # 이벤트 데이터 추출
            
            ts_dec1 = event_data[event_data["dec_order"] == 0]["ts_dec1"].values
            ts_dec2 = event_data[event_data["dec_order"] == event_data["dec_order"].max()]["ts_dec2"].values
            
            if len(ts_dec1) > 0 and len(ts_dec2) > 0:
                tot_us = (ts_dec2[0] - ts_dec1[0]) / 20
                hist.Fill(tot_us)
        return hist
        
    except pd.errors.EmptyDataError:
        print(f"경고: {csv_file}이 비어 있거나 읽을 수 없습니다.")
        return None
    except Exception as e:
        print(f"오류 발생: {csv_file} 처리 중 - {str(e)}")
        return None
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ToT Histogram from CSV")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file")
    
    ROOT.gStyle.SetOptStat(1111)  # StatBox에 모든 통계 표시
    ROOT.gStyle.SetOptFit(1111)   # StatBox에 피팅 결과 표시

    #ROOT.gStyle.SetStatStyle(0)           # StatBox 스타일 설정
    #ROOT.gStyle.SetStatBorderSize(0)      # 테두리 크기 제거
    #ROOT.gStyle.SetStatColor(0) 

    args = parser.parse_args()

    # csv 파일 이름에서 row, col 추출
    filename = os.path.basename(args.csv_file)
    row = col = 0
    match = re.search(r'r(\d+)c(\d+)', filename)
    if match:
        row = int(match.group(1))
        col = int(match.group(2))

    canvas = ROOT.TCanvas("canvas", "ToT Histogram", 800, 600)
    canvas.SetGrid()
    canvas.SetMargin(0.12, 0.05, 0.12, 0.05)
    hist = make_TH1F(args.csv_file, row, col)

    # 히스토그램 그리기
    hist.Draw("SAME")

    # Gaussian 피팅
    #mean = hist.GetMean()
    #sigma = hist.GetRMS()
    #fit = ROOT.TF1("gaus", "gaus", mean - 5*sigma, mean + 5*sigma)
    mpv = hist.GetBinCenter(hist.GetMaximumBin())
    fit = ROOT.TF1("gaus", "gaus", mpv - 15, mpv + 15)
    fit.SetLineColorAlpha(ROOT.kRed, 0.8)
    fit.SetLineWidth(1)

    hist.Fit(fit, "R")
    # 피팅 결과 출력
    fit_mean = fit.GetParameter(1)
    fit_sigma = fit.GetParameter(2)
    print(f"Fit Mean: {fit_mean}, Fit Sigma: {fit_sigma}")
    # 캔버스 저장
    canvas.SaveAs(f"./ToTHist_{filename}.pdf")

