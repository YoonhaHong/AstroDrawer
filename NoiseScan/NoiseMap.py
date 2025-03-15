#01/2025 Yoonha Hong at ANL

import os
import sys
import ROOT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import argparse

ROOT.gStyle.SetOptStat(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='노이즈 맵 생성')
    parser.add_argument('directory', help='데이터 디렉토리 경로')
    parser.add_argument('-o', '--outdir', type=str, default="./fig", help='출력 디렉토리 경로')
    args = parser.parse_args()

    # 캔버스 생성
    canvas = ROOT.TCanvas("canvas", "Noise Map", 800, 600)
    canvas.SetMargin(0.15, 0.15, 0.12, 0.12)

    # 2D 히스토그램 생성 (13x16 픽셀)
    hist = ROOT.TH2F("hist", "Noise Map;Column;Row", 16, 0, 16, 13, 0, 13)

    # 각 픽셀의 히트 수 계산
    for row in range(13):
        for col in range(16):
            # r0c0_날짜.csv 형식의 파일 찾기
            pattern = f"r{row}c{col}_*.csv"
            matching_files = glob.glob(os.path.join(args.directory, pattern))
            
            if not matching_files:  continue

            file_path = matching_files[0]  # 첫 번째 매칭 파일 사용
            df = pd.read_csv(file_path)
            if df.empty: 
                n_hits = 0
            else:
                filtered_data = df[(df['id'] == 0) & 
                          (df['payload'] == 7) & 
                          (df['row'] == row) & 
                          (df['col'] == col)]
                n_hits = len(filtered_data)

            hist.SetBinContent(col+1, row+1, n_hits)

    # 히스토그램 스타일 설정
    hist.SetMinimum(0)
    hist.GetXaxis().SetNdivisions(16)
    hist.GetYaxis().SetNdivisions(13)
    hist.GetXaxis().CenterLabels()
    hist.GetYaxis().CenterLabels()
    hist.GetZaxis().SetTitle("Number of hits")

    # 컬러 팔레트 설정
    ROOT.gStyle.SetPalette(ROOT.kBird)

    # 히스토그램 그리기
    hist.Draw("COLZ")

    # 각 빈에 값 표시
    labels = []
    for row in range(13):
        for col in range(16):
            value = hist.GetBinContent(col+1, row+1)
            if value > 0:
                labels.append(ROOT.TText(col+0.5, row+0.5, f"{int(value)}"))
                labels[-1].SetTextColor(ROOT.kRed)
                labels[-1].SetTextAlign(22)
                labels[-1].SetTextSize(0.02)
                labels[-1].Draw("SAME")

    # 출력 디렉토리 생성
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # 파일 저장
    dir_name = os.path.basename(os.path.normpath(args.directory))
    canvas.SaveAs(f"{args.outdir}/NoiseMap_{dir_name}.pdf")
