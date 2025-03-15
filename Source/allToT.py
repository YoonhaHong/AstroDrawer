import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import ROOT
import time

def tot_distribution_root(csv_file, exclude_pixels):
    df = pd.read_csv(csv_file)
    
    # 제외할 픽셀들을 필터링
    mask = pd.Series(True, index=df.index)  # 초기 마스크: 모든 데이터를 포함
    for row, col in exclude_pixels:
        mask &= ~((df["row"] == row) & (df["col"] == col))  # 제외할 픽셀을 마스크에서 제거
    
    # 제외된 데이터를 필터링
    filtered_data = df[mask]["tot_us"]
    
    # ROOT 히스토그램 생성
    hist = ROOT.TH1F("tot_us_dist", "ToT Distribution (Excluded Pixels)", 150, 0, 300)
    
    # 히스토그램에 데이터 채우기
    for value in filtered_data:
        hist.Fill(value)
    
    return hist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ToT distribution from CSV excluding specific pixels.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file")
    parser.add_argument('-e', "--exclude", nargs='+', required=True, help="List of pixels to exclude in format 'row1,col1 row2,col2 ...'")
    
    args = parser.parse_args()
    
    # 제외할 픽셀 리스트 파싱
    exclude_pixels = []
    for pixel in args.exclude:
        row, col = map(int, pixel.split(','))
        exclude_pixels.append((row, col))
    
    t_start = time.time()
    
    # ROOT 캔버스 생성
    canvas = ROOT.TCanvas("c1", "ToT Distribution", 800, 600)
    
    # ToT 분포 히스토그램 생성
    hist = tot_distribution_root(args.csv_file, exclude_pixels)
    
    # 히스토그램 설정
    hist.GetXaxis().SetTitle("ToT (us)")
    hist.GetYaxis().SetTitle("Entries")
    hist.SetLineColor(ROOT.kBlue)
    hist.SetFillColorAlpha(ROOT.kBlue, 0.3)
    hist.Draw()
    
    # 출력
    canvas.Update()
    
    # 결과 저장 경로 생성
    os.makedirs("./fig", exist_ok=True)
    output_file = f"./fig/{os.path.basename(args.csv_file)[:-4]}_except_"
    output_file += "_".join([f"r{row}c{col}" for row, col in exclude_pixels]) + "_tot_hist.pdf"
    canvas.SaveAs(output_file)
    
    t_end = time.time()
    print(f"Elapsed time: {t_end - t_start}")
    print(f"Saved histogram to {output_file}")
    input("Press Enter to exit...")  # 창 유지