#01/2025 Yoonha Hong at ANL
#csv 파일 두 개를 인자로 받음, row, col도 선택
#두 파일로부터 ToT_us의 1D histogram을 그리고, gaussian fitting하여 비교

import pandas as pd
import ROOT
import argparse
import re
import os

def make_TH1F(csv_file, row, col, color=ROOT.kBlack):
    # CSV 파일 읽기
    df = pd.read_csv(csv_file, sep=',')
    filtered_data = df[(df['id'] == 0) & 
                       (df['payload'] == 7) & 
                       (df['row'] == row) & 
                       (df['col'] == col)]
    
    if filtered_data.empty:
        print(f"No data matches the given criteria in {csv_file}")
        return None
        
    tot_us_values = filtered_data['tot_us'].to_numpy()

    hist_name = os.path.basename(csv_file)
    hist = ROOT.TH1F(hist_name, f"ToT Comparison;ToT (us);Counts", 500, 0, 500)
    
    hist.SetLineColor(color)
    hist.SetMarkerColor(color)
    hist.SetMarkerStyle(20)
    
    for value in tot_us_values:
        hist.Fill(value)
    
    return hist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ToT Histograms from two CSV files")
    parser.add_argument("csv_file1", type=str, help="Path to the first CSV file")
    parser.add_argument("csv_file2", type=str, help="Path to the second CSV file")
    parser.add_argument("-r", "--row", type=int, default=0, help="Row value to filter")
    parser.add_argument("-c", "--col", type=int, default=0, help="Column value to filter")
    parser.add_argument("-o", "--output", type=str, default="tot_histogram_comparison.pdf", 
                       help="Output file for the histogram")
    
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)
    ROOT.gStyle.SetStatStyle(0)
    ROOT.gStyle.SetStatColor(0)

    args = parser.parse_args()

    canvas = ROOT.TCanvas("canvas", "ToT Histogram Comparison", 800, 600)
    frame = canvas.DrawFrame(0, 0, 250, 500)
    frame.SetTitle("ToT Comparison;ToT (us);Counts")
    
    # 첫 번째 히스토그램 (빨간색)
    hist1 = make_TH1F(args.csv_file1, args.row, args.col, ROOT.kRed)
    if hist1:
        # DrawFrame으로 빈 프레임 먼저 그리기
        
        # 히스토그램 그리기
        hist1.Draw("SAME")
        mean1 = hist1.GetMean()
        sigma1 = hist1.GetRMS()
        fit1 = ROOT.TF1("gaus1", "gaus", mean1 - 5*sigma1, mean1 + 5*sigma1)
        fit1.SetLineColor(ROOT.kRed)
        #hist1.Fit(fit1, "RQ")
        
        # 피팅 결과 출력
        fit_mean1 = fit1.GetParameter(1)
        fit_sigma1 = fit1.GetParameter(2)
        print(f"File 1 - Fit Mean: {fit_mean1:.2f}, Fit Sigma: {fit_sigma1:.2f}")

    # 두 번째 히스토그램 (파란색)
    hist2 = make_TH1F(args.csv_file2, args.row, args.col, ROOT.kBlue)
    if hist2:
        hist2.Draw("SAME")
        mean2 = hist2.GetMean()
        sigma2 = hist2.GetRMS()
        fit2 = ROOT.TF1("gaus2", "gaus", mean2 - 5*sigma2, mean2 + 5*sigma2)
        fit2.SetLineColor(ROOT.kBlue)
        #hist2.Fit(fit2, "RQ")
        
        # 피팅 결과 출력
        fit_mean2 = fit2.GetParameter(1)
        fit_sigma2 = fit2.GetParameter(2)
        print(f"File 2 - Fit Mean: {fit_mean2:.2f}, Fit Sigma: {fit_sigma2:.2f}")

    # 범례 추가
    legend = ROOT.TLegend(0.5, 0.7, 0.9, 0.9)
    legend.AddEntry(hist1, os.path.basename(args.csv_file1), "l")
    legend.AddEntry(hist2, os.path.basename(args.csv_file2), "l")
    legend.Draw()

    canvas.RedrawAxis()
    canvas.Update()
    canvas.SaveAs(args.output)