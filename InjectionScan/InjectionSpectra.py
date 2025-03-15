#01/2025 Yoonha Hong at ANL

import os
import sys
import ROOT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
from SingleInjection import make_TH1F

TOT_MAX = 400
Y_MAX = 700

HV = -250
THR = 180

ROOT.gStyle.SetOptStat(1111)  # StatBox에 모든 통계 표시
ROOT.gStyle.SetOptFit(1111)   # StatBox에 피팅 결과 표시
# 메인 함수
if __name__ == "__main__":
    directory = sys.argv[1]  # 첫 번째 인자로 받은 디렉토리 경로
    parser = argparse.ArgumentParser(description='InjectionSpectra 분석')
    parser.add_argument('directory', help='데이터 디렉토리 경로')
    parser.add_argument('-o', '--outdir', type=str, default="./fig/SinglePixel", help='출력 디렉토리 경로')
    parser.add_argument('-r', '--row', type=int, default=0, help='분석할 row 위치 (기본값: 0)')
    parser.add_argument('-c', '--col', type=int, default=0, help='분석할 column 위치 (기본값: 0)')
    args = parser.parse_args()
    
    row = args.row
    col = args.col
    dir_name = os.path.basename(os.path.normpath(directory))

    # 디렉토리 내 모든 CSV 파일 찾기
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]

    # 캔버스 생성
    canvas = ROOT.TCanvas("canvas", "Canvas for multiple histograms", 800, 600)
    ROOT.gPad.SetMargin(0.1, 0.03, 0.1, 0.03)
    
    frame = ROOT.gPad.DrawFrame(0, 0, TOT_MAX, Y_MAX)
    frame.GetXaxis().SetTitle("ToT [us]")

    # 색상 설정을 위한 변수
    colors = [ROOT.kRed, ROOT.kOrange, ROOT.kYellow+1, ROOT.kGreen+1,ROOT.kCyan+1, 
              ROOT.kBlue, ROOT.kMagenta, ROOT.kViolet, ROOT.kBlack]
    legend = ROOT.TLegend(0.6, 0.2, 0.97, 0.97)
    legend.SetHeader(f"r{row}c{col}")
    legend.SetTextSize(0.03)
    # 각 CSV 파일에 대해 히스토그램을 만들고 그리기
    histograms = {}
    i=0

    for file in os.listdir(directory):
        if not file.endswith(".csv"):
            continue
        if not file.startswith(f"r{row}c{col}"):
            continue
        match = re.search(r'(\d+(\.\d+)?)VInj', file)
        if match:
            vinj = int(float(match.group(1)) * 1000)
            
            # CSV 파일 읽기
            csv_file = os.path.join(directory, file)
            hist = make_TH1F(csv_file, row, col)
            if hist:  # 히스토그램이 생성된 경우만 추가
                histograms[vinj] = hist 
    n = len(histograms)
    x = np.zeros(n)
    y = np.zeros(n)
    y_err = np.zeros(n)
    
    

    # 여러 히스토그램을 하나의 캔버스에 그리기
    for vinj in sorted(histograms.keys()):
        hist = histograms[vinj]
        hist.SetLineColor(colors[i % len(colors)])  # 색상 설정
        hist.Draw("SAME")

        #mean = hist.GetMean()
        #sigma = hist.GetRMS()
        #fit = ROOT.TF1("gaus", "gaus", mean - 5*sigma, mean + 5*sigma)
        mode = hist.GetBinCenter(hist.GetMaximumBin())
        fit_min = mode - 15
        fit_max = mode + 15

        fit = ROOT.TF1("gaus", "gaus", fit_min, fit_max)
        fit.SetLineColor(colors[i % len(colors)])  # 색상 설정
        fit.SetLineWidth(1)
        hist.Fit(fit, "RQ")

        fit_mean = fit.GetParameter(1)
        fit_sigma = fit.GetParameter(2)
        x[i] = vinj
        y[i] = fit_mean
        y_err[i] = fit_sigma

        legend.AddEntry(hist, f"{vinj} mV: #mu={fit_mean:.1f}, #sigma={fit_sigma:.1f}", "l")
        i+=1
    legend.SetFillStyle(0)
    legend.Draw("SAME")
    ROOT.gPad.RedrawAxis()
    canvas.Update()
    canvas.SaveAs(f"{args.outdir}/InjectionSpectra_HV{HV}_THR{THR}_r{row}c{col}_{dir_name}.pdf")


    # 캔버스 생성
    canvas2 = ROOT.TCanvas("canvas2", "Injection Voltage vs ToT mean", 800, 600)
    graph = ROOT.TGraphErrors(len(x), x, y, 0, y_err)

    # 피팅 함수 정의: f(x) = ax + b(1-exp(-x/c)) + d
    fit_func = ROOT.TF1("fitFunc", "[0]*x + [1]*(1-exp(-x/[2])) + [3]", min(x), max(x))

    # 초기 매개변수 설정
    fit_func.SetParameters(0.1, 1.0, 100.0, 0.0)  # a, b, c, d의 초기값
    fit_func.SetParNames("a", "b", "c", "d")

    # 그래프 스타일 설정
    graph.SetMarkerStyle(20)
    graph.SetMarkerColor(1)
    graph.SetLineColor(1)
    graph.SetTitle("Injection Voltage vs ToT mean from gaussian fitting")
    graph.GetXaxis().SetTitle("Vinj [mV]")
    graph.GetYaxis().SetTitle("ToT mean [us]")

    # 피팅 수행
    graph.Fit(fit_func, "RQ")  # "R"은 설정된 범위 내에서 피팅

    # 수식 문자열 생성


    canvas2.SetGrid()

    # 그래프 그리기
    graph.Draw("AP")
    canvas2.Modified()
    canvas2.Update()
    stats = graph.GetListOfFunctions().FindObject("stats")
    if stats:
        stats.SetX1NDC(0.60)  # 오른쪽
        stats.SetX2NDC(0.90)
        stats.SetY1NDC(0.15)  # 아래쪽
        stats.SetY2NDC(0.45)
    else:
        print("stats not found")
    latex = ROOT.TLatex()
    latex.SetTextColor(1)
    formula = f"fit func. = ax + b(1-e^{{-x/c}}) + d"
    latex.DrawLatexNDC(0.47, 0.50, formula)  
    # directory 이름에서 마지막 폴더명만 추출

    canvas2.SaveAs(f"{args.outdir}/InjectionGraph_HV{HV}_THR{THR}_r{row}c{col}_{dir_name}.pdf")

