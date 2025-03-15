#01/2025 Yoonha Hong at ANL

import os
import sys
import ROOT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from SingleInjection import make_TH1F
from fit_utils import fit_histogram
import array
import argparse

TOT_MAX = 400
Y_MAX = 700

HV = -250
THR = 180

ROOT.gStyle.SetOptStat(0)  # StatBox에 모든 통계 표시
ROOT.gStyle.SetOptFit(0)   # StatBox에 피팅 결과 표시

# 메인 함수
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Injection spectra analysis for multiple pixels")
    parser.add_argument("directory", type=str, help="Directory containing CSV files")
    parser.add_argument("-t", "--tot_thr", type=int, default=4,
                        help="ToT threshold for fitting")
    parser.add_argument("-o", "--outdir", type=str, default="./fig/MultiPixels",
                        help="Output directory")
    parser.add_argument("-R", "--row-range", nargs=2, type=int, default=[0, 5],
                        help="Row range (e.g., '0 5' for rows 0 to 5)")
    parser.add_argument("-C", "--col-range", nargs=2, type=int, default=[0, 3],
                        help="Column range (e.g., '0 3' for columns 0 to 3)")
    args = parser.parse_args()

    # row와 column 범위 설정
    def get_range(range_args):
        if range_args is None:
            return None
        start, end = range_args
        return range(start, end + 1)

    row_range = get_range(args.row_range)
    col_range = get_range(args.col_range)
    
    # 픽셀별로 파일을 분류할 딕셔너리 생성
    pixel_files = {}

    
    # 색상 설정
    colors = [ROOT.kRed, ROOT.kOrange, ROOT.kYellow+1, ROOT.kGreen+1, ROOT.kBlue, 
              ROOT.kMagenta, ROOT.kViolet, ROOT.kBlack]
    
    # 픽셀별 마커 스타일 설정
    markers = [20, 21, 22, 23, 24, 25, 26]
    

    dict_graph = {}
    
    pixel_count = 0

    for row in row_range:
        for col in col_range:
            files = [f for f in os.listdir(args.directory) if f.endswith(".csv") and f.startswith(f"r{row}c{col}_")]
            pixel_files[f"r{row}c{col}"] = files 

            histograms = {}
            x_data = []
            y_data = []
            y_err_data = [] 


            for v in files:
                match = re.search(r'(\d+(\.\d+)?)VInj', v)
                if match:
                    vinj = int(float(match.group(1)) * 1000)
                    hist = make_TH1F(os.path.join(args.directory, v), row, col)
                    histograms[vinj] = hist
            for vinj in sorted(histograms.keys()):
                hist = histograms[vinj]
                hist.SetLineColor(colors[pixel_count % len(colors)])

                fit_mean, fit_sigma, fit = fit_histogram(hist, args.tot_thr)
                x_data.append(vinj)
                y_data.append(fit_mean)
                y_err_data.append(fit_sigma)

            dict_graph[f"r{row}c{col}"] = {
                'x': np.array(x_data),
                'y': np.array(y_data),
                'y_err': np.array(y_err_data)
            }
        pixel_count += 1


    print(f"분석할 픽셀: {pixel_files.keys()}")
    # 두 번째 캔버스 (Injection Voltage vs ToT mean)
    canvas2 = ROOT.TCanvas("canvas2", "", 800, 600)
    canvas2.SetGrid()
    canvas2.SetMargin(0.12, 0.05, 0.12, 0.05)

    frame2 = ROOT.gPad.DrawFrame(200, -100, 1000, 500)
    frame2.GetXaxis().SetTitle("Vinj [mV]")
    frame2.GetYaxis().SetTitle("ToT mean [us]") 
    frame2.Draw()

    multi_graph = ROOT.TMultiGraph()
    
    legend2 = ROOT.TLegend(0.15, 0.64, 0.45, 0.94)
    
    pixel_count = 0
    for pixel in dict_graph.keys():
        data = dict_graph[pixel]
        n_points = len(data['x'])
        
        # 데이터가 비어있는지 확인
        if n_points == 0:
            print(f"경고: {pixel}에 대한 데이터가 없습니다.")
            continue
            
        # double 타입의 array로 변환
        x_arr = array.array('d', [float(x) for x in data['x']])
        y_arr = array.array('d', [float(y) for y in data['y']])
        x_err = array.array('d', [0.0] * n_points)  # x 오차는 0으로 설정
        y_err = array.array('d', [float(err) for err in data['y_err']])
        
        try:
            # TGraphErrors 생성
            graph = ROOT.TGraphErrors(n_points, x_arr, y_arr, x_err, y_err)
            
            # 그래프가 성공적으로 생성되었는지 확인
            if not graph:
                print(f"경고: {pixel}에 대한 그래프 생성 실패")
                continue
                
            # 그래프 스타일 설정
            graph.SetMarkerStyle(markers[pixel_count % len(markers)])
            graph.SetMarkerColor(colors[(pixel_count + 1) % len(colors)])
            graph.SetLineColor(colors[(pixel_count + 1) % len(colors)])
            
            # 피팅 함수
            fit_func = ROOT.TF1(f"fitFunc_{pixel}", "[0]*x + [1]*(1-exp(-x/[2])) + [3]", 
                               min(data['x']), max(data['x']))
            fit_func.SetParameters(0.04, 300, 300, -300)
            fit_func.SetParNames("a", "b", "c", "d")
            fit_func.SetLineColor(colors[(pixel_count + 1) % len(colors)])
            graph.Fit(fit_func, "RQ")
            
            multi_graph.Add(graph)
            legend2.AddEntry(graph, pixel, "p")
            pixel_count += 1
            
        except Exception as e:
            print(f"오류 발생: {pixel} 처리 중 - {str(e)}")
            continue

    multi_graph.Draw("SAME AP")
    multi_graph.GetXaxis().SetTitle("Vinj [mV]")
    multi_graph.GetYaxis().SetTitle("ToT mean [us]")
    multi_graph.GetXaxis().SetRangeUser(200, 1000)
    multi_graph.GetYaxis().SetRangeUser(-100, 500)

    legend2.SetNColumns(5)
    legend2.SetFillStyle(0)
    legend2.Draw("SAME")
    
    latex = ROOT.TLatex()
    latex.SetTextColor(1)
    formula = "fit func. = ax + b(1-e^{-x/c}) + d"
    #latex.DrawLatexNDC(0.47, 0.50, formula)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    dir_name = os.path.basename(os.path.normpath(args.directory))
    if args.row_range[0] == args.row_range[1]:
        file_name = f"InjectionMultiGraph_{dir_name}_R{args.row_range[0]}_C{args.col_range[0]}-{args.col_range[1]}.pdf"
    else:
        file_name = f"InjectionMultiGraph_{dir_name}_R{args.row_range[0]}-{args.row_range[1]}_C{args.col_range[0]}-{args.col_range[1]}.pdf"
    
    if args.col_range[0] == args.col_range[1]:
        file_name = f"InjectionMultiGraph_{dir_name}_R{args.row_range[0]}_C{args.col_range[0]}.pdf"
    else:
        file_name = f"InjectionMultiGraph_{dir_name}_R{args.row_range[0]}-{args.row_range[1]}_C{args.col_range[0]}-{args.col_range[1]}.pdf"

    canvas2.SaveAs(f"{args.outdir}/{file_name}") 

