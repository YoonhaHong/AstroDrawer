#01/2025 Yoonha Hong at ANL
#두 디렉토리를 인자로 받아서 InjectionSpectra 비교

import os
import sys
import ROOT
import pandas as pd
import numpy as np
import re
from SingleInjection import make_TH1F

TOT_MAX = 400
Y_MAX = 700

HV = -250
THR = 180

def process_directory(directory):
    histograms = {}
    colors = [ROOT.kRed, ROOT.kOrange, ROOT.kYellow+1, ROOT.kGreen+1,ROOT.kBlue, 
              ROOT.kMagenta, ROOT.kViolet, ROOT.kBlack]
    
    i = 0
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            match = re.search(r'(\d+(\.\d+)?)VInj', file)
            if match:
                vinj = int(float(match.group(1)) * 1000)
                
                csv_file = os.path.join(directory, file)
                hist = make_TH1F(csv_file, 0, 0)  # row=0, col=0
                if hist:
                    histograms[vinj] = hist
                    i += 1
    
    return histograms

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python InjectionSpectra_comp.py <directory1> <directory2>")
        sys.exit(1)

    directory1 = sys.argv[1]
    directory2 = sys.argv[2]

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)

    # 캔버스 생성
    spectra_canvas = ROOT.TCanvas("spectra_canvas", "Canvas for multiple histograms", 800, 600)
    ROOT.gPad.SetMargin(0.1, 0.03, 0.1, 0.03)
    
    frame = ROOT.gPad.DrawFrame(0, 0, TOT_MAX, Y_MAX)
    frame.GetXaxis().SetTitle("ToT [us]")

    # 두 디렉토리의 히스토그램 처리
    histograms1 = process_directory(directory1)  
    histograms2 = process_directory(directory2)  

    # 범례 생성
    spectra_legend1 = ROOT.TLegend(0.58, 0.2, 0.78, 0.97)
    spectra_legend2 = ROOT.TLegend(0.78, 0.2, 0.97, 0.97)
    

    # 데이터 저장을 위한 배열
    all_vinj = sorted(set(list(histograms1.keys()) + list(histograms2.keys())))
    n = len(all_vinj)
    x1 = np.zeros(len(histograms1))
    y1 = np.zeros(len(histograms1))
    y1_err = np.zeros(len(histograms1))
    x2 = np.zeros(len(histograms2))
    y2 = np.zeros(len(histograms2))
    y2_err = np.zeros(len(histograms2))

    colors = [ROOT.kRed, ROOT.kOrange, ROOT.kYellow+1, ROOT.kGreen+1,ROOT.kBlue, 
              ROOT.kMagenta, ROOT.kViolet, ROOT.kBlack]
    # 첫 번째 디렉토리 히스토그램 처리
    i = 0
    for vinj in sorted(histograms1.keys()):
        hist = histograms1[vinj]
        hist.SetLineColor(colors[i % len(colors)])
        hist.SetLineStyle(1)
        hist.Draw("SAME")
        
        mean = hist.GetMean()
        sigma = hist.GetRMS()
        fit = ROOT.TF1("gaus1_"+str(i), "gaus", mean - 5*sigma, mean + 5*sigma)
        fit.SetLineColor(hist.GetLineColor())
        fit.SetLineWidth(0)
        hist.Fit(fit, "RQ")

        fit_mean = fit.GetParameter(1)
        fit_sigma = fit.GetParameter(2)
        x1[i] = vinj
        y1[i] = fit_mean
        y1_err[i] = fit_sigma

        #spectra_legend1.AddEntry(hist, f"{vinj} mV (1): #mu={fit_mean:.1f}, #sigma={fit_sigma:.1f}", "l")
        spectra_legend1.AddEntry(hist, f"{vinj} mV: #mu={fit_mean:.1f}", "l")
        i += 1

    # 두 번째 디렉토리 히스토그램 처리
    i = 0
    for vinj in sorted(histograms2.keys()):
        hist = histograms2[vinj]
        hist.SetLineColor(colors[i % len(colors)])
        hist.SetLineStyle(2)
        hist.Draw("SAME")
        
        mean = hist.GetMean()
        sigma = hist.GetRMS()
        fit = ROOT.TF1("gaus2_"+str(i), "gaus", mean - 5*sigma, mean + 5*sigma)
        fit.SetLineColor(hist.GetLineColor())
        fit.SetLineWidth(0)
        hist.Fit(fit, "RQ")

        fit_mean = fit.GetParameter(1)
        fit_sigma = fit.GetParameter(2)
        x2[i] = vinj
        y2[i] = fit_mean
        y2_err[i] = fit_sigma

        #spectra_legend2.AddEntry(hist, f"{vinj} mV (2): #mu={fit_mean:.1f}, #sigma={fit_sigma:.1f}", "l")
        spectra_legend2.AddEntry(hist, f"{vinj} mV: #mu={fit_mean:.1f}", "l")
        i += 1

    spectra_legend1.SetHeader(os.path.basename(directory1))
    spectra_legend1.Draw("SAME")
    spectra_legend2.SetHeader(os.path.basename(directory2))
    spectra_legend2.Draw("SAME")
    ROOT.gPad.RedrawAxis()
    spectra_canvas.Update()
    spectra_canvas.SaveAs(f"./InjectionSpectra_comparison_HV{HV}_THR{THR}.pdf")

    # ToT vs Vinj 그래프
    graph_canvas = ROOT.TCanvas("graph_canvas", "Injection Voltage vs ToT mean", 800, 600)
    graph1 = ROOT.TGraphErrors(len(x1), x1, y1, 0, y1_err)
    graph2 = ROOT.TGraphErrors(len(x2), x2, y2, 0, y2_err)

    # 그래프 스타일 설정
    graph1.SetMarkerStyle(20)
    graph1.SetMarkerColor(ROOT.kRed)
    graph1.SetLineColor(ROOT.kRed)
    graph2.SetMarkerStyle(21)
    graph2.SetMarkerColor(ROOT.kBlue)
    graph2.SetLineColor(ROOT.kBlue)

    graph1.SetTitle("Injection Voltage vs ToT mean from gaussian fitting")
    graph1.GetXaxis().SetTitle("Vinj [mV]")
    graph1.GetYaxis().SetTitle("ToT mean [us]")

    # 피팅 함수 정의 및 피팅
    fit_func1 = ROOT.TF1("fitFunc1", "[0]*x + [1]*(1-exp(-x/[2])) + [3]", min(x1), max(x1))
    fit_func2 = ROOT.TF1("fitFunc2", "[0]*x + [1]*(1-exp(-x/[2])) + [3]", min(x2), max(x2))
    
    fit_func1.SetParameters(0.1, 1.0, 100.0, 0.0)
    fit_func2.SetParameters(0.1, 1.0, 100.0, 0.0)
    
    fit_func1.SetLineColor(ROOT.kRed)
    fit_func2.SetLineColor(ROOT.kBlue)

    graph1.Fit(fit_func1, "RQ")
    graph2.Fit(fit_func2, "RQ")

    graph_canvas.SetGrid()
    graph1.Draw("AP")
    graph2.Draw("P SAME")

    # 범례 추가
    graph_legend = ROOT.TLegend(0.15, 0.7, 0.45, 0.85)
    graph_legend.AddEntry(graph1, os.path.basename(directory1), "lp")
    graph_legend.AddEntry(graph2, os.path.basename(directory2), "lp")
    graph_legend.Draw()
    latex1 = ROOT.TLatex()
    latex1.SetTextColor(ROOT.kRed)
    formula1 = "y = {0:.3f}x + {1:.1f}(1-e^{{-x/{2:.1f}}}) + {3:.1f}".format(
        fit_func1.GetParameter(0),
        fit_func1.GetParameter(1), 
        fit_func1.GetParameter(2),
        fit_func1.GetParameter(3)
    )
    latex1.DrawLatexNDC(0.33, 0.37, formula1)

    latex2 = ROOT.TLatex()
    latex2.SetTextColor(ROOT.kBlue) 
    formula2 = "y = {0:.3f}x + {1:.1f}(1-e^{{-x/{2:.1f}}}) + {3:.1f}".format(
        fit_func2.GetParameter(0),
        fit_func2.GetParameter(1), 
        fit_func2.GetParameter(2),
        fit_func2.GetParameter(3)
    )
    latex2.DrawLatexNDC(0.33, 0.30, formula2)

    graph_canvas.SaveAs(f"./InjectionGraph_comparison_HV{HV}_THR{THR}.pdf") 