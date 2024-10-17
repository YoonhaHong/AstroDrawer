import ROOT
from ROOT import TCanvas, TGraph, TLatex, TH1F, TPad

# 샘플 데이터 생성
Vbb = [-150, -150, -150, -200, -250]
THR = [400, 300, 200, 200, 200]
hits = [74733, 114643, 143881, 176751, 205781]

# 그래프 생성
c = TCanvas("c", "c", 700, 600)
pad = TPad("pad", "pad", 0.05, 0.05, 1., 1.)
pad.SetFillColor(0)  # 패드 배경색을 투명하게 설정
pad.Draw()  # 패드 그리기
pad.cd()  # 패드에 그리기

n = len(hits)
hist = TH1F("hits_hist", "", n, 0, n)
hist.SetStats(0)
# TGraph 객체 생성
graph = TGraph(n)

for i in range(n):
    graph.SetPoint(i, i, hits[i])  # x좌표는 인덱스, y좌표는 hits

# X축 레이블 설정
for i in range(n):
    label = '#splitline{V_{BB} = ' + str(Vbb[i]) + ' V }{THR = ' + str(THR[i]) + ' mV}'
    hist.SetBinContent(i+1, hits[i]) 
    hist.GetXaxis().SetBinLabel(i+1, label)

# X축과 Y축 글자 크기 설정
hist.GetXaxis().SetLabelSize(0.050)  # X축 레이블 크기
hist.GetYaxis().SetTitle("Number of hits per 30 mins")
hist.GetYaxis().SetTitleSize(0.04)  # Y축 제목 크기
hist.GetYaxis().SetLabelSize(0.04)  # Y축 레이블 크기
hist.SetMarkerStyle(20)
hist.SetMarkerSize(1.5)
hist.Draw("P")

c.Update()
pad.RedrawAxis()
#c.SaveAs("nhit_graph.pdf")  # 캔버스 이미지를 파일로 저장
c.Draw()