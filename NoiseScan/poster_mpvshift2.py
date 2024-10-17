import ROOT
import pandas as pd

NHITTHR = 100

def plot_histograms_with_pyROOT(file_list, legend_labels, colors):
    # 히스토그램을 저장할 리스트
    histograms = []

    # 각 파일에 대해 히스토그램 생성
    for i, file in enumerate(file_list):
        path = '/Users/yoonha/BIC_Beam_Test/AstroPix/AstroDrawer/ToT_dist/'
        f = f"{path}{file}.csv"
        df = pd.read_csv(f, delimiter=',')

        # nhits가 500을 넘는 행 필터링
        filtered_df = df[df['nhits'] > NHITTHR]

        if filtered_df.empty:
            print(f"{file}에 nhits > {NHITTHR}인 데이터가 없습니다.")
            continue

        # 히스토그램 생성
        hist = ROOT.TH1F(f"hist_{i}", legend_labels[i], 50, 0, 20)
        
        # MPV 값을 히스토그램에 추가
        for mpv in filtered_df['MPV']:
            hist.Fill(mpv)

        # 색상 지정
        hist.SetLineColorAlpha(colors[i], 0.96)
        hist.SetLineWidth(3)
        hist.SetStats(0)
        histograms.append(hist)

    # 히스토그램을 그리기 위한 캔버스 생성
    canvas = ROOT.TCanvas("canvas", "", 700, 600)

    pad = ROOT.TPad("pad", "pad", 0., 0., 1., 1.)
    pad.SetFillColor(0)  # 패드 배경색을 투명하게 설정
    pad.Draw()  # 패드 그리기
    pad.cd()  # 패드에 그리기

    frame = pad.DrawFrame(0, 0, 13, 100) 
    frame.SetFillColor(0)  # 프레임 배경색을 투명하게 설정
    frame.SetLineColor(ROOT.kBlack)  # 프레임 테두리 색상
    frame.SetLineWidth(2)  # 프레임 테두리 두께
    frame.GetXaxis().SetTitle("MPV of ToT [us]")
    frame.GetXaxis().SetTitleSize(0.04)
    frame.GetXaxis().SetLabelSize(0.04)
    frame.GetYaxis().SetTitleSize(0.04)
    frame.GetYaxis().SetLabelSize(0.04)
    frame.Draw()  # 프레임 그리기

    # 히스토그램을 그리기
    for hist in histograms:
        hist.Draw("HIST SAME")

    # 범례 추가
    legend = ROOT.TLegend(0.45, 0.5, 0.9, 0.9)  # 범례 위치
    legend.SetTextSize(0.034)
    legend.SetHeader(f"MPV from pixels with nHits > {NHITTHR}")
    for i, hist in enumerate(histograms):
        legend.AddEntry(hist, legend_labels[i], "l")
    legend.Draw()

    # 캔버스 출력
    canvas.Update()
    pad.RedrawAxis()
    canvas.SaveAs("mpv_histograms.pdf")  # 캔버스 이미지를 파일로 저장
    canvas.Draw()
    ROOT.TPython.Prompt()




# 파일 리스트 (예시로 5개 파일 이름)
file_list = [
    'ToT_distributions_THR400_APS3-W08-S03_VBB150_20241014_180407',
    'ToT_distributions_THR300_APS3-W08-S03_VBB150_20241014_164817',
    'ToT_distributions_THR200_APS3-W08-S03_VBB150_20241014_160746',
    'ToT_distributions_THR200_APS3-W08-S03_VBB200_20241014_145445',
    'ToT_distributions_THR200_APS3-W08-S03_VBB250_20241014_152552'
]

# 사용자가 원하는 legend에 들어갈 텍스트 리스트
legend_labels = [
    'V_{BB} = -150 V, THR = 400 mV',
    'V_{BB} = -150 V, THR = 300 mV',
    'V_{BB} = -150 V, THR = 200 mV',
    'V_{BB} = -200 V, THR = 200 mV',
    'V_{BB} = -250 V, THR = 200 mV',
]

colors = [ ROOT.kBlue, ROOT.kGreen+1, ROOT.kBlack, ROOT.kOrange, ROOT.kRed ]

plot_histograms_with_pyROOT(file_list, legend_labels, colors)
