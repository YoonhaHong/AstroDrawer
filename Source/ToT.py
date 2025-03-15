import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import ROOT
import time

def tot_distribution_root(csv_file, row, col):
    df = pd.read_csv(csv_file)  
    filtered_data = df[(df["row"] == row) & (df["col"] == col)]["tot_us"]

    hist = ROOT.TH1F("tot_us_dist", f"ToT Distribution for row={row}, col={col}", 150, 0, 300)

    for value in filtered_data:
        hist.Fill(value)

    return hist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ToT distribution from CSV.")
    parser.add_argument("csv_file", type=str,  help="Path to the CSV file")
    parser.add_argument('-r', "--row", type=int, required=True, help="Row value to filter")
    parser.add_argument('-c', "--col", type=int, required=True, help="Column value to filter")
    parser.add_argument('-l', "--log", action='store_true', default=False, help="log scale")

    args = parser.parse_args()

    t_start = time.time()
    #plot_tot_us_distribution(args.csv_file, args.row, args.col)
    canvas = ROOT.TCanvas("c1", "ToT Distribution", 800, 600)
    hist = tot_distribution_root(args.csv_file, args.row, args.col)
    hist.GetXaxis().SetTitle("ToT (us)")
    hist.GetYaxis().SetTitle("Entries")
    hist.SetLineColor(ROOT.kBlue)
    hist.SetFillColorAlpha(ROOT.kBlue, 0.3)
    hist.Draw()

    figpath = f"./fig/{os.path.basename(args.csv_file)[:-4]}_r{args.row}c{args.col}_tot_hist.pdf" 
    if args.log:
        ROOT.gPad.SetLogy()
        figpath = figpath.replace("hist.pdf", "log.pdf")

    # 출력
    canvas.Update()
    canvas.SaveAs( figpath )
    t_end = time.time()
    print(f"Elapsed time: {t_end - t_start}")
    input("Press Enter to exit...")  # 창 유지