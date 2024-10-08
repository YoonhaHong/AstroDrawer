import os
import numpy as np
import array
import ROOT

def load_txt_file(directory, col, row):
    filename = f"ToT_distribution_col{col}_row{row}.txt"
    filepath = os.path.join(directory, filename)
    
    if os.path.exists(filepath):
        data = np.loadtxt(filepath)
        if data.ndim == 0:
            data = np.array([data])
        return data
    else:
        print(f"File {filename} not found.")
        return None

def langaufun(x, par):
    invsq2pi = 0.3989422804014  # (2 pi)^(-1/2)
    mpshift = -0.22278298       # Landau maximum location
    np_conv = 100.0             # number of convolution steps
    sc = 5.0                    # convolution extends to +-sc Gaussian sigmas

    mpc = par[1] - mpshift * par[0]
    xlow = x[0] - sc * par[3]
    xupp = x[0] + sc * par[3]
    step = (xupp - xlow) / np_conv

    sum_val = 0.0
    for i in range(int(np_conv / 2)):
        xx = xlow + (i + 0.5) * step
        fland = ROOT.TMath.Landau(xx, mpc, par[0]) / par[0]
        sum_val += fland * ROOT.TMath.Gaus(x[0], xx, par[3])

        xx = xupp - (i + 0.5) * step
        fland = ROOT.TMath.Landau(xx, mpc, par[0]) / par[0]
        sum_val += fland * ROOT.TMath.Gaus(x[0], xx, par[3])

    return par[2] * step * sum_val * invsq2pi / par[3]

def All_Fit_ToT(directory):
    cols_to_draw = [15, 16, 17, 18]
    rows_to_draw = [15, 16, 17, 18]

    nc = len(cols_to_draw)
    nr = len(rows_to_draw)

    # Create histograms
    hist_tot_all = ROOT.TH1F("hist_tot_all", "Total Histogram", 20, 0, 20)
    hist_tot_pixel = [[None for _ in range(35)] for _ in range(35)]
    fit_tot_pixel = [[None for _ in range(35)] for _ in range(35)]

    hist_MPV = ROOT.TH1F("hist_MPV", "MPV Distribution", 50, 0, 10)
    MPV = np.zeros((35, 35))

    pllo = [0.09, 0.5, 5, 0.2]
    plhi = [2.80, 20.0, 10000, 1.8]
    sv = array.array('d',[0.5, 4.0, 500, 0.4])

    for r in range(35):
        for c in range(35):
            # Load data for current column and row
            data = load_txt_file(directory, c, r)

            hist_tot_pixel[c][r] = ROOT.TH1F(f"c{c}_r{r}", f"c{c}_r{r}", 20, 0, 20)

            if data is None:
                continue
            for value in data:
                hist_tot_pixel[c][r].Fill(value)
                hist_tot_all.Fill(value)  # Fill total histogram

            fit_tot_pixel[c][r] = ROOT.TF1(f"langau_c{c}_r{r}", langaufun, 2, 10, 4)
            fit_tot_pixel[c][r].SetParNames("Width", "MP", "Area", "GSigma")
            fit_tot_pixel[c][r].SetParameters(sv)

            for par in range(4):
                fit_tot_pixel[c][r].SetParLimits(par, pllo[par], plhi[par])

            hist_tot_pixel[c][r].Fit(f"langau_c{c}_r{r}", "RQ")
            fp = fit_tot_pixel[c][r].GetParameters()

            MPV[c][r] = fp[1]

            if MPV[c][r] < 1e-3:
                print(f"{c} {r}  Small MPV!")

            hist_MPV.Fill(fp[1])

    # Create canvas and plot histograms
    cToT = ROOT.TCanvas("ToT", "ToT Dist.", 200 * nc, 200 * nr)
    cToT.Divide(nc, nr, 0, 0)
    ROOT.gStyle.SetOptStat(1111)
    ROOT.gStyle.SetOptFit(111)

    index = 0
    for row in rows_to_draw:
        for col in cols_to_draw:
            print(f"{MPV[col][row]}, ", end="")
            cToT.cd(index + 1)
            index += 1

            hist_tot_pixel[col][row].Draw()
            if MPV[col][row] < 0.1:
                continue
            fit_tot_pixel[col][row].Draw("L SAME")
        print("\n")

    # Plot MPV distribution
    c2 = ROOT.TCanvas("c2", "MPV Dist.", 800, 800)
    c2.cd()
    hist_MPV.Draw()

    fit_result = hist_MPV.Fit("gaus", "SQ")

    outdir = "."
    file_name = os.path.basename(directory)
    cToT.SaveAs(f"{outdir}/{file_name}_python.pdf")

if __name__ == "__main__":
    All_Fit_ToT("/Users/yoonha/cernbox/A-STEP/241005/ToT_distributions_THR200_APS3-W08-S03_20241005_175622") 