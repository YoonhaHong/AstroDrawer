import ROOT

def find_second_peak(hist, tot_thr, min_peak_ratio=50):
    """히스토그램에서 두 번째 피크를 찾는 함수
    
    Args:
        hist: ROOT.TH1F 히스토그램
        tot_thr: ToT 임계값
        min_peak_ratio: 두 번째 피크의 최소 높이
    
    Returns:
        float: 피팅에 사용할 mode 값
    """
    mode = hist.GetBinCenter(hist.GetMaximumBin())
    if mode < tot_thr:
        max_bin = hist.GetMaximumBin()
        max_value = hist.GetBinContent(max_bin)
        second_max = 0
        second_max_bin = 0
        
        for bin in range(1, hist.GetNbinsX() + 1):
            if abs(bin - max_bin) > 5:
                value = hist.GetBinContent(bin)
                if value > second_max:
                    second_max = value
                    second_max_bin = bin
                    
        if second_max > min_peak_ratio:
            mode = hist.GetBinCenter(second_max_bin)
            
    return mode

def fit_histogram(hist, tot_thr, fit_window=5):
    """히스토그램에 가우시안 피팅을 수행하는 함수
    
    Args:
        hist: ROOT.TH1F 히스토그램
        tot_thr: ToT 임계값
        fit_window: 피팅 윈도우 크기
    
    Returns:
        tuple: (fit_mean, fit_sigma, fit_function)
    """
    mode = find_second_peak(hist, tot_thr)
    
    fit = ROOT.TF1("gaus", "gaus", mode - fit_window, mode + fit_window)
    fit.SetLineWidth(1)
    fit.SetLineColor(hist.GetLineColor())
    hist.Fit(fit, "RQ")
    
    fit_mean = fit.GetParameter(1)
    fit_sigma = fit.GetParameter(2)
    
    return fit_mean, fit_sigma, fit 