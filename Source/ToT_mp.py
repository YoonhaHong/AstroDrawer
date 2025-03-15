import argparse
import pandas as pd
import ROOT
import os
import multiprocessing as mp
import time

def process_chunk(args):
    chunk, row, col = args
    # 주어진 row, col 값 필터링
    filtered_data = chunk[(chunk["row"] == row) & (chunk["col"] == col)]["tot_us"]
    hist = ROOT.TH1F("tot_us_dist", f"ToT Distribution for row={row}, col={col}", 150, 0, 300)

    for value in filtered_data:
        hist.Fill(value)

    return hist

def tot_distribution_root(csv_file, row, col, chunksize=100000):
    # ROOT 히스토그램 생성
    hist_combined = ROOT.TH1F("hist_combined", f"ToT Distribution for row={row}, col={col}", 150, 0, 300)
    df_header = pd.read_csv(csv_file, nrows=1)
    #print( df_header.columns)

    cpu_count = mp.cpu_count()
    print(cpu_count)
    pool = mp.Pool(cpu_count)
    
    first_chunk = True
    results = []
    for chunk in pd.read_csv(csv_file, chunksize=chunksize):
        if first_chunk:
            results.append(pool.apply_async(process_chunk, args=((chunk, row, col),)))
            first_chunk = False
        else:
            # 헤더를 없애고 청크를 읽기
            chunk.columns = df_header.columns
            results.append(pool.apply_async(process_chunk, args=((chunk, row, col),)))

    pool.close()
    pool.join()

    # 결과를 합치기
    for result in results:
        hist_combined.Add(result.get())  # AsyncResult에서 실제 히스토그램을 가져옴

    return hist_combined

if __name__ == "__main__":
    # 명령줄 인수 처리
    parser = argparse.ArgumentParser(description="Plot ToT distribution from CSV using multiprocessing.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file")
    parser.add_argument('-r', "--row", type=int, help="Row value to filter", required=True)
    parser.add_argument('-c', "--col", type=int, help="Column value to filter", required=True)
    parser.add_argument('--chunksize', type=int, default=1000000, help="Chunk size for reading CSV file")
    
    args = parser.parse_args()

    t_start = time.time()
    
    # ROOT 캔버스 생성
    canvas = ROOT.TCanvas("c1", "ToT Distribution", 800, 600)
    
    # ToT 분포 히스토그램 생성
    hist = tot_distribution_root(args.csv_file, args.row, args.col, chunksize=args.chunksize)
    
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
    output_file = f"./fig/{os.path.basename(args.csv_file)[:-4]}_r{args.row}c{args.col}_tot_hist.pdf"
    canvas.SaveAs(output_file)
    t_end = time.time()
    print(f"Elapsed time: {t_end - t_start}")
    input("Press Enter to exit...")  # 창 유지