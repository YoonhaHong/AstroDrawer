import numpy as np
import matplotlib.pyplot as plt
import ROOT
from ROOT import TCanvas, TH1F, gApplication
import os
import argparse

# Function to load .txt file based on column and row
def load_txt_file(directory, col, row):
    filename = f"ToT_distribution_col{col}_row{row}.txt"
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        return np.loadtxt(filepath)
    else:
        print(f"File {filename} not found.")
        return None

def main(directory):

    nc = len( args.col )
    nr = len( args.row )
    c1 = TCanvas( 'c1', 'Dynamic Filling Example',  200*nc, 200*nr )
    c1.Divide(nc, nr, 0, 0)

    hist_tot_all = TH1F("hist", "hist", 20, 0, 20)
    hist_tot_pixel = [None for _ in range(nc*nr)]

    index=0
    for row in args.row:
        for col in args.col:
            # Load data for the current column and row
            data = load_txt_file(directory, col, row)
            c1.cd(index+1)
            hist_tot_pixel[index] = TH1F(f"c{col}_r{row}", f"c{col}_r{row}" ,20, 0, 20)
            
            if data is not None: 
                hist_tot_pixel[index].FillN(data.size, data, np.ones(data.size))
                hist_tot_all.FillN(data.size, data, np.ones(data.size))

            hist_tot_pixel[index].Draw()

            index+=1


    figdir=args.outdir if args.outdir else "."
    file_name = os.path.basename(directory)
    c1.cd(0)
    c1.SaveAs(f"{figdir}/{file_name}.pdf")

    ROOT.TPython.Prompt()  # You can Ctrl+C in this mode


# Argument parser to receive the directory from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot 4x4 hist_tot_pixel from .txt files')
    parser.add_argument('-d', '--directory', type=str, required=True, 
                    help='Directory path containing .txt files')

    parser.add_argument('-o', '--outdir', default=None, required=False,
                    help='output directory for all png files')

    parser.add_argument('-c', '--col', nargs='+', default=[15, 16, 17, 18], type=int,
                    help =  'Loop over given columns. ')

    parser.add_argument('-r', '--row', nargs='+', default=[15, 16, 17, 18], type=int,
                    help =  'Loop over given rows. ')

    

    args = parser.parse_args()
    # Call the function to plot hist_tot_pixel
    main(args.directory)