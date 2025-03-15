from SingleInjection import make_TH1F
import argparse
import glob
import sys
import os
import re

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <directory> <row> <col>")
        sys.exit(1)

    directory = sys.argv[1]
    row = int(sys.argv[2])
    col = int(sys.argv[3])

    if not os.path.isdir(directory):
        print(f"{directory} is not a valid directory")
        sys.exit(1)

    output_dir = "output_histograms"
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(directory):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(directory, file_name)
        match = re.search(r'_(\d+\.\d+)VInj', file_name)
        if match:
            vinj_value = float(match.group(1)) * 1000  # mV로 변환
            output_file_name = f"{vinj_value:.0f}mV.pdf"
        else:
            output_file_name = f"{file_name}.pdf"

        output_file_path = os.path.join(output_dir, output_file_name)
        try:
            plot_histogram(file_path, row, col, output_file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
