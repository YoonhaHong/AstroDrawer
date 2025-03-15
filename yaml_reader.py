import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from utility import yaml_reader


def main(yaml_file):
    pixs = yaml_reader(yaml_file=yaml_file)

    fig, ax = plt.subplots(figsize=(8, 8)) 

    pn = ax.hist2d(x=pixs['col'], y=pixs['row'], 
        bins=35, range=[[0,35],[0,35]], 
        weights= pixs['disable'], 
        norm=Normalize(vmin=0,vmax=1),cmap='Greys')
    cbar = fig.colorbar(pn[3], ax=ax, fraction=0.046, pad=0.04)  # fraction, pad로 크기 조절
    #cbar.set_label(label='Hit Counts', weight='bold', size=14)

    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('Col', fontweight='bold', fontsize=14)
    ax.set_ylabel('Row', fontweight='bold', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid()

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a YAML file.')
    parser.add_argument('yaml_file', type=str, help='Path to the YAML file')
    args = parser.parse_args()

    main(args.yaml_file)