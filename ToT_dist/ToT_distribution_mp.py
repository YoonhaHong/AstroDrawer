import os
import re
import time
import argparse
import pandas as pd
import numpy as np
import glob
from utility import yaml_reader_astep
from Multiprocess_astep import parallel_process 

def main(args):

    pair = [] 
    f = args.inputfile
    file_name = os.path.basename(f)
    dir_name = os.path.dirname(f)
    file_name = file_name.rstrip('.csv')

    thr_match = re.search(r'THR(\d+)', file_name)

    # THR 뒤에 _로 구분된 문자열들을 추출 (csv 확장자는 제외)
    split_parts = re.split(r'THR\d+\.?\d*_', file_name)[-1].split('_')

    threshold = thr_match.group(1) if thr_match else args.threshold  
    name = split_parts[0]
    date = split_parts[1]

    pair = parallel_process(f, args.timestampdiff, args.totdiff) 
    print("... Matching is done!")
 
    ##### Find masked pixel and save it as pixs###########################################################
    findyaml = f"{dir_name}/{file_name}*.yml"
    yamlpath = glob.glob(findyaml)
    print(yamlpath[0])

    disablepix=yaml_reader_astep(yamlpath[0])
    navailpixs = disablepix[disablepix['disable'] == 0].shape[0]
    npixel = '%.2f' % ( (navailpixs/1225) * 100.)
    print(f"{navailpixs}, {npixel}% active")
    pixs=pd.DataFrame(disablepix, columns=['col','row','disable'])
     
    ##### Create hit pixel dataframes #######################################################
    # Hit pixel information for all events
    dffpair = pd.DataFrame(pair, columns=['col', 'row', 
                                          'timestamp_col', 'timestamp_row', 
                                          'tot_us_col', 'tot_us_row', 'avg_tot_us'])


    outdir = dir_name+"/ToT_distributions_"+file_name
    os.makedirs(outdir, exist_ok=True)

# 각 col, row의 tot_us 분포 저장
    for col, row in dffpair[['col', 'row']].drop_duplicates().values:
        subset = dffpair[(dffpair['col'] == col) & (dffpair['row'] == row)]
        
        #npyfile = os.path.join(outdir, f"ToT_distribution_col{col}_row{row}.npy")
        #np.save(npyfile, subset['avg_tot_us'].values)

        txtfile = os.path.join(outdir, f"ToT_distribution_col{col}_row{row}.txt")
        np.savetxt(txtfile, subset['avg_tot_us'].values, fmt='%f')
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Astropix Driver Code')

    parser.add_argument('-if', '--inputfile', required=True, default =None,
                    help = 'input file')

    parser.add_argument('-n', '--name', default='APSw08s03', required=False,
                    help='chip ID that can be used in name of output file (default=APSw08s03)')

    parser.add_argument('-o', '--outdir', default=None, required=False,
                    help='output directory for all png files')

    parser.add_argument('-t', '--threshold', type = float, action='store', required=False, default=None,
                    help = 'Threshold voltage for digital ToT (in mV). DEFAULT value in yml OR 100mV if voltagecard not in yml')

    parser.add_argument('-rt','--rdothr', type=int, required=False, default=9,
                    help = 'threshold for nReadouts')
   
    parser.add_argument('-ht','--hitthr', type=int, required=False, default=9,
                    help = 'threshold for nHits')

    parser.add_argument('-td','--timestampdiff', type=float, required=False, default=2,
                    help = 'difference in timestamp in pixel matching (default:col.ts-row.ts<2)')
   
    parser.add_argument('-tot','--totdiff', type=float, required=False, default=10,
                    help = 'error in ToT[us] in pixel matching (default:(col.tot-row.tot)/col.tot<10%)')
    
    parser.add_argument('-b', '--beaminfo', default='None', required=False,
                    help='beam information ex) proton120GeV')

    parser.add_argument('-ns', '--noisescandir', action='store', required=False, type=str, default ='../astropix-python/noisescan',
                    help = 'filepath noise scan summary file containing chip noise infomation.')

    parser.add_argument
    args = parser.parse_args()

    t_start = time.time()
    main(args)
    t_end = time.time()
    print(f"{t_end-t_start} Elapsed")
