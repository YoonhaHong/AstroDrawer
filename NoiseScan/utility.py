import pandas as pd
import glob
import os
import yaml

def yaml_reader(yaml_file):
    disablepix=[]
    file = open(yaml_file, 'r')
    config = yaml.safe_load(file)

    for col in range(0, 35, 1):
        value = config['astropix3']['config']['recconfig'][f'col{col}'][1]
        for row in range(0, 35, 1): 
                disable = (value & (2 << row)) >> (row+1)
                disablepix.append([col, row, disable])
    pixs=pd.DataFrame(disablepix, columns=['col','row','disable'])
    return pixs

def makesummary(threshold, output_path, timestamp_diff, tot_time_limit ):
    output = pd.DataFrame( columns=['row','col','nReadouts','nHits'] )
    datadir = "../../data" if os.path.exists("../../data") else "/Users/yoonha/cernbox/AstroPix"
    noise_scan_dir = f"{datadir}/NoiseScan/noise_THR{threshold}" 

    for r in range(0,35,1):
        for c in range(0,35,1):
            findfile = f"{noise_scan_dir}/noisescan_col{c}_row{r}_*.csv"
            filename = glob.glob(findfile)
            try: 
                df = pd.read_csv(filename[0],sep=',')
            #except (FileNotFoundError, pd.errors.EmptyDataError):
            except:
                #print(f"r{r} c{c}: FILE NOT FOUND")
                continue
            

            n_all_rows = df.shape[0] #num of rows in .csv file
            n_non_nan_rows = df['readout'].count() #valid number of row of 'readout' in .csv file
            n_nan_evts = n_all_rows - n_non_nan_rows
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
            df['readout'] = df['readout'].astype('Int64')

            if df.empty: continue
            max_readout_n = df['readout'].iloc[-1] #value from last row of 'readout'
            nreadouts = max_readout_n+1
            #print(f"n_all_rows={n_all_rows}")
            #print(f"n_non_nan_rows={n_non_nan_rows}")
            #print(f"max_readout_n={max_readout_n}")
    
            nhits = 0
            disable = 0

            for ievt in range(0, nreadouts, 1):
                dff = df.loc[(df['readout'] == ievt) & (df['payload'] == 4) & (df['Chip ID'] == 0)]
                if dff.empty:
                    continue
        # Match col and row to find hit pixel
                else:
                    dffcol = dff.loc[dff['isCol'] == 1]
                    dffrow = dff.loc[dff['isCol'] == 0]

                    for indc in dffcol.index:
                       for indr in dffrow.index:
                            if (abs(dffcol['timestamp'][indc] - dffrow['timestamp'][indr]) > timestamp_diff): continue
                            if (dffcol['tot_us'][indc] == 0): continue
                            if (abs(dffcol['tot_us'][indc] - dffrow['tot_us'][indr])/dffcol['tot_us'][indc]*100 > tot_time_limit): continue
                            if (dffcol['location'][indc] != c or dffrow['location'][indr] != r):
                                #print(f"[Matching but Continue] col.location, row.location = {dffcol['location'][indc]},{dffrow['location'][indr]}")
                                continue
                            else:
                                #average_tot = ((dffcol['tot_us'][indc] + dffrow['tot_us'][indr])/2)
                                #pair.append([ dffcol['location'][indc], dffrow['location'][indr], dffcol['timestamp'][indc], dffrow['timestamp'][indr], dffcol['tot_us'][indc], dffrow['tot_us'][indr], ((dffcol['tot_us'][indc] + dffrow['tot_us'][indr])/2)])
                                nhits += 1

            
            print(f"r{r} c{c}: {nhits} hits of {nreadouts} readouts")# Summary of how many events being used
            output.loc[len(output)] = [r, c, nreadouts, nhits]

    output = output.sort_values(by='nReadouts',ascending=False)
    output.to_csv(output_path, index=False)
    print(f"Made {output_path} file")


