# AstroDrawer
Drawing codes for AstroPix_v3

## generate_event_display_YH.py
It draws 6 figures including hitmap & tot histogram 

## NoiseScan
    
### noise_map.py 
It draws hitmap & readoutmap from noisescan directory

   -  -t: THERSHOLD (REQUIRED!!)

   -  -n: Chip ID (Default = APSw08s03)
   -  -rt: nReadouts threshold for noisy pixel (Default>9)
   -  -ht: nHits threshold for noisy pixel (Default>9)
   -  -td: matching condition ~ Timestamp Difference (Default = <2 )
   -  -tot: matching condition ~ Tot Difference (Default = <10% )

### noise_map_brief.py
   It only draws readoutmap with digits inside
   Share same flags with noise_map.py

   - -draw: Drawing option(nReadouts/nHits/Mask)

