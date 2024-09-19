# AstroDrawer
Drawing codes for AstroPix_v3

## generate_event_display_YH.py
It draws 6 figures including hitmap & tot histogram 

## NoiseScan
    
### noise_map.py 
    It draws hitmap & readoutmap from noisescan directory
    - -d: DIRECTORY
   -  -t: THERSHOLD
   - -n: Chip ID (Default = APSw08s03)
   - -td: matching condition ~ Timestamp Difference (Default = <2 )
   - -tot: matching condition ~ Tot Difference (Default = <10% )

### noise_map_brief.py
   It only draws readoutmap with digits inside


