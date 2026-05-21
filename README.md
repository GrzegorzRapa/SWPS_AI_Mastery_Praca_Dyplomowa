# HowTo?

1. Clone this repo
2. Clone https://github.com/gsoh/VED (see https://ieeexplore.ieee.org/document/9262035 for details)
3. In VED folder, unzip Part1 and Part2 and copy all csv files directly to this repo (where 00_Driving_Cycle_chopper.py is
4. Open **00_Driving_cycle_chopper.py** and set parameters (default values below):
    - LONG_TRIP = 1  # 1 to enable filtering by duration, 0 to process all
    - MIN_LENGTH_S = 2400  # Minimum cycle duration in seconds (e.g., 60s)
    - ELEV_SAMPLING_M = 20    # Sample elevation every 20 meters to avoid ban at open-elevation
    - API_BATCH_SIZE = 100    #hard limit of open-elevation WEBSITE.
    - PROCESSED_DIR = 'processed'

5. Run **00_Driving_cycle_chopper.py** (it will take some time....). The source filles will be moved to 'processed'
6. Open **01_Cycle_Generator.py**  and set parameters (default values below):

    - SOURCE_DIR = 'source_data'
    - OUTPUT_DIR = 'generated_cycles'
    - T_DRIVE_S = 28800 #desired driving time for generated cycles)
    - SAMPLE_TIME_MS = 100 #sample time for the created files
    - BLEND_WINDOW_S = 120 #height difference bleeding window
    - BLEND_SAMPLES = int((BLEND_WINDOW_S * 1000) / SAMPLE_TIME_MS)
    - N_CLUSTERS = 3
    - REQUIRED_MIN_HIGH_SPEED = 100.0 #minimum speed to recognise driving as a motorway
    - STRETCH_ACCEL_LIMIT = 2.5
    - TARGET_PROPORTIONS = {0: 0.1, 1: 0.89, 2: 0.01}


7. Set desired parameters and run for as many times as you like (1 run generates one driving cycle)


8. Open **02_Vehicle_SIM_2.py** and set parameters (self - explanatory, better do not touch HEATING and COOLING)
    * It will generate output files for all driving cycles created in point 6
    * 
9. Train and validate NNs with **98_SOC_estimator.py** and **99_RANGE_estimator.py**
10. Validation only is possible with **101_SOC_validator_0_2.py** and **102_Range_validator_0_3.py**
