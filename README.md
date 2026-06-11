# Stream_release_LDP

This repository contains the source code and datasets for stream data release under local differential privacy (LDP). The repository is organized for experimental reproduction.


## Overview


This project focuses on stream data publication under local differential privacy. It includes multiple real-world and synthetic datasets, several comparison methods, and pre-computed statistical results for large-scale datasets.


## Repository Structure

```text

Stream data release LDP/

├── dataset/

│   ├── Syn datasets/          # All synthetic datasets

│   ├── AirQualityUCI.csv      # Air quality dataset

│   ├── data_deal.py           # Data preprocessing script

│   ├── Foursquare.csv         # Foursquare dataset

│   ├── TDrive.csv             # Original TDrive dataset

│   ├── TDrive_new.csv         # Processed TDrive dataset

│   ├── Unemployment.csv       # Unemployment dataset

│   └── Volume.csv             # Volume dataset

│

├── Methods/

│   ├── statistical result/    # Pre-computed statistical results for large datasets

│   ├── AGS.py                 # AGS method

│   ├── CAPP.py                # CAPP method

│   ├── Fle_BA.py              # Fle-BA method

│   ├── LBA.py                 # LBA method

│   ├── MPS.py                 # MPS method

│   ├── Naive_Sample.py        # Naive sampling method

│   └── Naive_Uniform.py       # Uniform method

```



## Environment



The code is implemented in Python. Please install the required dependencies before running the experiments.





## Experimental Results



For large datasets, some true statistical results are pre-computed and stored in:



Methods/statistical result/



These files can be directly used by the corresponding methods to improve experimental efficiency.




