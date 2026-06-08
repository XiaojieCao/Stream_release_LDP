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





## Datasets



The dataset/ directory contains all datasets used in the experiments.



* Syn datasets/: synthetic datasets generated for experimental evaluation.

* AirQualityUCI.csv: air quality dataset.

* Foursquare.csv: Foursquare dataset.

* TDrive.csv: original TDrive dataset.

* TDrive_new.csv: processed TDrive dataset used in the experiments.

* Unemployment.csv: unemployment dataset.

* Volume.csv: volume dataset.

* data_deal.py: script for dataset preprocessing.



## Methods



The Methods/ directory contains all methods implemented and evaluated in this project.



The included methods are:



* AGS.py

* CAPP.py

* Fle_BA.py

* LBA.py

* MPS.py

* Naive_Sample.py

* Naive_Uniform.py



The folder statistical result/ stores pre-computed statistical results for relatively large datasets. These files are prepared in advance to reduce the cost of repeated computation during experiments.



## Environment



The code is implemented in Python. Please install the required dependencies before running the experiments.





## Experimental Results



For large datasets, some true statistical results are pre-computed and stored in:



Methods/statistical result/



These files can be directly used by the corresponding methods to improve experimental efficiency.




