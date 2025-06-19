# MLOps Zoomcamp Homework - Week 4

## Q1. Notebook

What's the standard deviation of the predicted duration for this dataset?

* 1.24
* **6.24** ✅
* 12.28
* 18.28

## Q2. Preparing the output

What's the size of the output file?

* 36M
* 46M
* 56M
* **66M** ✅

## Q3. Creating the scoring script

Which command you need to execute for that?

`jupyter nbconvert --to script starter.ipynb`

## Q4. Virtual environment

What's the first hash for the Scikit-Learn dependency?

`sha256:057b991ac64b3e75c9c04b5f9395eaf19a6179244c089afdebaad98264bff37c`

## Q5. Parametrize the script

What's the mean predicted duration for April 2023?

* 7.29
* **14.29** ✅
* 21.29
* 28.29

## Q6. Docker container

What's the mean predicted duration for May 2023?

* **0.19** ✅
* 7.24
* 14.24
* 21.19
~/C/M/m/0/homework (main)> docker exec zealous_herschel python ./starter.py -y 2023 -m 5 Year: 2023 Month: 5 Reading data from https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-05.parquet Predictions: count 3.399555e+06 mean 1.917442e-01
