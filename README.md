# BST 593 Applied Project

---

## Introduction

This study presents an unsupervised machine learning approach for anomaly detection in wearable-device data, using inter-model agreement to identify irregular patterns.

This repository contains the anomaly detection pipeline applied to the [Kaggle Health and Fitness dataset](https://www.kaggle.com/datasets/evan65549/health-and-fitness-dataset), which includes data from **3,000 participants** and more than **650,000 rows** of time-series observations.

---

- **01_Introduction**: Contains data preprocessing and summary statistics, including missing-data exploration, data visualization, and hypothesis testing.

- **02_DataAnalysis**: Corresponding to the main data analysis section; machine-learning–based anomaly detection is implemented using k-means clustering, one-class SVM, and autoencoders.

- **03_results**: Includes the linear mixed-effects logistic regression used to analyze covariates in the activity logs.

- **data/pipeline**: Stores processed outputs from each stage of the pipeline, including Parquet, RDS, and CSV files. Detailed descriptions are provided in [DATA.md](./data/DATA.md).

- **figure**: Contains visualizations summarizing the dataset and plots used for optimizing anomaly detection parameters.

---

```
├── 01_Introduction
│   ├── Fitness_data_pipeline.R
│   └── Introduction.rmd
├── 02_DataAnalysis
│   ├── AppliedProject.ipynb
│   └── src
│       ├── algorithms.py
│       ├── dataset.py
│       └── preprocessing.py
├── 03_results
│   ├── BST593-lme.Rmd
│   └── utils
│       └── analysis.R
├── data
│   └── pipeline
│       └── health_fitness_dataset.csv
├── figure
└── README.md

```
- **01_Introduction**: preprocessing data and summarizing data, including missing summary, data visualization and hypothesis tests. 

- **02_DataAnalysis**: corresponding data analysis section, machine learning is implemented with k-means based anomaly detection, one class svm and autoencoder. 

- **03_results**: linear mixed effect logistic regression is used to analysis covariants in activity logs. 

- **data/pipeline**: processed files in each stages, including parquet, rds and csv files. Detail description for data pipeline in [DATA.md](./data/pipeline/DATA.md). 

- **figure**:including visualization for overview of dataset and optimizing variables for anomaly detection. 