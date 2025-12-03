# BST 593 Applied Project

---

## 📘 Introduction

This study presents an unsupervised machine learning approach for anomaly detection in wearable-device data, using inter-model agreement to identify irregular patterns.

This repository contains the full anomaly detection pipeline applied to the [Kaggle Health and Fitness dataset](https://www.kaggle.com/datasets/evan65549/health-and-fitness-dataset).  
The dataset includes **3,000 participants** and more than **650,000 rows** of time-series measurements.

---

## 📦 Required Libraries

### **Python**
- pandas — dataframe manipulation  
- numpy  
- pyarrow — required for reading/writing Parquet files  
- scikit-learn  
- torch (PyTorch)  
- torchvision  
- torchaudio  
- notebook / ipykernel  

### **R**
- tidyr  
- dplyr  
- ggplot2  
- arrow  
- glmmTMB  
- xtable  
- mltools  
- tidyverse  
- irr  
- purrr  

---

## 📂 Repository Structure

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
│       ├── autoencoder_running.parquet
│       ├── autoencoder_swimming.parquet
│       ├── DATA.md
│       ├── fitness.parquet
│       ├── fitness.rds
│       ├── health_fitness_dataset.csv
│       ├── kMeans_running.parquet
│       ├── kMeans_swimming.parquet
│       ├── oneClassSVM_running.parquet
│       └── oneClassSVM_swimming.parquet
├── figure
│   ├── activity-pie-chart-1.pdf
│   ├── avg-hr-boxplot-1.pdf
│   ├── diastolic-bp-boxplot-1.pdf
│   ├── health-condition-bar-1.pdf
│   ├── kmeans_anomaly_Running.png
│   ├── kmeans_anomaly_Swimming.png
│   ├── kmeans_parameter_Running.png
│   ├── kmeans_parameter_Swimming.png
│   ├── oneClassSVM_Running.png
│   ├── oneClassSVM_Swimming.png
│   ├── resting-hr-boxplot-1.pdf
│   └── systolic-bp-boxplot-1.pdf
└── README.md
```

---

## 📑 Folder Descriptions

### **01_Introduction**
Includes data preprocessing and summary statistics, such as missing-value exploration, descriptive visualizations, and hypothesis testing.

### **02_DataAnalysis**
Contains the main analysis workflow. Machine-learning–based anomaly detection is implemented using:
- k-means clustering  
- one-class SVM  
- autoencoders  

### **03_results**
Includes the linear mixed-effects logistic regression models used to analyze covariates in the activity-tracking data.

### **data/pipeline**
Stores processed outputs from each step of the workflow, including:
- **CSV**
- **RDS**
- **Parquet** files  

A detailed description of each dataset is available in  
➡️ [`DATA.md`](./data/pipeline/DATA.md)

### **figure**
Contains all visualizations, including:
- dataset summaries  
- anomaly detection results  
- model parameter optimization plots  

---

