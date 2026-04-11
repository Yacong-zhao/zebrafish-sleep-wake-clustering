# Zebrafish Physiological Data Clustering Analysis

This repository contains the MATLAB code for clustering analysis of zebrafish physiological time-series data (heart rate, EMG, calcium activity). The code performs time-window feature extraction, K-means clustering (2 classes), PCA visualization, and exports cluster thresholds and original data with labels.

## 1. System Requirements

### Operating Systems
- Windows 10/11 (tested)
- macOS (should work, not formally tested)
- Linux (Ubuntu 20.04+, should work, not formally tested)

### Software Dependencies
- MATLAB R2020b or later

### Required MATLAB Toolboxes
- Statistics and Machine Learning Toolbox (for `prctile`, `skewness`, `kurtosis`, `iqr`)
- No other toolboxes required

### Non-standard Hardware
- None. The code runs on any standard desktop computer.

### Tested Versions
- MATLAB R2022b on Windows 11
- MATLAB R2020b on Windows 10

## 2. Installation Guide

### Instructions
1. Download all code files into a single folder.
2. Ensure the main script (e.g., `zebrafish_clustering.m`) is in your MATLAB path.
3. No compilation or building is required – the code is interpreted MATLAB.

### Typical install time
- Less than 1 minute on a normal desktop computer.

## 4. Instructions for Use
Running the code on your own data
Prepare your CSV file with time in the first column and physiological signals in subsequent columns.

Run the script and provide the file path when prompted.

Set the time window size (in seconds) when asked. Default is 5 seconds.

The script will automatically:

Load and validate the data

Extract window-wise maximum values

Perform outlier replacement (only for calcium data, using 3×IQR)

Standardize features (only for clustering; original values preserved in outputs)

Run K-means clustering (k=2) with fixed random seed (42)

Calculate cluster thresholds as the midpoint between cluster means

Export all results
