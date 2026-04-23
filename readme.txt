# Zebrafish Physiological Data Clustering Analysis

A MATLAB-based KNN classifier for sleep-wake state classification in zebrafish using physiological time-series data (heart rate, EMG, calcium activity). The code performs time-window feature extraction, K-means clustering for pseudo-label generation, anchor-based label assignment, KNN classifier training, and 5-fold cross-validation.
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
2. Ensure the following files are in your MATLAB path:
   - `zebrafish-sleep-wake-clustering.m` - Training script
   - `sleep_wake_classifier.m` - Prediction script
   - `classifier_params.mat` - Pre-trained classifier parameters
3. No compilation or building is required – the code is interpreted MATLAB.### Typical install time
- Less than 1 minute on a normal desktop computer.

## 4. Instructions for Use
Running the prediction script code on your own data
Prepare your CSV file with time in the first column and physiological signals in subsequent columns.

Run the script and provide the file path when prompted.

The script will automatically:

Load and validate the data


Export all results
