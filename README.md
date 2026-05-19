# Predictive Maintenance — Tire Intelligence

Exploration of machine learning techniques for predictive maintenance,
motivated by tire intelligence applications in fleet management.

## Notebooks

**01_rul_prediction.ipynb**
Remaining Useful Life (RUL) prediction on the NASA CMAPSS FD001 dataset.
- Sensor selection based on variance analysis (11 of 21 sensors retained)
- Rolling feature engineering — mean and std over 30-cycle window
- Gradient Boosting regression with safety-oriented evaluation
- Key finding: 62% dangerous prediction rate in critical zone (RUL < 30)
  motivates asymmetric loss functions in future work

**02_anomaly_detection.ipynb** *(in progress)*
Unsupervised anomaly detection using LSTM Autoencoder.
- Train on healthy early-life sensor data only
- Reconstruction error as degradation health index
- No failure labels required

## Dataset
NASA CMAPSS — https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
Place files in /data folder (not d by git).
