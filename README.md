# Quantum-Enhanced Crop Market Price Prediction Using Hybrid Machine Learning
## Preprocessed Onion Market Price Dataset (2014–2024)

### 1. Project Overview & Scope
This repository houses the end-to-end data preprocessing, feature engineering, and quality validation pipeline for predicting monthly wholesale onion market prices in India. 
The generated dataset is specifically curated for direct ingestion into both **Classical Machine Learning** models (Random Forest, XGBoost, LightGBM, LSTM, SVR) and **Quantum Machine Learning** architectures (Variational Quantum Regressors - VQR, Quantum Neural Networks - QNN, and Quantum Support Vector Regressors - QSVR).

### 2. Dataset Key Metrics
- **Original Daily Records**: 965,314 rows across 56 columns (2014-01-01 to 2024-12-31).
- **Cleaned Daily Records**: 964,570 rows (removed invalid zero target prices, severe scale typo errors > Rs 25,000/qtl, and genuine duplicate records).
- **Final Monthly Prediction Observations**: 54,002 records across 812 wholesale mandis, 223 districts, and 24 states.
- **Supervised Prediction Target**: `target_price_next_month` ($P_{t+1}$) in Rupees per Quintal (Rs/qtl).
- **Missing Values in Final Features**: Exactly 0 (100% clean and verified).

### 3. Chronological Partitions (Strictly Leakage-Free)
To ensure zero lookahead bias and rigorous time-series out-of-sample evaluation, data is split chronologically:
1. **`train.csv`** (2014–2021 | 8 Years): **38,017 observations** (Used for model training, scaler fitting, and feature selection).
2. **`validation.csv`** (2022–2023 | 2 Years): **10,876 observations** (Used for hyperparameter tuning and model checkpointing).
3. **`test.csv`** (2024 | 1 Year): **5,109 observations** (Out-of-time test benchmark for classical vs. quantum model comparison).

### 4. Methodological Highlights
1. **Target Formulation ($P_{t+1}$)**:
   For any monthly row indexed at month $t$, all input features are derived strictly from month $t$ and prior historical windows ($t-1, t-2, t-3, t-6, t-12$). The target to forecast is the expected modal price in month $t+1$. Discontinuous reporting months are automatically checked to ensure no forward leakage.
2. **Continuous Regression Imbalance Handling**:
   Agricultural price spikes (> Rs 5,000/qtl) represent genuine market crises (e.g. Dec 2019, late 2023). Rather than applying synthetic oversampling (e.g. SMOTE) which distorts continuous multivariate relationships, we provide **continuous sample weights** (`sample_weight`) calculated via inverse tier frequency on the training set:
   $$w_i = \frac{N_{train}}{K \cdot N_k}$$
   This ensures models penalize errors on crisis price spikes proportionately without fabricating artificial data.
3. **Quantum ML Compatibility**:
   - **Feature Scaling**: RobustScaler features (`*_scaled`) centered by median and scaled by IQR.
   - **Angle Encoding**: MinMaxScaler features (`*_angle`) mapped into $[0, \pi]$ for rotation gates ($R_y(\theta)$).
   - **Top-8 Quantum Features**: `price_current_month, price_min_current_month, tmin_current_month, price_max_current_month, price_change_ratio_1m, rainfall_total_current_month, price_median_current_month, sin_month`.
   - **Top-16 Quantum Features**: `price_current_month, price_min_current_month, tmin_current_month, price_max_current_month, price_change_ratio_1m, rainfall_total_current_month, price_median_current_month, sin_month, price_lag_12, rainfall_rolling_3m_sum, price_std_current_month, price_lag_6, price_lag_1, price_rolling_12m_mean, price_lag_3, temp_avg_lag_1`.

### 5. Repository File Structure
```
Capstone/
├── preprocessed_onion_2014_2024_prices_weather_events.zip   # Raw source archive
├── preprocessed_onion_2014_2024_prices_weather_events.csv   # Raw source CSV
├── preprocessing.py                                         # Reproducible preprocessing script
├── cleaned_onion_monthly.csv                                # Full monthly processed dataset
├── train.csv                                                # Training split (2014-2021)
├── validation.csv                                           # Validation split (2022-2023)
├── test.csv                                                 # Test split (2024)
├── preprocessing_report.csv                                 # 56-column audit & classification
├── feature_description.csv                                  # Feature dictionary & QML flags
├── README.md                                                # Documentation
└── preprocessing_visualizations/                            # 10 High-Resolution PNG figures
    ├── 01_target_distribution_before_after.png
    ├── 02_price_range_imbalance_comparison.png
    ├── 03_missing_values_before_after.png
    ├── 04_outlier_boxplot_comparison.png
    ├── 05_monthly_price_trends_over_time.png
    ├── 06_yearly_price_distribution.png
    ├── 07_feature_correlation_heatmap.png
    ├── 08_dataset_size_comparison.png
    ├── 09_feature_distributions_before_after.png
    └── 10_training_price_sample_weights.png
```

### 6. How to Reproduce
Run the standalone pipeline script directly:
```bash
python preprocessing.py
```
The script will execute autonomously, profile the data, generate all CSV files, render 10 publication-quality graphs at 300 DPI, and print the before-and-after comparison table.