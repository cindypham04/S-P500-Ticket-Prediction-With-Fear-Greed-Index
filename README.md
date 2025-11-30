# S&P 500 Stock Movement Prediction Project

## Project Overview

This project implements a comprehensive machine learning pipeline for predicting S&P 500 stock movement direction using sequential validation methodology. The project follows time-series best practices to avoid data leakage and provides realistic performance estimates for stock prediction models.

### Key Features

- **Sequential Validation**: Time-ordered splits respecting temporal dependencies
- **Multiple Models**: Logistic Regression, Random Forest, and Gradient Boosted Trees (XGBoost)
- **Comprehensive Feature Engineering**: 
  - Technical indicators (40+ features)
  - Lag features (20+ features)
  - International market indices (35+ features)
  - **Fear & Greed Index** (7 features including one-hot encoded classifications)
- **Detailed Evaluation**: Comprehensive plots comparing all models across all metrics
- **Model Persistence**: All trained models saved for future use

## Quick Start

### Prerequisites

- Python 3.8+
- Required libraries (see Installation section)
- Internet connection (for downloading stock data and Fear & Greed Index)

### Installation

1. **Clone or download the project**

2. **Install required libraries**:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib yfinance tqdm requests
```

3. **Download S&P 500 stock data** (if not already available):
```bash
cd dataset_creation
python download_sp500_data.py
cd ..
```

### Complete Pipeline Workflow

#### Step 1: Create Dataset from Scratch

If you're starting from scratch, run these scripts in order:

```bash
cd dataset_creation

# 1. Download S&P 500 stock data
python download_sp500_data.py


**Final dataset**: `dataset_creation/sp500_daily_features_with_fear_greed.csv`

#### Step 2: Create Sequential Validation Splits

```bash
python sequential_validation.py
```

This creates time-ordered train/validation/test splits in `sequential_splits_v1/`

#### Step 3: Train Models

```bash
python train_models.py
```

This trains 3 models (Logistic Regression, Random Forest, XGBoost) on 5 folds and saves them to `saved_models/`


## Detailed Component Description

### 1. Dataset Creation (`dataset_creation/`)

#### `download_sp500_data.py`
- **Purpose**: Downloads S&P 500 stock data from Yahoo Finance
- **Method**: Fetches list of S&P 500 tickers from GitHub, downloads historical data
- **Output**: Individual CSV files in `sp500_data/` directory (502 stocks)
- **Date Range**: 2018-01-01 to present

#### `add_fear_greed.py`
- **Purpose**: Adds Fear & Greed Index features
- **Features**:
  - `Fear_Greed_Value`: Numeric value (0-100)
  - `Fear_Greed_Classification`: Text classification
  - One-hot encoded classifications (5 binary features):
    - `fg_class_Extreme_Fear`
    - `fg_class_Fear`
    - `fg_class_Neutral`
    - `fg_class_Greed`
    - `fg_class_Extreme_Greed`
      
- **Data Handling**:
  - Forward-fills weekend/holiday values to next trading day
  - Shifts by 1 day to avoid look-ahead bias
- **Input**: Dataset with international indices
- **Output**: `sp500_daily_features_with_fear_greed.csv` (119 total features)

### 2. Sequential Validation (`sequential_validation.py`)

- **Purpose**: Creates time-ordered data splits for realistic evaluation
- **Method**: Sequential validation with expanding training windows
- **Splits**: 5 folds + final test set (20% of data)
- **Output**: `sequential_splits_v1/` directory with train/validation/test sets

#### Split Details:
- **Fold 1**: Train (2018-2019), Validate (2019-2020)
- **Fold 2**: Train (2018-2020), Validate (2020-2021)
- **Fold 3**: Train (2018-2021), Validate (2021-2023)
- **Fold 4**: Train (2018-2023), Validate (2023-2024)
- **Fold 5**: Train (2018-2024), Validate (2024-2025)
- **Final Test**: Last 20% of data (2025+)

### 3. Model Training (`train_models.py`)

- **Purpose**: Trains and evaluates multiple models using sequential validation
- **Models**: 
  - **Logistic Regression**: Linear model with feature scaling and class balancing
  - **Random Forest**: Ensemble tree model with class balancing
  - **Gradient Boosted Trees (XGBoost)**: Advanced ensemble with class balancing
- **Features**: 119 engineered features per stock per day
- **Output**: Trained models saved in `saved_models/` with metrics

#### Model Configuration:
```python
# Logistic Regression
LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

# Random Forest
RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)

# XGBoost
XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=1, n_jobs=-1, eval_metric='logloss')
```

### 4. Model Testing (`test_models.py` - Optional)

- **Purpose**: Loads trained models and evaluates them on test set
- **Evaluation**: All models across all folds + final models
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC
- **Output**: `test_results.txt` with comprehensive results

## Dataset Information

### Data Sources
- **S&P 500 Stocks**: 502 companies with daily OHLCV data
- **Time Period**: 2018-2025 (training), 2025+ (testing)
- **International Indices**: 7 major global market indices
- **Fear & Greed Index**: Daily sentiment index from CNN Business

### Feature Categories

1. **Technical Indicators** (49 features):
   - Price ratios (High/Low, Close/Open, etc.)
   - Moving averages (5-day, 20-day, 50-day)
   - Volume indicators
   - Volatility measures
   - Momentum indicators (RSI, MACD)
   - Bollinger Bands
   - Autocorrelation features

2. **Lag Features** (21 features):
   - Returns from past 1, 2, 3, 4, 5, 10, 15, 20, 22 days
   - Lag return statistics (mean, std over 5 and 22 day windows)
   - Lag of technical indicators (price ratios, volume, volatility)

3. **International Market Features** (42 features):
   - Returns, moving averages, volatility for each index
   - Price-to-MA ratios for global markets
   - 7 indices × 6 features each

4. **Fear & Greed Index Features** (7 features):
   - `Fear_Greed_Value`: Numeric sentiment value (0-100)
   - `Fear_Greed_Classification`: Text classification
   - 5 one-hot encoded binary features for each classification level

**Total Features**: 119 features per stock per day

### Target Variable
- **Binary Classification**: 0 (Down movement), 1 (Up movement)
- **Definition**: Next day's return > 0
- **Class Distribution**: ~47.7% Down, ~52.3% Up (slightly imbalanced)

## Methodology

### Sequential Validation
- **Why**: Avoids data leakage in time series prediction
- **Method**: Expanding training windows with fixed validation periods
- **Advantage**: More realistic performance estimates than random CV
- **Implementation**: 5 folds with expanding training data

### Feature Engineering
- **Technical Analysis**: Standard financial indicators
- **Lag Features**: Captures momentum and mean reversion patterns
- **Global Context**: International market influence
- **Sentiment Analysis**: Fear & Greed Index for market psychology

### Fear & Greed Index Integration
- **Weekend Handling**: Forward-fills weekend values to next trading day
- **Look-ahead Prevention**: Shifts by 1 day (uses previous day's sentiment)
- **Encoding**: One-hot encoding for classification levels (no ordinal assumption)

### Model Selection
- **Linear Model**: Logistic Regression (baseline, interpretable)
- **Tree-based**: Random Forest (ensemble, handles non-linearity)
- **Gradient Boosting**: XGBoost (advanced ensemble, best for complex patterns)

## Results Summary

### Model Performance Overview

The models are evaluated across 5 sequential validation folds. Performance metrics are averaged across folds:

| Model | Avg Accuracy | Avg Precision | Avg Recall | Avg F1-Score | Avg AUC |
|-------|--------------|---------------|------------|--------------|---------|
| Logistic Regression | ~0.50 | ~0.55 | ~0.28 | ~0.37 | ~0.53 |
| Random Forest | ~0.48 | ~0.51 | ~0.05 | ~0.09 | ~0.50 |
| Gradient Boosted Trees | ~0.49 | ~0.52 | ~0.24 | ~0.33 | ~0.49 |

### Key Findings
- **Logistic Regression** performs best overall across most metrics
- **All models** show performance close to random chance (~50% accuracy)
- **Stock prediction** is inherently challenging due to market efficiency
- **Sequential validation** provides realistic performance estimates
- **Fear & Greed Index** adds market sentiment context to features

### Visualizations

All comparison plots are available in the `plots/` directory:
- Model performance across folds
- ROC and Precision-Recall curves
- Confusion matrices
- Comprehensive metric comparisons

## Usage Examples

### Complete Workflow from Scratch

```bash
# 1. Download and create dataset
cd dataset_creation
python download_sp500_data.py
python create_technical_features.py
python add_lag_features.py
python add_international_indices.py
python add_fear_greed.py
cd ..

# 2. Create validation splits
python sequential_validation.py

# 3. Train models
python train_models.py

# 4. Generate plots
python create_comprehensive_plots.py
```

### Using Existing Dataset

If you already have the final dataset:

```bash
# Create splits
python sequential_validation.py

# Train models
python train_models.py

# Generate visualizations
python create_comprehensive_plots.py
```

### Loading Saved Models

```python
import joblib
import pandas as pd
import numpy as np

# Load a specific model
model = joblib.load('./saved_models/fold_1/Logistic_Regression/model.joblib')
scaler = joblib.load('./saved_models/fold_1/Logistic_Regression/scaler.joblib')

# Load metrics
metrics = joblib.load('./saved_models/fold_1/Logistic_Regression/metrics.joblib')
print(f"AUC: {metrics['auc']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Make predictions
# (Prepare your data in the same format as training)
# X_scaled = scaler.transform(X)
# predictions = model.predict(X_scaled)
# probabilities = model.predict_proba(X_scaled)[:, 1]
```

## Dependencies

### Core Libraries
```python
pandas>=1.3.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
scikit-learn>=1.0.0    # Machine learning
xgboost>=1.5.0         # Gradient boosting
```

### Visualization
```python
matplotlib>=3.5.0      # Plotting
seaborn>=0.11.0        # Statistical visualization
```

### Data Processing
```python
joblib>=1.1.0          # Model persistence
yfinance>=0.1.70       # Financial data download
tqdm>=4.62.0           # Progress bars
requests>=2.25.0       # HTTP requests for data download
```

### Installation Command
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib yfinance tqdm requests
```

## Configuration

### Model Parameters
Models can be customized by modifying parameters in `train_models.py`:

```python
# Logistic Regression
LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

# Random Forest
RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)

# XGBoost
XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=1, n_jobs=-1, eval_metric='logloss')
```

### Data Splits
Sequential validation parameters can be modified in `sequential_validation.py`:
- `n_splits`: Number of validation folds (default: 5)
- `test_size`: Proportion for final test set (default: 0.2)

### Feature Engineering
- Technical indicators: Modify `create_technical_features.py`
- Lag features: Adjust lag periods in `add_lag_features.py`
- International indices: Add/remove indices in `add_international_indices.py`
- Fear & Greed: Configure in `add_fear_greed.py`

## File Descriptions

### Core Scripts
- `sequential_validation.py`: Creates time-ordered data splits
- `train_models.py`: Trains and saves models using sequential validation
- `create_comprehensive_plots.py`: Generates all comparison plots
- `test_models.py`: Loads and evaluates saved models (optional)

### Data Processing
- `dataset_creation/download_sp500_data.py`: Downloads S&P 500 stock data
- `dataset_creation/create_technical_features.py`: Technical indicators
- `dataset_creation/add_lag_features.py`: Lag features
- `dataset_creation/add_international_indices.py`: International indices
- `dataset_creation/add_fear_greed.py`: Fear & Greed Index features

### Output Files
- `saved_models/`: All trained models and metrics
- `sequential_splits_v1/`: Time-ordered data splits
- `plots/`: All comparison plots
- `model_summary.txt`: Performance summary
- `test_results.txt`: Comprehensive test results (if test_models.py is run)

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib yfinance tqdm requests
   ```

2. **Data Download Issues**:
   - Check internet connection
   - yfinance may have rate limits - wait and retry
   - Some tickers may be delisted - script handles this automatically

3. **Memory Issues**:
   - Reduce `n_estimators` in Random Forest/XGBoost
   - Process data in chunks
   - Close other applications

4. **Disk Space Issues**:
   - Dataset files are large (~1-2GB each)
   - Ensure sufficient disk space (recommended: 10GB+ free)
   - Clean up intermediate files if needed

5. **Model Loading Errors**:
   - Ensure models are trained first: `python train_models.py`
   - Check file paths in scripts
   - Some models may be corrupted if training was interrupted

6. **Fear & Greed Data Issues**:
   - Ensure `milestone_3/fear_greed_2018_2025.csv` exists
   - Check file path in `add_fear_greed.py`
   - Verify date format matches stock data

### Performance Tips
- Use `n_jobs=-1` for parallel processing (already configured)
- Monitor memory usage with large datasets
- Consider feature selection for faster training
- Training takes 30 minutes to several hours depending on hardware

## Project Workflow Summary

```
1. Download Data
   └─> download_sp500_data.py
       └─> sp500_data/ (502 CSV files)

2. Feature Engineering Pipeline
   ├─> create_technical_features.py
   │   └─> sp500_daily_technical_features.csv (49 features)
   ├─> add_lag_features.py
   │   └─> sp500_daily_features_with_lags.csv (70 features)
   ├─> add_international_indices.py
   │   └─> sp500_daily_features_with_indices.csv (112 features)
   └─> add_fear_greed.py
       └─> sp500_daily_features_with_fear_greed.csv (119 features) ✓ FINAL

3. Data Splitting
   └─> sequential_validation.py
       └─> sequential_splits_v1/ (train/validation/test sets)

4. Model Training
   └─> train_models.py
       └─> saved_models/ (3 models × 5 folds)

5. Visualization
   └─> create_comprehensive_plots.py
       └─> plots/ (10 comparison plots)
```

## References

- Sequential Validation methodology based on time series best practices
- Technical indicators from standard financial analysis
- Model evaluation following machine learning standards
- Fear & Greed Index: CNN Business Market Sentiment

## Notes

- **Stock prediction is inherently difficult** due to market efficiency. Results close to random chance (50% accuracy) are expected and normal for this type of problem.
- **Feature engineering** is crucial - the combination of technical, lag, international, and sentiment features provides comprehensive market context.
- **Sequential validation** provides realistic performance estimates compared to random cross-validation.
- **Model interpretability**: Logistic Regression provides the best balance of performance and interpretability.

---

**Last Updated**: November 2024  
**Dataset**: S&P 500 stocks (2018-2025)  
**Total Features**: 119  
**Models**: Logistic Regression, Random Forest, XGBoost  
**Validation Method**: Sequential (5 folds)
