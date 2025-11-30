#!/usr/bin/env python3
"""
Script to add Fear & Greed Index to the existing dataset
Handles weekends by forward-filling and shifts by 1 day to avoid look-ahead bias
"""

import pandas as pd
import numpy as np

def add_fear_greed_features():
    """Add Fear & Greed Index features to the dataset"""
    
    # File paths
    stock_file = "./sp500_daily_features_with_indices.csv"
    fear_greed_file = "../../milestone_3/fear_greed_2018_2025.csv"
    output_file = "./sp500_daily_features_with_fear_greed.csv"
    
    print("Adding Fear & Greed Index to stock dataset...")
    print(f"Stock data file: {stock_file}")
    print(f"Fear & Greed file: {fear_greed_file}")
    print(f"Output file: {output_file}\n")
    
    # Load stock data
    print("Loading stock data...")
    sp_df = pd.read_csv(stock_file, parse_dates=['Date'])
    print(f"Stock data shape: {sp_df.shape}")
    print(f"Stock date range: {sp_df['Date'].min()} to {sp_df['Date'].max()}\n")
    
    # Load Fear & Greed data
    print("Loading Fear & Greed data...")
    fg_df = pd.read_csv(fear_greed_file, parse_dates=['date'])
    fg_df = fg_df.rename(columns={'date': 'Date'})
    print(f"Fear & Greed data shape: {fg_df.shape}")
    print(f"Fear & Greed date range: {fg_df['Date'].min()} to {fg_df['Date'].max()}\n")
    
    # Step 1: Merge on Date (left merge - keeps all trading days)
    print("Step 1: Merging Fear & Greed data with stock data...")
    df = sp_df.merge(fg_df[['Date', 'fear_greed_value', 'classification']], on='Date', how='left')
    print(f"After merge shape: {df.shape}")
    print(f"Fear & Greed values before forward-fill: {df['fear_greed_value'].notna().sum()} non-null out of {len(df)} rows\n")
    
    # Step 2: Forward-fill Fear & Greed values
    # This pushes weekend/holiday values forward to the next trading day
    print("Step 2: Forward-filling Fear & Greed values (to handle weekends/holidays)...")
    df['fg_filled'] = df['fear_greed_value'].ffill()
    print(f"Fear & Greed values after forward-fill: {df['fg_filled'].notna().sum()} non-null out of {len(df)} rows\n")
    
    # Step 3: Shift by 1 day to avoid look-ahead bias
    # Since Fear & Greed is based on end-of-day values, we can't use it for same-day predictions
    print("Step 3: Shifting Fear & Greed by 1 day to avoid look-ahead bias...")
    df['Fear_Greed_Value'] = df['fg_filled'].shift(1)
    df['Fear_Greed_Classification'] = df['classification'].ffill().shift(1)
    print(f"Fear & Greed values after shift: {df['Fear_Greed_Value'].notna().sum()} non-null out of {len(df)} rows\n")
    
    # Step 3.5: Create one-hot encoded features for Fear & Greed classification
    print("Step 3.5: Creating one-hot encoded features for Fear & Greed classification...")
    dummies = pd.get_dummies(df['Fear_Greed_Classification'], prefix='fg_class')
    # Normalize column names: replace spaces with underscores
    dummies.columns = dummies.columns.str.replace(' ', '_')
    df = pd.concat([df, dummies], axis=1)
    print(f"Created {len(dummies.columns)} one-hot encoded columns: {list(dummies.columns)}")
    print(f"Dataset shape after one-hot encoding: {df.shape}\n")
    
    # Step 4: Drop rows where Fear_Greed_Value is NaN (first trading day and early days)
    print("Step 4: Dropping rows with missing Fear & Greed values...")
    initial_rows = len(df)
    df = df.dropna(subset=['Fear_Greed_Value']).reset_index(drop=True)
    dropped_rows = initial_rows - len(df)
    print(f"Dropped {dropped_rows} rows (initial days without Fear & Greed data)")
    print(f"Final dataset shape: {df.shape}\n")
    
    # Clean up temporary columns
    df = df.drop(columns=['fear_greed_value', 'classification', 'fg_filled'], errors='ignore')
    
    # Count new features
    original_features = len([col for col in sp_df.columns if col not in ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Target']])
    new_features = len([col for col in df.columns if col not in ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Target']])
    fear_greed_features_added = new_features - original_features
    
    print(f"Added {fear_greed_features_added} Fear & Greed features")
    print(f"Total features now: {new_features}\n")
    
    # Save the enhanced dataset
    df.to_csv(output_file, index=False)
    print(f"Enhanced dataset with Fear & Greed Index saved to: {output_file}\n")
    
    # Show sample of new features
    print("Sample Fear & Greed features:")
    sample_cols = ['Ticker', 'Date', 'Return', 'Fear_Greed_Value', 'Fear_Greed_Classification']
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].head(10))
    print()
    
    # Show one-hot encoded columns
    fg_class_cols = [col for col in df.columns if col.startswith('fg_class_')]
    print("One-hot encoded Fear & Greed classification columns:")
    print(f"  {fg_class_cols}")
    print(f"\nSample one-hot encoded values:")
    sample_cols_with_dummies = ['Ticker', 'Date', 'Fear_Greed_Classification'] + fg_class_cols
    print(df[sample_cols_with_dummies].head(10))
    print()
    
    # Show summary statistics
    print("Enhanced Dataset Summary:")
    print(f"   • Total records: {len(df):,}")
    print(f"   • Number of stocks: {df['Ticker'].nunique()}")
    print(f"   • Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   • Average records per stock: {len(df) / df['Ticker'].nunique():.1f}")
    print(f"   • Total features: {new_features}")
    print(f"   • Fear & Greed features added: {fear_greed_features_added}")
    print(f"   • Target distribution: {df['Target'].value_counts().to_dict()}")
    print(f"   • Fear & Greed value range: {df['Fear_Greed_Value'].min():.1f} to {df['Fear_Greed_Value'].max():.1f}")
    print(f"   • Fear & Greed mean: {df['Fear_Greed_Value'].mean():.2f}")
    print(f"   • Fear & Greed std: {df['Fear_Greed_Value'].std():.2f}")
    print(f"   • One-hot encoded classification columns: {len(fg_class_cols)}")
    print(f"   • Classification distribution:")
    for col in sorted(fg_class_cols):
        count = df[col].sum()
        pct = (count / len(df)) * 100
        print(f"     - {col}: {count:,} ({pct:.2f}%)")
    
    return df

if __name__ == "__main__":
    try:
        enhanced_df = add_fear_greed_features()
        print("\n✅ Successfully added Fear & Greed Index features!")
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        print("Please make sure the input files exist.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

