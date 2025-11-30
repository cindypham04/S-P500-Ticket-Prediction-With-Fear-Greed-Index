#!/usr/bin/env python3
"""
Script to download S&P 500 stock data and save as CSV files
"""

import pandas as pd
import yfinance as yf
import os
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_sp500_tickers():
    """Get list of S&P 500 tickers"""
    # Try GitHub first (most reliable)
    try:
        import requests
        from io import StringIO
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            tickers = df['Symbol'].tolist()
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            print(f"Found {len(tickers)} S&P 500 tickers from GitHub")
            return tickers
    except Exception as e:
        print(f"Error fetching from GitHub: {e}")
    
    # Try Wikipedia with user agent
    try:
        import requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            tables = pd.read_html(response.text)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            print(f"Found {len(tickers)} S&P 500 tickers from Wikipedia")
            return tickers
    except Exception as e:
        print(f"Error fetching from Wikipedia: {e}")
    
    # Fallback: Use a comprehensive static list (major S&P 500 stocks)
    print("Using comprehensive static S&P 500 ticker list...")
    print("Note: This is a subset. For full dataset, ensure internet access for automatic download.")
    
    # Comprehensive list of major S&P 500 tickers
    return [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V',
        'JNJ', 'WMT', 'JPM', 'MA', 'PG', 'UNH', 'HD', 'DIS', 'BAC', 'ADBE',
        'CRM', 'XOM', 'CVX', 'ABBV', 'COST', 'NFLX', 'AVGO', 'PEP', 'TMO', 'CSCO',
        'ACN', 'ABT', 'DHR', 'LIN', 'NKE', 'MRK', 'TXN', 'PM', 'BMY', 'RTX',
        'HON', 'QCOM', 'AMGN', 'UPS', 'LOW', 'INTU', 'SPGI', 'DE', 'AMT', 'C',
        'TJX', 'GE', 'BKNG', 'AXP', 'ADP', 'SYK', 'GILD', 'ISRG', 'VRTX', 'CB',
        'MDT', 'ZTS', 'REGN', 'CME', 'EQIX', 'CL', 'EL', 'ICE', 'SHW', 'WM',
        'KLAC', 'CDNS', 'FIS', 'APH', 'ITW', 'ETN', 'FTV', 'EMR', 'NOC', 'PH',
        'PSA', 'AON', 'CMI', 'APD', 'CTAS', 'ROST', 'PCAR', 'ANET', 'FAST', 'TDG',
        'MCHP', 'PAYX', 'IDXX', 'DXCM', 'ODFL', 'CTSH', 'VRSK', 'FTNT', 'CDW', 'KEYS',
        'MPWR', 'ON', 'ZBRA', 'SWAV', 'TER', 'MSCI', 'NDAQ', 'CPRT', 'GGG', 'WWD',
        'POOL', 'CHRW', 'EXPD', 'TTD', 'FDS', 'BR', 'ROL', 'WST', 'ALGN', 'CPT',
        'EXAS', 'INCY', 'MRNA', 'BIO', 'ILMN', 'ALNY', 'ARWR', 'BEAM', 'CRSP', 'EDIT',
        'FOLD', 'IONS', 'NTLA', 'RGNX', 'SGMO', 'VERV', 'BLUE', 'RARE', 'SRPT', 'TGTX'
    ]

def download_stock_data(ticker, start_date='2018-01-01', end_date=None):
    """Download historical data for a single stock"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            return None
        
        # Reset index to get Date as column
        hist = hist.reset_index()
        
        # Convert Date to datetime and remove timezone if present
        if 'Date' in hist.columns:
            hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
        
        # Select and rename columns to match expected format
        hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Remove any rows with missing data
        hist = hist.dropna()
        
        return hist
    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")
        return None

def download_all_sp500_data():
    """Download all S&P 500 stock data"""
    
    # Create output directory
    output_dir = "./sp500_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading S&P 500 stock data...")
    print(f"Output directory: {output_dir}")
    
    # Get ticker list
    tickers = get_sp500_tickers()
    
    print(f"\nDownloading data for {len(tickers)} stocks...")
    print(f"Date range: 2018-01-01 to {datetime.now().strftime('%Y-%m-%d')}")
    
    successful = 0
    failed = 0
    
    for ticker in tqdm(tickers, desc="Downloading stocks"):
        # Download data
        data = download_stock_data(ticker)
        
        if data is not None and not data.empty:
            # Save to CSV
            output_file = os.path.join(output_dir, f"{ticker}.csv")
            data.to_csv(output_file, index=False)
            successful += 1
        else:
            failed += 1
            print(f"  Failed to download {ticker}")
    
    print(f"\nDownload complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total files in {output_dir}: {len(os.listdir(output_dir))}")
    
    return successful, failed

if __name__ == "__main__":
    download_all_sp500_data()

