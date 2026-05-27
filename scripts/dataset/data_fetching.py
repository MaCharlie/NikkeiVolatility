import os
import traceback

import numpy as np
import pandas as pd
import yfinance as yf

START_DATE = '2010-01-01'
END_DATE = '2025-12-31'
OUTPUT_DIR = "raw_data/fetched_data"
cur_dir = os.getcwd()
scripts_dir = os.path.dirname(cur_dir)
root_dir = os.path.dirname(scripts_dir)


TICKERS = {
    # core micro features
    "US_SP500": "^GSPC",
    "EU_STOXX50": "^STOXX50E",
    "JP_225": "^N225",
    "HK_HSI": "^HSI",

    # macro interest rate as liquidity agent
    "10Y_US": "^TNX",

    "USD_JPY": "JPY=X",
    "DXY": "DX-Y.NYB",

    "BCO": "BZ=F",
    "GOLD" : "GC=F",
    "VIX": "^VIX",
}

ADDITIONAL_TICKERS = {
    "10Y_JP": "10YJPY.B"
}

def fetch_raw_data():
    """
    fetch micro OHLCV(JP, US, EU, HK) and macro variables(except for JGB)
    :return:
    """
    data = {}
    os.makedirs(os.path.join(root_dir, OUTPUT_DIR), exist_ok=True)

    for name, ticker in TICKERS.items():
        file_path = os.path.join(root_dir, OUTPUT_DIR, f"{name}.csv")
        if os.path.exists(file_path):
            continue

        """1. Download original data (auto_adjust = False to reserve Adj Close)"""
        print(f"Fetching data for {name} ({ticker})...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
        if df.empty:
            print(f"No data for {name}:{ticker} found...")
            continue

        """2. Robustness check: rehabilitation factor"""
        if 'Adj Close' in df.columns:
            df['factor'] = df['Adj Close']/df['Close']
        else:
            df['factor'] = 1.0

        if 'Volume' not in df.columns: # foreign exchange without volume
            df['Volume'] = 0.0


        df = df[['Open', 'High', 'Low', 'Close', 'Volume','factor']]
        df.columns = ['open', 'high', 'low', 'close', 'volume', 'factor']

        # DatetimeIndex --> normal column 'date'
        df.index.name = 'date'
        df = df.reset_index()
        df['symbol'] = name

        final_df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'factor']]


        final_df.to_csv(file_path, index=False)
        print(f"Saved to {file_path}...")


def fetch_JGB():

    print("Fetching JGB stock data from MOF of Japan...")

    url = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"

    try:
        file_path = os.path.join(root_dir, OUTPUT_DIR, "10Y_JP.csv")
        if os.path.exists(file_path):
            return

        df = pd.read_csv(url, skiprows=1)
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)]

        df = df.replace(',', np.nan)
        df['10Y'] = pd.to_numeric(df['10Y'], errors='coerce')
        df = df.dropna(subset=['10Y'])

        df_jgb = pd.DataFrame()
        df_jgb['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df_jgb['close'] = df['10Y'].values
        df_jgb['open'] = df_jgb['close']
        df_jgb['high'] = df_jgb['close']
        df_jgb['low'] = df_jgb['close']
        df_jgb['volume'] = 0.0
        df_jgb['factor'] = 1.0

        df_jgb['symbol'] = '10Y_JP'
        df_jgb = df_jgb[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'factor']]

        os.makedirs(os.path.join(root_dir, OUTPUT_DIR), exist_ok=True)
        df_jgb.to_csv(file_path, index=False)
        print(f"Fetched JGB data, total records: {len(df_jgb)}, saved to {file_path}...")

    except Exception as ex:
        print(f"Error fetching JGB data: {ex}")
        traceback.print_exc()

if __name__ == "__main__":
    fetch_raw_data()
    fetch_JGB()