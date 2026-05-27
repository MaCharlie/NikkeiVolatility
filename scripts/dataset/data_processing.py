import os

import pandas as pd

INPUT_DIR = "raw_data/fetched_data"
OUTPUT_DIR = "raw_data/split_data"
cur_dir = os.getcwd()
scripts_dir = os.path.dirname(cur_dir)
root_dir = os.path.dirname(scripts_dir)


market_configs = {
        'JP_225.csv':     {'symbol': 'JP_225',      'shift_days': 0, "reserve_whole_price": True},
        'HK_HSI.csv':      {'symbol': 'HK_HSI',       'shift_days': 0, "reserve_whole_price": True},
        'US_SP500.csv':     {'symbol': 'US_SP500',      'shift_days': 1, "reserve_whole_price": True},
        'EU_STOXX50.csv': {'symbol': 'EU_STOXX50',  'shift_days': 1, "reserve_whole_price": True},
        '10Y_JP.csv': {'symbol': '10Y_JP', 'shift_days': 0, "reserve_whole_price": False},
        '10Y_US.csv': {'symbol': '10Y_US', 'shift_days': 1, "reserve_whole_price": False},
        "BCO.csv": {"symbol": "BCO", "shift_days": 1, "reserve_whole_price": False},
        "DXY.csv": {"symbol": "DXY", "shift_days": 1, "reserve_whole_price": False},
        "GOLD.csv": {"symbol": "GOLD", "shift_days": 1, "reserve_whole_price": False},
        "USD_JPY.csv": {"symbol": "USD_JPY", "shift_days": 1, "reserve_whole_price": False},
        "VIX.csv": {"symbol": "VIX", "shift_days": 1, "reserve_whole_price": False},
    }


def align_global_markets(data_dir, configs):

    print("Starting data alignment and global markets...")

    """ 1. Nikkei 225 as pivot calendar """
    master_filepath = os.path.join(data_dir, "JP_225.csv")
    if not os.path.exists(master_filepath):
        raise ValueError("JP_225 data is required for alignment but not found.")
    df_master = pd.read_csv(master_filepath)
    df_master['date'] = pd.to_datetime(df_master['date'])
    master_calendar = df_master.set_index('date').sort_index().index

    # aligned_dfs = []


    os.makedirs(os.path.join(root_dir, OUTPUT_DIR), exist_ok=True)
    for filename, config in configs.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print("File {} not found, skipping...".format(filepath))
            continue

        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # for macro factors, only reserve their close price.
        if not config['reserve_whole_price']:
            df['open'] = df['close']
            df['high'] = df['close']
            df['low'] = df['close']
            df['volume'] = 0.0
            df['factor'] = 1.0

        # dates of real trades, used for decide whether market is closed
        traded_dates = set(df.index)

        """ forward interpolation by Nikkei calendar """
        df_aligned = df.reindex(master_calendar, method='ffill')

        # for someday in master_calendar but not in original data(of other markets), it means market is closed, so set volume to 0
        holidays = master_calendar.difference(traded_dates)
        df_aligned.loc[holidays, 'volume'] = 0.0

        """ time zone alignment (prevent look-ahead bias)"""
        if config['shift_days'] > 0:
            df_aligned = df_aligned.shift(config['shift_days'])
            # df_aligned = df_aligned.bfill()

        # df_aligned['symbol'] = config['symbol']
        if 'factor' not in df_aligned.columns:
            df_aligned['factor'] = 1.0

        df_aligned = df_aligned.reset_index()
        # df_aligned = df_aligned[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'factor']]
        df_aligned = df_aligned[['date', 'open', 'high', 'low', 'close', 'volume', 'factor']]

        # aligned_dfs.append(df_aligned)
        print(f"Aligned {config['symbol']} data, total records after alignment: {len(df_aligned)}")

        """ save ticker.csv in directory """
        df_aligned = df_aligned.sort_values('date').reset_index(drop=True)
        df_aligned = df_aligned[df_aligned['date'] > df_aligned['date'].min()]
        output_path = os.path.join(root_dir, OUTPUT_DIR, filename)
        df_aligned.to_csv(output_path, index=False)
        print(f"Aligned data for {config['symbol']} saved to {output_path}...")

    # final_panel = pd.concat(aligned_dfs, ignore_index=True)
    # final_panel = final_panel.sort_values(by=['date', 'symbol']).reset_index(drop=True)
    #
    # final_panel = final_panel[final_panel['date'] > final_panel['date'].min()]
    #
    #
    # output_path = os.path.join(root_dir, OUTPUT_DIR, "global_market_data.csv")
    # final_panel.to_csv(output_path, index=False)
    # print(f"\nData processing and alignment completed. Global market data saved to {output_path}")





TEST_INPUT_DIR = "test_data"


if __name__ == "__main__":
    align_global_markets(os.path.join(root_dir, INPUT_DIR), configs=market_configs)


