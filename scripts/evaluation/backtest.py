import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

cur_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.dirname(cur_dir)
root_dir = os.path.dirname(script_dir)

def backtest(pred_path, origin_path, threshold):
    pred_df = pd.read_csv(os.path.join(pred_path, "prediction_results.csv"))
    pred_df['datetime'] = pd.to_datetime(pred_df['datetime'])
    pred_df.set_index('datetime', inplace=True)

    jp_df = pd.read_csv(os.path.join(origin_path, "JP_225.csv"))
    jp_df['date'] = pd.to_datetime(jp_df['date'])
    jp_df.set_index('date', inplace=True)
    jp_df.sort_index(inplace=True)

    jp_df['daily_return'] = jp_df['close'].pct_change()

    backtest_df = jp_df[['daily_return']].join(pred_df[['prediction']], how='inner')
    backtest_df.rename(columns={'prediction': 'pred_vol'}, inplace=True)

    # strategy construction
    backtest_df['weight_baseline'] = 1.0
    vol_threshold = backtest_df['pred_vol'].quantile(threshold)

    #
    backtest_df['weight_strategy_A'] = np.where(backtest_df['pred_vol'] > vol_threshold, 0.0, 1.0)

    #
    target_vol = backtest_df['pred_vol'].median()
    backtest_df['weight_strategy_B'] = target_vol / backtest_df['pred_vol']
    backtest_df['weight_strategy_B'] = backtest_df['weight_strategy_B'].clip(upper=1.5)

    backtest_df['return_baseline'] = backtest_df['weight_baseline'].shift(1) * backtest_df['daily_return']
    backtest_df['return_strategy_A'] = backtest_df['weight_strategy_A'].shift(1) * backtest_df['daily_return']
    backtest_df['return_strategy_B'] = backtest_df['weight_strategy_B'].shift(1) * backtest_df['daily_return']

    print("backtest evaluation")
    cum_baseline = calc_metrics(backtest_df['return_baseline'], "Baseline: Always Hold")
    cum_strategy_A = calc_metrics(backtest_df['return_strategy_A'], "Strategy A: Risk Hedging")
    cum_strategy_B = calc_metrics(backtest_df['return_strategy_B'], "Strategy B: Volatility Target Weighting")

    plt.figure(figsize=(12, 8))
    plt.plot(cum_baseline.index, cum_baseline, label = "Baseline: Buy and Hold", color='gray', alpha=0.7)
    plt.plot(cum_strategy_A.index, cum_strategy_A, label = "Strategy A: Risk Hedging", color='red')
    plt.plot(cum_strategy_B.index, cum_strategy_B, label = "Strategy B: Volatility Target Weighting", color='blue')

    plt.title("Out-of-Sample Backtest: Nikkei 225 Equity Curves", fontsize=14)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Net Value", fontsize=14)
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.legend()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(pred_path, "backtest_equity_curve.png"))
    return backtest_df


def calc_metrics(returns, strategy_name):
    returns = returns.dropna()
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)

    sharpe_ratio = ann_return / ann_vol if ann_vol != 0 else 0

    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()

    print(f"[{strategy_name}]")
    print(f"annualized return: {ann_return:.2%}")
    print(f"annualized volatility: {ann_vol:.2%}")
    print(f"sharpe ratio: {sharpe_ratio:.2%}")
    print(f"max drawdown: {max_drawdown:.4f}\n")

    return cum_returns

if __name__ == "__main__":
    # pred_path = os.path.join(root_dir, "results/transformer/best")
    pred_path = os.path.join(root_dir, "results/lightgbm/20260610-202157")
    origin_path = os.path.join(root_dir, "raw_data/split_data")
    backtest_df = backtest(pred_path, origin_path, 0.85)
    backtest_df.to_csv(os.path.join(pred_path, "backtest_results.csv"))
