import datetime

import joblib
import numpy as np
import pandas as pd
import yaml
import qlib
import lightgbm as lgb
from matplotlib import pyplot as plt
from matplotlib.pyplot import xlabel
from numpy import ndarray
import matplotlib.dates as mdates
from qlib.utils import init_instance_by_config
import os
import shap
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

cur_dir = os.getcwd()
scripts_dir = os.path.dirname(cur_dir)
root_dir = os.path.dirname(scripts_dir)


def train():
    provider_uri = os.path.join(root_dir, "data/nikkei_data")
    qlib.init(provider_uri=provider_uri, region="cn")

    with open(os.path.join(root_dir, "feature_config.yaml"), 'r', encoding="utf-8") as f:
        feature_config = yaml.safe_load(f)

    with open(os.path.join(cur_dir, "lightgbm.yaml"), 'r', encoding="utf-8") as f:
        lightgbm_config = yaml.safe_load(f)

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": feature_config["data_handler"],
            "segments": lightgbm_config["dataset_segments"],
            "step_len": 1
        }
    }

    print("Instantiating dataset. Feature engineering. Data splitting.")
    dataset = init_instance_by_config(dataset_config)

    print("Initializing LightGBM")
    model = init_instance_by_config(lightgbm_config["model"])


    evals_record = {}
    print("Fitting LightGBM model.")
    model.fit(dataset, evals_result=evals_record)


    print("Saving LightGBM")
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join(root_dir, f"results/lightgbm/{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 1. training curve
    training_curve(evals_record, model, save_dir)

    # 2. eval_df construction
    pred_series = model.predict(dataset, segment = "test")
    pred_df = pred_series.to_frame(name="prediction")

    label_df = dataset.prepare("test", col_set='label')
    label_df.columns = ['Target']
    eval_df = pred_df.join(label_df, how="inner")

    eval_df.to_csv(os.path.join(save_dir, "prediction_results.csv"))

    """ evaluation"""
    X_test = dataset.prepare("test", col_set='feature')

    if isinstance(X_test.columns, pd.MultiIndex):
        X_test.columns = X_test.columns.droplevel(level=0)
    feature_names = X_test.columns.to_list()

    # 3. feature importance
    feature_importance(feature_names, model, save_dir)

    # 4. SHAP values
    shap_summary(model, save_dir, X_test)


    joblib.dump(model, os.path.join(save_dir, "lightgbm_model.pkl"))
    with open(os.path.join(save_dir, "config.yaml"), 'w', encoding="utf-8") as f:
        yaml.dump([dataset_config, lightgbm_config], f)

    # evaluate_and_visualize(model, dataset, save_dir)
    return save_dir

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # For macOS, use 'Arial Unicode MS' or 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus signs are displayed correctly
sns.set_theme(style='whitegrid')

def training_curve(evals_record, model, save_dir):
    metric_name = list(evals_record['train'].keys())[0]
    train_loss = evals_record['train'][metric_name]
    val_loss = evals_record['valid'][metric_name]

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='train', linewidth=2, color='tab:blue')
    plt.plot(val_loss, label='valid', linewidth=2, color='tab:orange')

    best_iteration = model.model.best_iteration
    if best_iteration > 0:
        plt.axvline(x=best_iteration, color='red', linestyle='--', label=f'Early Stop (Round: {best_iteration})')

    plt.title(f"LightGBM Learning Curve ({metric_name})", fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(f'Loss ({metric_name})', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    curve_path = os.path.join(save_dir, "learning_curve.png")
    plt.savefig(curve_path, dpi=300)
    print(f"Learning curve saved to {curve_path}")
    plt.show()


def feature_importance(feature_names, model, save_dir, top_n = 20):
    print(f"Feature importance, top n: {top_n}")

    lgb_booster = model.model
    importances = lgb_booster.feature_importance(importance_type="gain")

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance(Gain)": importances
    }).sort_values(by="Importance(Gain)", ascending=False)

    top_fi_df = fi_df.head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance(Gain)', y='Feature', data=top_fi_df, palette='viridis')
    plt.title(f"LightGBM Top {top_n} Feature Importances", fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()

    fi_path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()

    fi_csv_path = os.path.join(save_dir, "feature_importance.csv")
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"Feature importance saved to {fi_csv_path}")


def shap_summary(model, save_dir, X_test):
    print("\n--- calculating SHAP values")

    lgb_booster = model.model

    explainer = shap.TreeExplainer(lgb_booster)

    shap_values = explainer.shap_values(X_test.values)

    plt.figure(figsize=(12, 8))
    if isinstance(shap_values, list):
        target_shap_values = shap_values[0]
    else:
        target_shap_values = shap_values

    shap.summary_plot(
        target_shap_values,
        features=X_test,
        feature_names=X_test.columns.tolist(),
        max_display=20,
        show=False
    )

    pd.DataFrame(target_shap_values).to_csv(os.path.join(save_dir, "shap_values.csv"))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_summary.png"))
    plt.close()
    print(f"SHAP summary saved to {os.path.join(save_dir, 'shap_summary.png')}")



def IC(save_dir, rolling_window=60):
    eval_df = pd.read_csv(os.path.join(save_dir, "prediction_results.csv"))
    print(f"Time Series IC Evaluation")

    overall_ic = pearsonr(eval_df['prediction'], eval_df['Target'])[0]
    overall_rank_ic = spearmanr(eval_df['prediction'], eval_df['Target'])[0]

    print(f"Overall IC (Pearson): {overall_ic:.4f}")
    print(f"Overall Rank IC (Spearman): {overall_rank_ic:.4f}")

    pred_rank = eval_df['prediction'].rank()
    target_rank = eval_df['Target'].rank()
    eval_df['rolling_rank_ic'] = pred_rank.rolling(rolling_window).corr(target_rank)



    fig, ax = plt.subplots(figsize=(10, 5))
    dates = pd.to_datetime(eval_df['datetime'])
    ax.plot(dates, eval_df['rolling_rank_ic'], color='tab:purple', label='rolling rank ic', linewidth=1.5)

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.6)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=range(1, 13)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax.fill_between(dates, eval_df['rolling_rank_ic'], 0, where=(eval_df['rolling_rank_ic'] > 0), color='tab:red', alpha=0.3)
    ax.fill_between(dates, eval_df['rolling_rank_ic'], 0, where=(eval_df['rolling_rank_ic'] < 0), color='tab:blue', alpha=0.3)

    plt.title(f"{rolling_window}-Day Rolling Rank IC (Time Series)\nOverall IC: {overall_ic:.4f}, Overall Rank IC: {overall_rank_ic:.4f}", fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Rolling Rank IC', fontsize=12, alpha=0.6)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.grid(which='major', linestyle=':', linewidth=0.5, color='black')
    plt.grid(which='minor', linestyle=':', linewidth=0.3, color='grey')
    plt.tight_layout()

    ic_path = os.path.join(save_dir, "IC.png")
    plt.savefig(ic_path, dpi=300)
    print(f"Rolling rank IC saved to {ic_path}")
    plt.close()



if __name__ == "__main__":
    # save_dir = train()
    save_dir = os.path.join(root_dir, "results/lightgbm/20260610-202157")
    IC(save_dir)

