import datetime

import joblib
import numpy as np
import pandas as pd
import yaml
import qlib
import lightgbm as lgb
from matplotlib import pyplot as plt
from matplotlib.pyplot import xlabel
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
    save_dir = os.path.join(root_dir, f"results/{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    training_curve(evals_record, model, save_dir)

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





def evaluate_and_visualize(save_dir):

    # individual run without training
    provider_uri = os.path.join(root_dir, "data/nikkei_data")
    qlib.init(provider_uri=provider_uri, region="cn")

    model = joblib.load(os.path.join(save_dir, "lightgbm_model.pkl"))

    # 加载配置，重建 dataset
    with open(os.path.join(save_dir, "config.yaml"), 'r', encoding="utf-8") as f:
        dataset_config, lightgbm_config = yaml.safe_load(f)

    dataset = init_instance_by_config(dataset_config)


    """ 1. learning curve """
    print("1. learning curve")
    if hasattr(model, 'evals_result_') and model.evals_result_:
        try:
            lgb.plot_metric(model.evals_result_, metric='l2')
            plt.title('Learning Curve')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "learning_curve.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error plotting learning curve: {e}")
    else:
        print("No evaluation results found in the model. Skipping learning curve plot.")

    # plt.figure(figsize=(12, 8))
    # metric_name = list(evals_result['training'].keys())[0]
    # plt.plot(evals_result['training'][metric_name], label='Train Loss')
    # plt.plot(evals_result['validation'][metric_name], label='Val Loss')
    # plt.title('Learning Curve')
    # plt.xlabel('Epoch/Boosting Iteration')
    # plt.ylabel(f"residuals {metric_name}")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "learning_curve.png"), dpi=300)
    # plt.show()

    # prepare for predicting
    print("Evaluating model performance on test set.")
    df_test = dataset.prepare("test", col_set=["feature", "label"])
    X_test = df_test["feature"]
    y_test = df_test["label"].iloc[:,0]

    pred = model.predict(dataset, segment='test')
    eval_df = pd.DataFrame({"Target": y_test, "Prediction": pred})
    eval_df = eval_df.dropna()

    lgb_booster = model.model

    """ 2. Feature Importance - Gain"""
    print("2. Feature Importance - Gain")
    plt.figure(figsize=(12, 8))
    lgb.plot_importance(lgb_booster, importance_type='gain',
                        max_num_features=20,
                        title='Top 20 Feature Importance Based on Gain',
                        xlabel='Gain', ylabel = 'features',
                        color='steelblue', figsize=(10, 8))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=300)
    plt.show()

    """ 3. SHAP """
    print("3. SHAP values")
    X_test_sampled = X_test.sample(min(2000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(lgb_booster)
    shap_values = explainer.shap_values(X_test_sampled)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sampled, show=False)
    plt.title("SHAP Summary Plot", fontsize = 14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_summary.png"), dpi=300)
    plt.show()

    """ 4. IC and Rank IC"""
    print("4. calculating Information Coefficient")
    daily_ic = eval_df.groupby(level='datetime').apply(lambda x: clac_daily_ic(x, method="pearson"))
    daily_rank_ic = eval_df.groupby(level='datetime').apply(lambda x: clac_daily_ic(x, method="spearman"))
    mean_ic = daily_ic.mean()
    mean_rank_ic = daily_rank_ic.mean()
    ir = mean_rank_ic / daily_rank_ic.std()

    print(f"=================")
    print("test set report")
    print(f"=================")
    print(f"mean IC: {mean_ic: .4f}")
    print(f"mean rank_IC: {mean_rank_ic: .4f}")
    print(f"rank IR: {ir: .4f}")
    print(f"==================")

    plt.figure(figsize=(10, 5))
    daily_rank_ic.cumsum().plot(color='darkorange')
    plt.title("Cumulative Rank IC Over Time", fontsize=14)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Rank IC", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cumulative_rank_ic.png"), dpi=300)
    plt.show()

    # Here you can add code to calculate performance metrics like MSE, R^2, etc.
    print("Evaluation completed. (Add metric calculations here)")

def clac_daily_ic(df, method="pearson"):
    if len(df)<5:
        return np.nan
    if method=="pearson":
        return pearsonr(df['Predict'], df['Target'])[0]
    else:
        return spearmanr(df['Predict'], df['Target'])[0]

if __name__ == "__main__":
    save_dir = train()

    # save_dir = "/Users/light/Projects/landQuant/results/20260605-200857"
    # model = joblib.load(os.path.join(save_dir, "lightgbm_model.pkl"))
    #
    # # 调试：打印模型的所有属性
    # print("Model attributes:", dir(model))
    # print("LGBBooster attributes:", dir(model.model))
    #
    # # 检查是否有评估结果
    # if hasattr(model.model, 'evals_result_'):
    #     print("Found evals_result_:", model.model.evals_result_)
    # evaluate_and_visualize(save_dir)

