from csv import DictWriter
from datetime import datetime

import joblib
import qlib
import yaml
import os

import pandas as pd
from matplotlib import pyplot as plt
from qlib.utils import init_instance_by_config

from scripts.feature_engineering.configuration_builder import PipelineConfigurator
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow import R
import seaborn as sns
import torch

# MacOS setting to prevent C/C++ threads conflict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_OK'] = 'True'


cur_dir = os.getcwd()
scripts_dir = os.path.dirname(cur_dir)
root_dir = os.path.dirname(scripts_dir)

def feature_configurate(gbm_save_dir):
    """
    based on gbm feature importance, dynamically adjust the processor chain in feature engineering before Transformer training
    add the dimension reduction processor
    :param gbm_save_dir:
    :return:
    """
    print("Feature reduction based on GBM feature importance...")

    if not os.path.exists(gbm_save_dir):
        raise FileNotFoundError(f"{gbm_save_dir} does not exist")
    df_fi = pd.read_csv(f"{gbm_save_dir}/feature_importance.csv")


    top_100_features = df_fi.sort_values(by="Importance(Gain)", ascending=False).head(15)['Feature'].tolist()
    manual_macro_features = [col for col in df_fi['Feature'] if col.startswith("Macro_")]

    selected_features = set(top_100_features + manual_macro_features)

    selector_config = {
        "class": "FeatureSelectionProcessor",
        "module_path": "scripts.feature_engineering.custom_processors",
        "kwargs": {
            "selected_features": selected_features
        }
    }

    configurator = PipelineConfigurator(os.path.join(root_dir, "feature_config.yaml"))
    dynamic_handler_config = (
        configurator
        .insert_processor_after("MacroBroadcastProcessor", selector_config)
        .build()
    )

    with open(os.path.join(cur_dir, "transformer.yaml"), 'r', encoding="utf-8") as f:
        transformer_config = yaml.safe_load(f)

    transformer_config["dataset"]["kwargs"]["handler"] = dynamic_handler_config
    transformer_config["model"]["kwargs"]["d_feat"] = len(selected_features)

    return transformer_config


def train_transformer(gbm_save_dir):
    qlib.init(provider_uri=os.path.join(root_dir, "data/nikkei_data"), region="cn")

    transformer_config = feature_configurate(gbm_save_dir)

    dataset = init_instance_by_config(transformer_config["dataset"])
    # dataset_fetch(dataset)
    # train = dataset.prepare('train')


    model = init_instance_by_config(transformer_config["model"])

    with R.start(experiment_name="Transformer Nikkei Pipeline"):
        print("Starting transformer training...")
        evals_record = {}
        model.fit(dataset, evals_result=evals_record)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(root_dir, "results/transformer", timestamp)
        os.makedirs(save_path, exist_ok=True)

        training_curve(evals_record, model, save_path)

        pred = model.predict(dataset)
        pred.to_csv(os.path.join(save_path, "predictions.csv"))

        with open(os.path.join(save_path, "transformer_config.yaml"), 'w', encoding="utf-8") as f:
            yaml.dump(transformer_config, f)
        joblib.dump(model, os.path.join(save_path, "model.pkl"))
        print(f"Transformer trained successfully. Predictions saved to {save_path}")

    return save_path



def dataset_fetch(dataset):
    df_debug = dataset.handler.fetch(data_key=DataHandlerLP.DK_I)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(root_dir, "feature_engineering_output/transformer"), exist_ok=True)
    out_path = os.path.join(root_dir, f"feature_engineering_output/transformer/{timestamp}.csv")
    df_debug.to_csv(out_path)
    print(f"Saved debug DataFrame to {out_path}")

def training_curve(evals_record, model, save_dir):
    metric = model.loss
    train_loss = evals_record['train']
    val_loss = evals_record['valid']
    train_loss = [-loss for loss in train_loss]
    val_loss = [-loss for loss in val_loss]

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='train', linewidth=2, color='tab:blue')
    plt.plot(val_loss, label='valid', linewidth=2, color='tab:orange')

    best_iteration = val_loss.index(min(val_loss))
    if best_iteration > 0:
        plt.axvline(x=best_iteration, color='red', linestyle='--', label=f'Early Stop (Round: {best_iteration})')

    plt.title(f"Transformer Learning Curve ({metric})", fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(f'Loss ({metric})', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    curve_path = os.path.join(save_dir, "learning_curve.png")
    plt.savefig(curve_path, dpi=300)
    print(f"Learning curve saved to {curve_path}")
    plt.show()



def extract_and_plot_attention(target_date, save_dir,
                               gbm_save_dir,instrument="JP_225", window_size=20):
    qlib.init(provider_uri=os.path.join(root_dir, "data/nikkei_data"), region="cn")

    transformer_config = feature_configurate(gbm_save_dir)

    dataset = init_instance_by_config(transformer_config["dataset"])
    model_path = os.path.join(save_dir, "model.pkl")

    """
    针对特定日期和标的，深入 PyTorch 底层提取 Transformer 注意力矩阵并绘制热图
    """
    print(f"Loading Qlib model wrapper from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    qlib_model = joblib.load(model_path)

    # 1. 剥离 Qlib 外壳，获取纯 PyTorch 模型
    if hasattr(qlib_model, 'model'):
        nn_model = qlib_model.model
    elif hasattr(qlib_model, 'nn_model'):
        nn_model = qlib_model.nn_model
    else:
        raise ValueError("Cannot find underlying PyTorch model. Check Qlib model object.")

    nn_model.eval()  # 开启推断模式，关闭 Dropout
    device = next(nn_model.parameters()).device  # 获取模型所在的设备 (CPU/MPS/CUDA)

    # 2. 定位到最后一层 Transformer 层
    try:
        last_layer = nn_model.transformer_encoder.layers[-1]
    except AttributeError:
        raise AttributeError("Failed to locate 'transformer_encoder'. Check your PyTorch architecture.")

    # 3. 准备挂载 Hook：不改变原有计算图，手动触发一次带权重的自注意力计算
    attention_weights_store = []

    def hook_fn(module, input_tensor, output_tensor):
        # input_tensor 是一个 tuple，第一个元素是进入该层的特征张量 (src)
        src = input_tensor[0]
        with torch.no_grad():
            # 绕开 PyTorch 底层的 need_weights=False，我们手动调用一次来窃取权重！
            _, attn_weights = module.self_attn(src, src, src, need_weights=True)
            attention_weights_store.append(attn_weights.cpu().numpy())

    # 注册拦截器
    handle = last_layer.register_forward_hook(hook_fn)

    # 4. 精准准备“单点数据” (Single Sample Data Preparation)
    print(f"Extracting historical 20-day data for {instrument} up to {target_date}...")
    df_features = dataset.prepare("test", col_set="feature")

    # 如果列包含 multi-index（比如全是 'feature' 前缀），需要拍平
    if isinstance(df_features.columns, pd.MultiIndex):
        df_features.columns = df_features.columns.droplevel(0)

    # 仅抽取目标标的的数据
    try:
        df_target = df_features.xs(instrument, level="instrument")
    except KeyError:
        raise KeyError(f"Instrument {instrument} not found in the test dataset.")

    # 找到目标日期的索引位置
    try:
        target_loc = df_target.index.get_loc(pd.to_datetime(target_date))
    except KeyError:
        raise KeyError(
            f"Target date {target_date} not found. Please ensure it is a valid trading day in your test set.")

    if target_loc < window_size - 1:
        raise ValueError(f"Not enough historical data before {target_date} to form a {window_size}-day window.")

    # 切片出 [T-19, T] 共 20 天的历史数据
    window_df = df_target.iloc[target_loc - window_size + 1: target_loc + 1]

    # 转换为 PyTorch 接受的 3D Tensor，形状 [Batch=1, TimeStep=20, Features]
    x_tensor = torch.tensor(window_df.values, dtype=torch.float32).unsqueeze(0).to(device)

    # 5. 执行一次“外科手术式”的前向推断
    print("Running forward pass to trigger hook...")
    with torch.no_grad():
        _ = nn_model(x_tensor)

    # 卸载拦截器（保持代码整洁）
    handle.remove()

    if not attention_weights_store:
        raise RuntimeError("Hook failed to capture attention weights.")

    # 6. 提取矩阵并绘图
    # 此时获取的矩阵形状通常为 [Batch=1, target_len=20, source_len=20]
    attn_matrix = attention_weights_store[0]
    if attn_matrix.ndim == 4:  # [batch, head, tgt, src]
        attn_matrix = attn_matrix[0, 0]  # 取出第一个 batch, 第一个 Head（或者你可以求均值）
    elif attn_matrix.ndim == 3:
        attn_matrix = attn_matrix[0]

    print(f"Successfully extracted Attention Matrix of shape {attn_matrix.shape}")

    plt.figure(figsize=(10, 8))

    # 构造 X 轴与 Y 轴的学术标签
    labels = [f"T-{window_size - 1 - i}" if i != window_size - 1 else "T (Target)" for i in range(window_size)]

    sns.heatmap(attn_matrix, cmap='YlGnBu', annot=False,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor='gray')

    plt.title(f"Transformer Self-Attention Weights\n(Predicting {instrument} Volatility on {target_date})", fontsize=14,
              fontweight='bold')
    plt.xlabel("Key (Historical Data Looked At)", fontsize=12)
    plt.ylabel("Query (Processing Step)", fontsize=12)

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"attention_heatmap_{target_date}.png")
    plt.savefig(out_path, dpi=300)
    print(f"🎉 Attention heatmap perfectly saved to: {out_path}")



if __name__ == '__main__':
    gbm_save_dir = os.path.join(root_dir, f"results/lightgbm/20260610-202157")
    # train_transformer(gbm_save_dir)
    save_dir = os.path.join(root_dir, f"results/transformer/20260612-173811_best")
    extract_and_plot_attention(save_dir=save_dir, gbm_save_dir=gbm_save_dir,
                               target_date="2024-08-05",
                               instrument="JP_225")