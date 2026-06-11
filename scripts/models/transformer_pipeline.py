from csv import DictWriter
from datetime import datetime

import qlib
import yaml
import os

import pandas as pd
from qlib.utils import init_instance_by_config

from scripts.feature_engineering.configuration_builder import PipelineConfigurator
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow import R

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


    top_100_features = df_fi.sort_values(by="Importance(Gain)", ascending=False).head(100)['Feature'].tolist()
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
        model.fit(dataset)

        pred = model.predict(dataset)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(root_dir, "results/transformer")
        os.makedirs(save_path, exist_ok=True)
        pred.to_csv(os.path.join(save_path, f"{timestamp}.csv"))
        print(f"Transformer trained successfully. Predictions saved to {save_path}")



def dataset_fetch(dataset):
    df_debug = dataset.handler.fetch(data_key=DataHandlerLP.DK_I)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(root_dir, "feature_engineering_output/transformer"), exist_ok=True)
    out_path = os.path.join(root_dir, f"feature_engineering_output/transformer/{timestamp}.csv")
    df_debug.to_csv(out_path)
    print(f"Saved debug DataFrame to {out_path}")


if __name__ == '__main__':
    gbm_save_dir = os.path.join(root_dir, f"results/lightgbm/20260610-202157")
    train_transformer(gbm_save_dir)