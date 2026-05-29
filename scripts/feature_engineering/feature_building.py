import qlib
from qlib.utils import init_instance_by_config
import yaml
import warnings
import os

warnings.filterwarnings("ignore")

cur_dir = os.getcwd()
scripts_dir = os.path.dirname(cur_dir)
root_dir = os.path.dirname(scripts_dir)

def feature_engineering():
    qlib.init(provider_uri = os.path.join(root_dir, "data/nikkei_data"), region = "cn")

    print("Starting feature engineering process...")

    with open(os.path.join(root_dir, "feature_config.yaml"), 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Loaded feature configuration from YAML file.")
    handler_config = config["data_handler"]
    handler = init_instance_by_config(handler_config)

    final_df = handler.fetch()

    print("Finished feature engineering process.")
    print("Final feature DataFrame shape: {}".format(final_df.shape))

    # features = final_df['feature']
    # labels = final_df['label']
    #
    # print(f"Features shape: {features.shape}")
    # print("first 5 features: {}", list(features.columns)[:5])
    # print("last 5 features: {}", list(features.columns)[-5:])

    return final_df



if __name__ == "__main__":
    features_df = feature_engineering()
    os.makedirs(os.path.join(root_dir, "feature_engineering_output"), exist_ok=True)
    features_df.to_csv(os.path.join(root_dir, "feature_engineering_output", "feature_engineering_output.csv"))