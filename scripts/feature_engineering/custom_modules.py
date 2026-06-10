import numpy as np
import pandas as pd
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.processor import Processor
from qlib.data import D

class CustomFeatures(Alpha158):

    def get_feature_config(self):
        # 1. original Alpha158 factors dictionary
        feature_dict, feature_names = super().get_feature_config()

        # 2. Add custom features
        custom_features = {
            "Log_Return": "Log($close / Ref($close, 1))",
            "GK_Volatility": "0.5 * (Log($high / $low) ** 2) - 0.38629 * (Log($close / $open) ** 2)",
            "High_Low_Spread": "($high - $low) / $close"
        }

        for name, expr in custom_features.items():
            if isinstance(feature_dict, list):
                feature_dict.append(expr)
            else:
                feature_dict[name] = expr
            feature_names.append(name)

        return feature_dict, feature_names

    def get_label_config(self):
        return ["10000 * Ref(Mean(0.5 * (Log($high / $low) ** 2) - 0.38629 * (Log($close / $open) ** 2), 5), -5)"], ["Target_Volatility"]

