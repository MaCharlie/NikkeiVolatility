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
        return ["Ref(Mean(0.5 * (Log($high / $low) ** 2) - 0.38629 * (Log($close / $open) ** 2), 5), -5)"], ["Target_Volatility"]

class MacroBroadcastProcessor(Processor):
    def __init__(self, macro_tickers, fields=['$close'], make_stationary=True):
        super().__init__()
        self.macro_tickers = macro_tickers
        self.fields = fields
        self.make_stationary = make_stationary

    def fit(self, df: pd.DataFrame = None):
        pass

    def __call__(self, df: pd.DataFrame):
        dates = df.index.get_level_values("datetime").unique()
        start_time, end_time = dates.min(), dates.max()

        print(f"Fetching macro data for tickers. stationary processing: {self.make_stationary}...")

        macro_data = D.features(self.macro_tickers, self.fields, start_time=start_time, end_time=end_time)

        macro_df = macro_data.unstack(level='instrument')

        if ('$close', '10Y_US') in macro_df.columns and ('$close', '10Y_JP') in macro_df.columns:
            macro_df[('$close', 'US_JP_Spread')] = macro_df[('$close', '10Y_US')] - macro_df[('$close', '10Y_JP')]

            macro_df = macro_df.drop(columns=[('$close', '10Y_US'), ('$close', '10Y_JP')])
            print("Added US_JP_Spread feature by calculating the difference between 10Y US and 10Y JP yields. Original absolute value deleted to prevent total collinearity")

        if self.make_stationary:
            macro_df = macro_df.diff()
            print("Applied differencing to macro features to make them stationary.")

        # macro_df.columns = [f"Macro_{tic}" for field, tic in macro_df.columns]
        macro_df.columns = pd.MultiIndex.from_tuples(
            [('feature', f"Macro_{tic}") for filed, tic in macro_df.columns]
        )

        macro_df = macro_df.fillna(method="ffill").fillna(0)

        overlap_cols = df.columns.intersection(macro_df.columns)
        if not overlap_cols.empty:
            df = df.drop(columns=overlap_cols)

        merged_df = df.join(macro_df, on='datetime', how='left')
        merged_df = merged_df.fillna(method='ffill').fillna(0)

        return merged_df


# class CrossMarketSpilloverProcessor(Processor):
#     def __init__(self, target_instrument='JP_225'):
#         super().__init__()
#         self.target_instrument = target_instrument
#
#     def fit(self, df: pd.DataFrame = None):
#         pass
#
#     def __call__(self, df: pd.DataFrame):
#         print(f"Executing Cross-Market Spillover to target instrument: {self.target_instrument}...")
#         idx = pd.IndexSlice
#
#         # 1. fetching all features, instrument spread into columns
#         feat_df = df['feature'].unstack(level='instrument')
#
#         # 2. other markets
#         all_insts = df.index.get_level_values("instrument").unique()
#         other_insts = [inst for inst in all_insts if inst != self.target_instrument]
#
#         # nikkei's row mask and date
#         target_mask = df.index.get_level_values('instrument') == self.target_instrument
#         target_dates = df[target_mask].index.get_level_values("datetime")
#
#         # 3. rename features of other markets and spillover to Nikkei's row
#         for inst in other_insts:
#             if inst in feat_df.columns.get_level_values('instrument'):
#                 # other markets' features
#                 inst_features = feat_df.xs(inst, level='instrument', axis=1)
#
#                 for feat_name in inst_features.columns:
#                     new_col_name = f"{inst}_{feat_name}"
#                     # reindex to
#                     aligned_values = inst_features.reindex(target_dates)[feat_name].values
#                     df.loc[target_mask, ('feature', new_col_name)] = aligned_values
#
#         df.loc[~target_mask, 'label'] = np.nan
#
#         print(f"Spillover complete. Total features per row now: {df['feature'].shape[1]}")
#         return df
