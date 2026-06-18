import torch
import numpy as np
import copy

from qlib.contrib.model.pytorch_transformer import TransformerModel
from qlib.data.dataset import DataHandlerLP
from torch.optim.lr_scheduler import CosineAnnealingLR

class AdvancedTransformerModel(TransformerModel):

    def __init__(self, T_max=150, eta_min=1e-6, **kwargs):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(**kwargs)

    def fit(self, dataset, evals_result=dict(), **kwargs):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]


        scheduler = CosineAnnealingLR(self.train_optimizer, T_max=self.T_max, eta_min=self.eta_min)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.train_epoch(x_train, y_train)

            # CosineAnnealingLR
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            self.logger.info("train %.6f, valid %.6f, current lr: %.7f" % (train_score, val_score, current_lr))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)

        if self.use_gpu:
            torch.cuda.empty_cache()

