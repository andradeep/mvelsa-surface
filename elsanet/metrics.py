from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import numpy as np
import torch


class Metrics:
    def __init__(self):
        self.metrics = {}
        self.metrics_means = {}
        self.internal_counter = 0

    def _get_rmse_metric(self):
        # RMSE (root mean squared error) entire signal and only test part
        rmse = np.sqrt(mean_squared_error(self.original, self.pred))
        self.update_metric(rmse)

    def _get_mse_metric(self):
        # MSE (mean squared error) entire signal and only test part
        mse = mean_squared_error(self.original, self.pred)
        self.update_metric(mse)

    def _get_mae_metric(self):
        # MAE (mean absolut error) entire signal and only test part
        mae = mean_absolute_error(self.original, self.pred)
        self.update_metric(mae)

    def _get_meae_metric(self):
        # MEAE (median absolut error) entire signal and only test part
        meae = median_absolute_error(self.original, self.pred)
        self.update_metric(meae)

    def _get_mape_metric(self):
        # MAPE (mean absolute percentage error) entire signal and only test part
        mape = mean_absolute_percentage_error(self.original, self.pred)
        self.update_metric(mape)

    def _get_precision_metric(self):
        precision = precision_score(self.original, self.pred, average="micro")
        self.metrics.update({"precision": precision})

    def _get_recall_metric(self):
        recall = recall_score(self.original, self.pred, average="micro")
        self.metrics.update({"recall": recall})

    def _get_f1_metric(self):
        f1 = f1_score(self.original, self.pred, average="micro")
        self.metrics.update({"f1": f1})

    def _get_accuracy_metric(self):
        accuracy = accuracy_score(self.original, self.pred) * 100
        self.metrics.update({"accuracy": accuracy})

    def update_metric(self, value):
        if self.internal_counter != 0:
            metric_list = self.metrics[self.function_name]
            metric_list.append(value)
            self.metrics.update({self.function_name: metric_list})
        else:
            self.metrics.update({self.function_name: [value]})

    def get_metrics(self, pred, original, functions: list = None):
        if torch.is_tensor(pred):
            self.pred = pred.to("cpu")
        else:
            self.pred = pred

        if torch.is_tensor(original):
            self.original = original.to("cpu")
        else:
            self.original = original

        if functions:
            for function in functions:
                self.function_name = function
                function_name = f"_get_{function}_metric"
                getattr(self, function_name)()
        else:
            if self.internal_counter == 0:
                ts = self.aviable_metrics()["time-series"]
                img = self.aviable_metrics()["image"]
                print(
                    f"Selecione uma das possibilidades: \
                        \n 0 - time series: {ts} \
                        \n 1 - iamges: {img}"
                )
                self.selected = int(input())
            functions = list(self.aviable_metrics().values())[self.selected]
            for function in functions:
                self.function_name = function
                function_name = f"_get_{function}_metric"
                getattr(self, function_name)()

        self.internal_counter += 1

    def aviable_metrics(self):
        return {
            "time-series": ["rmse", "mae", "meae", "mape"],
            "image": ["accuracy", "precision", "recall", "f1"],
        }

    def get_metrics_means(self):
        for metric, list_values in self.metrics.items():
            self.metrics_means[metric + "_mean"] = np.mean(list_values)
