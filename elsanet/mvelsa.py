import random
import uuid
import torch.nn as nn
import numpy as np
import torch
import time

from elsanet.elsa import ELSA


class MVELSA(nn.Module):
    def __init__(self, model_hyperparameters):
        super().__init__()
        self.model_hyperparameters = model_hyperparameters

        self.seed = model_hyperparameters["seed"]

        self.__gen_uiid()
        self.fix_seed()

        self.mvelsa = torch.nn.ModuleList()
        self.labels = list()

    def __gen_uiid(self):
        self.mvelsa_id = str(uuid.uuid1())

    def fix_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def fit(self, data_instance, validation: bool = True):
        if data_instance.data_type == "image":
            self.train_multivariable_image(data_instance, validation)
        elif data_instance.data_type == "time-serie":
            self.train_multivariable_time_serie(data_instance, validation)

    def train_multivariable_time_serie(self, data_instance, validation):
        start = time.time()
        self.mvelsa_default_name_to_save = (
            data_instance.data_type
            + "_"
            + str(data_instance.dataset_name)
            + "_"
            + "series"
        )
        for idx, data_label in enumerate(data_instance.series_data):
            data_instance.data_train = data_label[0]
            data_instance.data_val = data_label[1]

            serie_data = data_instance.series_list[idx]

            model = ELSA(self.model_hyperparameters)
            model.fit(
                data_instance,
                validation=validation,
                label=serie_data,
                progress_bar=True,
            )

            self.mvelsa_default_name_to_save = (
                self.mvelsa_default_name_to_save + "-" + str(serie_data)
            )

            self.mvelsa.append(model)

        self.labels = data_instance.series_list
        end = time.time()
        self.elapsed_time = end - start

        for elsa in self.mvelsa:
            for autoencoder in elsa.elsa:
                autoencoder.set_autoencoders_weights_unadjustable()

    def train_multivariable_image(self, data_instance, validation):
        start = time.time()
        self.mvelsa_default_name_to_save = (
            data_instance.data_type
            + "_"
            + str(data_instance.dataset_name)
            + "_"
            + "labels"
        )

        for idx, data_label in enumerate(data_instance.labeled_data):
            data_instance.data_train = data_label[0]
            data_instance.data_val = data_label[1]
            data_instance.data_test = data_label[2]

            label_data = data_instance.labels_list[idx]

            model = ELSA(self.model_hyperparameters)
            model.fit(
                data_instance,
                validation=validation,
                label=label_data,
                progress_bar=True,
            )

            self.mvelsa_default_name_to_save = (
                self.mvelsa_default_name_to_save + "-" + str(label_data)
            )

            # MEMÓRIA: Move o especialista treinado para a CPU imediatamente.
            # Isso garante que apenas 1 especialista por vez ocupe a VRAM,
            # evitando CUDA Out of Memory em GPUs com menos de 8GB.
            model.to("cpu")
            import gc
            import torch as _torch
            _torch.cuda.empty_cache()
            gc.collect()

            self.mvelsa.append(model)

        self.labels = data_instance.labels_list
        end = time.time()
        self.elapsed_time = end - start

        for elsa in self.mvelsa:
            for autoencoder in elsa.elsa:
                autoencoder.set_autoencoders_weights_unadjustable()

    def save(self, file_name=None, file_path=None):
        for elsa in self.mvelsa:
            for autoencoder in elsa.elsa:
                autoencoder.set_autoencoders_weights_unadjustable()

        if file_name:
            name_to_save = file_name
        else:
            name_to_save = self.mvelsa_default_name_to_save

        if file_path:
            path_to_save = file_path + name_to_save
        else:
            path_to_save = name_to_save

        torch.save(self, path_to_save)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss
