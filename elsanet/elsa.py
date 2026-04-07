import random
import time
import uuid
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from .autoencoder import AutoEncoder


class ELSA(nn.Module):
    def __init__(self, model_hyperparameters):
        super().__init__()
        self.model_hyperparameters = model_hyperparameters

        for item, value in model_hyperparameters.items():
            setattr(self, item, value)

        if hasattr(self, "device"):
            self.device = self.device
        else:
            self.device = None

        self.__gen_uiid()
        self.fix_seed()
        self.generate_elsa()

        # have to be called after layer models
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def __gen_uiid(self):
        self.elsa_id = str(uuid.uuid1())

    def fix_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def generate_elsa(self):
        self.elsa = nn.ModuleList()
        self.latent_space_dim = self.ae_architecture[-1]
        for _ in range(self.ae_times):
            self.add_autoencoder()

    def add_autoencoder(self):
        self.elsa.append(
            AutoEncoder(self.ae_architecture, self.activation, self.device)
        )

    def average_layer(self, tensor):
        return sum(tensor) / tensor.shape[0]

    def forward_autoencoders(self, batch):
        self.elsas_encoded, self.elsas_decoded = list(), list()
        for autoencoder in self.elsa:
            enc, dec = autoencoder.forward(batch)
            self.elsas_encoded.append(enc)
            self.elsas_decoded.append(dec)

        self.elsas_decoded = torch.stack(self.elsas_decoded)
        self.elsas_encoded = torch.stack(self.elsas_encoded)
        return self.elsas_encoded, self.elsas_decoded

    def concatenate_encoded_space(self):
        # movedim changes the dimension from [3, batch_size, 1, input_len]
        # to [batch_size, 3, 1, input_len]
        dimension_organized = torch.movedim(self.elsas_encoded, 1, 0)
        # for over batch_size dim, so, for each input
        # reshapes the desired encoded dim to autoencdoers_qtds * latent_space_dim
        # To a ELSA with 3 autoencoders and latent space dim de 4 neuros, reshapes to 12
        self.concatenated_encoded = [
            enc.reshape(1, self.ae_times * self.latent_space_dim)
            for enc in dimension_organized
        ]

    def concatenate_encoded_image_space(self):
        # movedim changes the dimension from [3, batch_size, 1, input_len]
        # to [batch_size, 3, 1, input_len]
        dimension_organized = torch.movedim(self.elsas_encoded, 1, 0)
        # for over batch_size dim, so, for each input
        # reshapes the desired encoded dim to autoencdoers_qtds * latent_space_dim
        # To a ELSA with 3 autoencoders and latent space dim de 4 neuros, reshapes to 12
        self.concatenated_encoded = torch.stack(
            [
                enc.reshape(1, self.ae_times * self.latent_space_dim)
                for enc in dimension_organized
            ]
        )

    def image_forward(self, batch):
        enc, dec = self.forward_autoencoders(batch)
        decoded_average = self.average_layer(dec)
        self.concatenate_encoded_image_space()

        return self.concatenated_encoded, decoded_average

    def forward(self, batch):
        enc, dec = self.forward_autoencoders(batch)
        decoded_average = self.average_layer(dec)
        self.concatenate_encoded_space()

        return self.concatenated_encoded, decoded_average

    def fit(
        self, data_instance, validation: bool = True, label=None, progress_bar=False
    ) -> None:
        self.data_instance = data_instance
        self.data_train = self.data_instance.data_train
        self.validation = validation
        if self.validation:
            self.data_validation = data_instance.data_val

        if progress_bar:
            progress_bar = tqdm(
                range(self.epochs), desc="Training label " + str(label) + ": "
            )
        else:
            progress_bar = range(self.epochs)

        input_data_dimension = 1
        for dimension in self.data_train.dataset[0][0].shape:
            input_data_dimension *= dimension

        self.loss_train, self.loss_val = [], []
        start = time.time()
        for epoch in progress_bar:
            for batch_train in self.data_train:
                inputs, targets = batch_train
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # OTIMIZAÇÃO DE TESE: Permitir achatamento total (Full Flatten) para 12288 neurônios
                # Se a dimensão esperada for o triplo da resolução (RGB), tratamos como 1 canal gigante.
                if self.data_instance.data_type == "image":
                    actual_channels = inputs.shape[1] if len(inputs.shape) == 4 else 1
                    if input_data_dimension == (inputs.shape[-1] * inputs.shape[-2] * actual_channels):
                        channels = 1 # Force flatten
                    else:
                        channels = actual_channels

                    inputs = inputs.view(inputs.shape[0], channels, int(input_data_dimension // channels)).to(
                        dtype=torch.float
                    )
                    forward_output = self.image_forward(inputs)
                    batch_train = inputs, targets
                else:
                    inputs = inputs.view(inputs.shape[0], 1, inputs.shape[1])
                    forward_output = self.forward(inputs)
                    batch_train = inputs, targets

                loss = self.loss_calculation(batch_train, forward_output)
                self.weights_adjustment(loss)

            self.loss_train.append(loss.item())  # logs last-batch loss per epoch

            if self.validation:
                loss_validation = self.train_validation()
                self.loss_val.append(loss_validation.item())

        end = time.time()
        self.elapsed_time = end - start

    def loss_calculation(self, batch_train, forward_output):
        inputs, targets = batch_train
        encoded, decoded = forward_output
        autoencoder_loss = self.loss_function(decoded, inputs)
        return autoencoder_loss

    def weights_adjustment(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_validation(self):
        with torch.no_grad():
            input_data_dimension = 1
            for dimension in self.data_validation.dataset[0][0].shape:
                input_data_dimension *= dimension

            for batch_validation in self.data_validation:
                inputs_validation, targets_validation = batch_validation
                inputs_validation = inputs_validation.to(self.device)
                targets_validation = targets_validation.to(self.device)

                if self.data_instance.data_type == "image":
                    actual_channels = inputs_validation.shape[1] if len(inputs_validation.shape) == 4 else 1
                    if input_data_dimension == (inputs_validation.shape[-1] * inputs_validation.shape[-2] * actual_channels):
                        channels = 1
                    else:
                        channels = actual_channels

                    inputs_validation = inputs_validation.view(
                        inputs_validation.shape[0],
                        channels,
                        int(input_data_dimension // channels),
                    ).to(dtype=torch.float)

                    # BUGFIX: use image_forward (not forward) for consistent encoding path
                    forward_output = self.image_forward(inputs_validation)
                    batch_validation = inputs_validation, targets_validation
                else:
                    inputs_validation = inputs_validation.view(
                        inputs_validation.shape[0],
                        1,
                        inputs_validation.shape[1],
                    )
                    forward_output = self.forward(inputs_validation)
                    batch_validation = inputs_validation, targets_validation

                joined_loss_validation = self.loss_validation_calculation(
                    batch_validation, forward_output
                )
        return joined_loss_validation

    def loss_validation_calculation(self, batch_validation, forward_output):
        with torch.no_grad():
            inputs, targets = batch_validation
            encoded, decoded = forward_output
            autoencoder_loss = self.loss_function(decoded, inputs)
            return autoencoder_loss


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss
