import torch
import torch.nn as nn
import numpy as np
import random
import uuid
import time
from tqdm import tqdm


class MultiVariablePredictor(nn.Module):
    def __init__(self, model_hyperparameters):
        super().__init__()
        self.model_hyperparameters = model_hyperparameters
        for item, value in model_hyperparameters.items():
            setattr(self, item, value)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.predictors = nn.ModuleList()
        self.fix_seed(self.seed)
        self.generate_predictors()
        self.__gen_uiid()
        # have to be called after layer models
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def __gen_uiid(self):
        self.predictor_id = str(uuid.uuid1())

    def fix_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def generate_predictors(self):
        for _ in range(self.n_targets):
            self.add_predictor()

    def add_predictor(self):
        predictor_layers = nn.Sequential()

        predictor_layers.append(
            nn.Linear(
                self.latent_space_length * self.n_features,
                self.predictor_architecture[0],
            ).to(self.device)
        )

        for idx in range(len(self.predictor_architecture) - 1):
            predictor_layers.append(
                nn.Linear(
                    self.predictor_architecture[idx],
                    self.predictor_architecture[idx + 1],
                ).to(self.device)
            )
            if self.batch_norm:
                predictor_layers.append(
                    nn.BatchNorm1d(
                        self.predictor_architecture[idx + 1], affine=False
                    ).to(self.device)
                )

        self.predictors.append(predictor_layers)

    def forward(self, batch):
        output = torch.stack(
            [predictor(batch) for predictor in self.predictors.to(self.device)]
        )
        return output.movedim(0, 1)

    def train_model(self, data_instance, validation=True, test=True) -> None:
        self.data_instance = data_instance
        self.data_train = self.data_instance.data_train
        self.validation = validation
        if self.validation:
            self.data_validation = data_instance.data_validation
        self.test = test
        if self.test:
            self.data_test = data_instance.data_test

        progress_bar = tqdm(range(self.epochs), desc="Training: ")

        self.loss_train, self.loss_validation, self.loss_test = [], [], []
        start = time.time()
        for epoch in progress_bar:
            for batch_train in self.data_train:
                inputs, targets = batch_train

                inputs = inputs.view(
                    inputs.size(0), inputs.size(1) * inputs.size(2)
                ).to(self.device)
                targets = targets.to(self.device)

                forward_output = self.forward(inputs)

                loss = self.loss_calculation(targets, forward_output)
                self.weights_adjustment(loss)

            self.loss_train.append(loss.item())

            if self.validation:
                loss_val, last_val_pred, last_val_target = self.get_validation_error()
                self.loss_validation.append(loss_val.item())
                self.last_val_pred = last_val_pred
                self.last_val_target = last_val_target

            if self.test:
                loss_t, last_test_pred, last_test_target = self.get_test_error()
                self.loss_test.append(loss_t.item())
                self.last_test_pred = last_test_pred
                self.last_test_target = last_test_target

        end = time.time()
        self.elapsed_time = end - start

    def loss_calculation(self, targets, forward_output):
        predicted_loss = self.loss_function(forward_output, targets)
        return predicted_loss

    def weights_adjustment(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_validation_error(self):
        with torch.no_grad():
            for data_validation in self.data_validation:
                inputs_validation, targets_validation = data_validation
                inputs_validation = inputs_validation.view(
                    inputs_validation.size(0),
                    inputs_validation.size(1) * inputs_validation.size(2),
                ).to(self.device)
                targets_validation = targets_validation.to(self.device)

                forward_output = self.forward(inputs_validation)

                loss_validation = self.loss_function(forward_output, targets_validation)
        return loss_validation, forward_output, targets_validation

    def get_test_error(self):
        with torch.no_grad():
            for batch_test in self.data_test:
                inputs_test, targets_test = batch_test

                inputs_test = inputs_test.view(
                    inputs_test.size(0),
                    inputs_test.size(1) * inputs_test.size(2),
                ).to(self.device)
                targets_test = targets_test.to(self.device)

                forward_output = self.forward(inputs_test)

                loss_test = self.loss_function(forward_output, targets_test)
        return loss_test, forward_output, targets_test
