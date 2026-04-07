import torch.nn as nn
import torch
from tqdm import tqdm
import time
import random
import numpy as np
import uuid

class MultiVariableClassifier(nn.Module):
    def __init__(self, model_hyperparameters):
        super().__init__()
        self.model_hyperparameters = model_hyperparameters
        for item, value in model_hyperparameters.items():
            setattr(self, item, value)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.__gen_uiid()
        self.fix_seed()
        self.generate_classifier()

        # have to be called after layer models
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def __gen_uiid(self):
        self.classifier_id = str(uuid.uuid1())

    def fix_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def generate_classifier(self):
        # Camada oculta proporcional ao espaço latente completo (todos os especialistas)
        hidden_size = max(self.input_size // 4, self.n_classes * 8)
        self.predictor = nn.Sequential(
            nn.Linear(self.input_size, hidden_size).to(self.device),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, self.n_classes).to(self.device),
            nn.LogSoftmax(dim=-1).to(self.device),
        )

    def forward(self, inputs):
        # Dynamically get device from parameters instead of a fixed self.device
        device = next(self.parameters()).device
        # BUGFIX: O encodór produz (batch, 1, 256). Reshape para (batch, 256) antes do Linear.
        if inputs.dim() == 3:
            inputs = inputs.squeeze(1)
        return self.predictor(inputs.to(device))

    def fit(self, data_instance, validation=True):
        self.data_instance = data_instance
        self.data_train = self.data_instance.data_train
        self.validation = validation
        if self.validation:
            self.data_validation = self.data_instance.data_val

        progress_bar = tqdm(range(self.epochs), desc="Training: ")

        self.loss_train, self.loss_val = list(), list()
        start = time.time()

        for epoch in progress_bar:
            for batch in self.data_train:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                classifier_out = self.forward(inputs)
                classified = classifier_out.view(
                    classifier_out.shape[0], classifier_out.shape[-1]
                )

                loss = self.loss_function(classified, targets)

                self.loss_train.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                for batch in self.data_validation:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    classifier_out = self.forward(inputs)
                    classified = classifier_out.view(
                        classifier_out.shape[0], classifier_out.shape[-1]
                    )

                    loss = self.loss_function(classified, targets)

                    self.loss_val.append(loss.item())

        end = time.time()
        self.elapsed_time = end - start
