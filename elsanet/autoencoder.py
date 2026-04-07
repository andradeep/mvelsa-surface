import torch.nn as nn
import torch
import uuid


class AutoEncoder(nn.Module):
    def __init__(self, autoencoder_architecture, activation, device=None):
        super().__init__()

        # BUGFIX: store the target device before generating layers.
        # _set_initial_device() called before generate_autoencoder() is a no-op
        # because `.to(device)` only moves existing parameters.
        self._target_device = device
        self.__get_activation_function(activation)
        self.__decompose_autoencoder_architecture(autoencoder_architecture)
        self.__gen_uiid()

        self.generate_autoencoder()
        # Now parameters exist — move the whole model to the target device
        if self._target_device is not None:
            self.to(self._target_device)

    def __gen_uiid(self):
        self.autoencoder_id = str(uuid.uuid1())

    def __inicialize_autoencoders(self):
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

    def __decompose_autoencoder_architecture(
        self, autoencoder_architecture: list
    ) -> None:
        self.autoencoder_architecture = autoencoder_architecture
        self.encoder_architecture = self.autoencoder_architecture
        self.decoder_architecture = self.autoencoder_architecture.copy()
        self.decoder_architecture.reverse()

    def __gen_simple_model_architecture_format(self):
        encoder = str(self.encoder_architecture)
        decoder = str(self.decoder_architecture)
        self.simple_architecture_format = encoder + "<->" + decoder

    def __get_activation_function(self, activation: str) -> None:
        activation = activation + " "
        space_index = activation.index(" ")
        self.no_activation_at_end = activation[space_index:].strip()

        self.activation = activation[:space_index]

    def __generate_encoder(self):
        encoder_layers = nn.Sequential()

        for idx in range(len(self.encoder_architecture) - 1):
            encoder_layers.append(
                nn.Linear(
                    self.encoder_architecture[idx], self.encoder_architecture[idx + 1]
                ).to(self.device)
            )
            encoder_layers.append(getattr(nn, self.activation)())

        if self.no_activation_at_end == "n":
            self.encoders.append(encoder_layers[:-1])
        else:
            self.encoders.append(encoder_layers)

    def __generate_decoder(self):
        decoder_layers = nn.Sequential()
        for idx in range(len(self.decoder_architecture) - 1):
            decoder_layers.append(
                nn.Linear(
                    self.decoder_architecture[idx], self.decoder_architecture[idx + 1]
                ).to(self.device)
            )
            decoder_layers.append(getattr(nn, self.activation)())

        if self.no_activation_at_end == "n":
            self.decoders.append(decoder_layers[:-1])
        else:
            self.decoders.append(decoder_layers)

    def generate_autoencoder(self):
        self.__inicialize_autoencoders()
        self.__generate_encoder()
        self.__generate_decoder()
        self.__gen_simple_model_architecture_format()

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _set_initial_device(self, device):
        # Kept for backwards compatibility; actual device placement now happens
        # at the end of __init__ via self.to(self._target_device).
        pass

    def forward(self, batch):
        encoded_batch, decoded_batch = [], []
        # Ensure input is on the same device as the module
        device = self.device
        for ae_input in batch:
            encoded_batch.append(
                torch.stack(
                    [
                        encoder(xs.to(device))
                        for encoder, xs in zip(self.encoders, ae_input)
                    ]
                )
            )
        # stack is used to transform a list of tensor in a unic tensor of tensor
        for enc in encoded_batch:
            decoded_batch.append(
                torch.stack(
                    [
                        decoder(z.to(device))
                        for decoder, z in zip(self.decoders, enc)
                    ]
                )
            )
        return torch.stack(encoded_batch), torch.stack(decoded_batch)

    def set_decoders_weights_unadjustable(self):
        for param in self.decoders.parameters():
            param.requires_grad = False

    def set_encoders_weights_unadjustable(self):
        for param in self.encoders.parameters():
            param.requires_grad = False

    def set_autoencoders_weights_unadjustable(self):
        self.set_decoders_weights_unadjustable()
        self.set_encoders_weights_unadjustable()
