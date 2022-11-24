import torch.nn as nn
import torch.optim.optimizer
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader


def get_sample_shape(dataset):
    if isinstance(dataset, DataLoader):
        batch, targets = next(iter(dataset))
        return batch.shape[0], batch.shape[2], targets.shape[0]
    else:
        raise TypeError("Invalid input dataset type")


class FCNN(LightningModule):
    def __init__(self,
                 batch_size, n_features, n_targets,
                 hidden_dims=None, dropout=0, bn=True, actf=nn.LeakyReLU(),
                 optimizer=torch.optim.Adam, optimizer_parameters=None):
        super(FCNN, self).__init__()

        self.batch_size = batch_size
        self.n_features = n_features
        self.n_targets = n_targets

        hidden_dims = [] if hidden_dims is None else hidden_dims
        optimizer_parameters = {} if optimizer_parameters is None else optimizer_parameters

        self.hidden_dims = [n_features] + list(hidden_dims)
        self.n_layers = len(self.hidden_dims)
        self.dropout = dropout
        self.bn = bn
        self.actf = actf
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters

        self.loss = F.mse_loss
        # self.loss = self.cross_entropy_loss

        self.sequential = self.make_structure()

        self.valid_losses = []
        self.train_losses = []

    def make_structure(self):
        layers = []
        for i, val in enumerate(self.hidden_dims[:-1]):
            layers += [self.lin_block(self.hidden_dims[i], self.hidden_dims[i + 1])]
        layers += [nn.Linear(self.hidden_dims[-1], self.n_targets)]
        return nn.Sequential(*layers)

    def lin_block(self, in_f, out_f, *args, **kwargs):
        return nn.Sequential(
            nn.Linear(in_features=in_f, out_features=out_f, *args, **kwargs),
            nn.Dropout(self.dropout),
            self.actf,
        )

    def forward(self, x):
        x = x.to(torch.float32)
        return self.sequential(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_parameters)
        return optimizer

    def get_model_structure(self):
        return {
            "hidden_dims": self.hidden_dims,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "bn": self.bn,
            "actf": type(self.actf).__name__,
            "optimizer": type(self.optimizer).__name__,
            "optimizer_parameters": self.optimizer_parameters,
        }

    def training_step(self, data, *args, **kwargs):
        x, target = data
        output = self.forward(x)
        loss = self.loss(target, output)
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, data, *args, **kwargs):
        x, target = data
        output = self.forward(x)
        loss = self.loss(target, output)
        self.log('val_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.valid_losses.append(float(avg_loss))

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.train_losses.append(float(avg_loss))

    def get_architecture(self):
        return {"conv_layer": self.conv_layer,
                "hidden_conv": self.hidden_conv,
                "hidden_linear": self.hidden_linear,
                "n_conv": self.n_conv,
                "n_linear": self.n_linear,
                "conv_dropout": self.conv_dropout,
                "linear_dropout": self.linear_dropout,
                "linear_bn": self.linear_bn,
                "conv_actf": self.conv_actf,
                "linear_actf": self.linear_actf,
                "optimizer": self.optimizer,
                }

    def freeze_layers(self, number_of_layers: int) -> None:
        pass
        # n_l = self.n_linear
        # parameters = list(self.named_parameters())
        # if number_of_layers <= n_c:
        #     for i in range(number_of_layers):
        #         for name, tensor in parameters:
        #             if name.startswith(f"conv_sequential.module_{i}"):
        #                 tensor.requires_grad = False
