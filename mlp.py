import torch
from torch import nn


class LinearBlock(nn.Module):
    ''' Linear -> LeakyReLU -> Dropout'''

    def __init__(self, input_dim: int, output_dim: int, dropout: int = 0.0, slope: float = -0.01):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.LeakyReLU(slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        return self.dropout(self.relu(self.fc(X)))


class MLPBaseline(torch.nn.Module):
    def __init__(self, drug_encoder: str, mlp_input_dim: int, mlp_hidden_dims: list[int], dropout: float):
        super().__init__()
        self.drug_encoder = drug_encoder

        self.mlp = self.create_mlp(
            mlp_input_dim,
            mlp_hidden_dims,
            dropout
        )

    @staticmethod
    def _create_cell_line_transformation(input_dim: int, output_dim: int):
        return nn.Linear(input_dim, output_dim)

    @staticmethod
    def create_mlp(input_dim: int, hidden_dims: list[int], dropout: float = 0.0, slope: float = -0.01, output_dim: int = 1):
        mlp = nn.Sequential(
            LinearBlock(input_dim, hidden_dims[0], dropout, slope),
            *[LinearBlock(input_, output_, dropout) for input_, output_ in zip(hidden_dims, hidden_dims[1:])],
            nn.Linear(hidden_dims[-1], output_dim)
        )

        linear_layers = [m for m in mlp.modules() if (isinstance(m, nn.Linear))]

        for layer in linear_layers[:-1]:
            nn.init.kaiming_uniform_(layer.weight.data, a=slope)
            nn.init.uniform_(layer.bias.data, -1, 0)

        last_linear_layer = linear_layers[-1]
        nn.init.xavier_normal_(last_linear_layer.weight.data)
        nn.init.uniform_(last_linear_layer.bias.data, -1, 0)

        return mlp

    def _obtain_drug_embedding(self, drug):
        raise NotImplementedError

    def freeze_encoder_layers(self):
        for p in self.drug_encoder.parameters():
            p.requires_grad = False

    def forward(self, drugA, drugB, cell_line, average_predictions: bool = False):
        drugA_embedding = self._obtain_drug_embedding(drugA)
        drugB_embedding = self._obtain_drug_embedding(drugB)

        drugAB_input = torch.concat((drugA_embedding, drugB_embedding, cell_line), dim=1)
        output = self.mlp(drugAB_input)

        if average_predictions:
            drugBA_input = torch.concat((drugB_embedding, drugA_embedding, cell_line), dim=1)
            drugBA_output = self.mlp(drugBA_input)
            output = (output+drugBA_output)/2
        
        return output
