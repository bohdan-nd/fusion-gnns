from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import SAGEConv, GATConv, GPSConv, GINEConv, TransformerConv, GCNConv
from torch_geometric.utils import add_self_loops
import torch_geometric.transforms as T
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import torch
import wandb
import json
from constants import *
import argparse

from mlp import MLPBaseline
from graph_baseline import train_graph_model, num_atom_type, num_chirality_tag, num_bond_type, num_bond_direction

class FusionGNN(nn.Module):
    def __init__(self,
                 conv_name="GINEConv",
                 num_layer=5,
                 context_dim=908,
                 emb_dim=300,
                 feat_dim=512,
                 mlp_hidden_dims: list[int] = [256, 64],
                 number_of_fusion_layers=0,
                 initialization: str = "Context",
                 keep_original_context=False,
                 device=torch.device('cuda')):

        super().__init__()

        self.num_layer = num_layer
        self.inject_layer = num_layer - number_of_fusion_layers
        self.conv_name = conv_name
        self.emb_dim = emb_dim
        self.device = device
        self.feat_dim = feat_dim
        self.drop_ratio = 0
        self.initialization = initialization
        self.keep_original_context = keep_original_context

        self.mlp = MLPBaseline.create_mlp(
            feat_dim // 2 * 2 + emb_dim,
            mlp_hidden_dims
        )

        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList()
        for _ in range(self.num_layer):
            nnchick = nn.Sequential(
                nn.Linear(emb_dim, 2*emb_dim),
                nn.ReLU(),
                nn.Linear(2*emb_dim, emb_dim)
            )

            if conv_name == "GINEConv":
                conv = GINEConv(nnchick)
            elif conv_name == "GPSConv":
                conv = GPSConv(emb_dim, GINEConv(nnchick), heads=4, attn_dropout=0.5)
            elif conv_name == "GATConv":
                conv = GATConv(emb_dim, emb_dim, add_self_loops=True, concat=False)
            elif conv_name == "GCNConv":
                conv = GCNConv(emb_dim, emb_dim)

            self.gnns.append(conv)

        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        self.context_gnn = SAGEConv(in_channels=(-1, -1), out_channels=self.emb_dim)
        self.node_gnn = GATConv(in_channels=(-1, -1), out_channels=self.emb_dim, add_self_loops=False)

        self.pool = global_mean_pool

        self.contex_encoder = nn.Sequential(
            nn.Linear(context_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, emb_dim)
        )

        self.out_lin = nn.Sequential(
            nn.Linear(self.emb_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim//2)
        )
    
    def _node_to_node(self, layer, h, edge_index, batch, edge_embeddings):
        if self.conv_name == "GPSConv":
            h = self.gnns[layer](h, edge_index, batch, edge_attr=edge_embeddings)
        elif self.conv_name == "GCNConv":
            h = self.gnns[layer](h, edge_index)
        else:
            h = self.gnns[layer](h, edge_index, edge_embeddings)
        
        return h
    
    def _create_drug_context_edges(self, drug):
        return torch.cat([
            drug.batch.unsqueeze(0),
            torch.arange(drug.batch.size(0)).unsqueeze(0).to(self.device)
        ], dim=0)
    
    def _create_context_drug_edges(self, drug):
        return torch.cat([
            torch.arange(drug.batch.size(0)).unsqueeze(0).to(self.device),
            drug.batch.unsqueeze(0),
        ], dim=0)

    def forward(self, drugA, drugB, context):
        xA = drugA.x
        batchA = drugA.batch
        edge_indexA = drugA.edge_index
        edge_attrA = drugA.edge_attr

        xB = drugB.x
        batchB = drugB.batch
        edge_indexB = drugB.edge_index
        edge_attrB = drugB.edge_attr

        drugA_context_edges = self._create_drug_context_edges(drugA)
        drugB_context_edges = self._create_drug_context_edges(drugB)

        context_drugA_edges = self._create_context_drug_edges(drugA)
        context_drugB_edges = self._create_context_drug_edges(drugB)

        # node+edge encoding
        hA = self.x_embedding1(xA[:, 0])
        hB = self.x_embedding1(xB[:, 0])
        edge_embeddingsA = self.edge_embedding1(edge_attrA[:, 0])
        edge_embeddingsB = self.edge_embedding1(edge_attrB[:, 0])

        original_context = self.contex_encoder(context)

        if self.initialization == "Bert":
            context = torch.empty(context.shape[0], self.emb_dim)
            torch.nn.init.normal_(context, std=0.02)
        elif self.initialization == "Graph":
            context_A = self.pool(hA, drugA.batch)
            context_B = self.pool(hB, drugB.batch)
            context = (context_A + context_B) / 2
        elif self.initialization == "Context":
            context = original_context

        context = context.to(self.device)

        for layer in range(self.num_layer):
            if layer >= self.inject_layer:

                hA_group = self.context_gnn(
                    (context, hA),
                    drugA_context_edges
                )
                hA = hA_group + hA

                hA = self._node_to_node(layer, hA, edge_indexA, batchA, edge_embeddingsA)
                hA = self.batch_norms[layer](hA)

                if layer != self.num_layer - 1:
                    hA = F.relu(hA)

                hB_group = self.context_gnn(
                    (context, hB),
                    drugB_context_edges
                )
                hB = hB_group + hB
                
                hB = self._node_to_node(layer, hB, edge_indexB, batchB, edge_embeddingsB)
                hB = self.batch_norms[layer](hB)

                if layer != self.num_layer - 1:
                    hB = F.relu(hB)

                # update context
                contextA = self.node_gnn(
                    (hA, context),
                    context_drugA_edges
                )
                contextB = self.node_gnn(
                    (hB, context),
                    context_drugB_edges
                )
                context = (contextA+contextB) / 2

            else:
                hA = self._node_to_node(layer, hA, edge_indexA, batchA, edge_embeddingsA)
                hA = self.batch_norms[layer](hA)

                if layer != self.num_layer - 1:
                    hA = F.relu(hA)
                
                hB = self._node_to_node(layer, hB, edge_indexB, batchB, edge_embeddingsB)
                hB = self.batch_norms[layer](hB)

                if layer != self.num_layer - 1:
                    hB = F.relu(hB)

        hA = self.pool(hA, drugA.batch)
        hA = self.out_lin(hA)

        hB = self.pool(hB, drugB.batch)
        hB = self.out_lin(hB)

        if self.keep_original_context:
            context = original_context

        drugAB_input = torch.concat((hA, hB, context), dim=1)
        output = self.mlp(drugAB_input)
        return output


def create_model(device):
    model = FusionGNN(
        conv_name=wandb.config["conv_name"],
        num_layer=wandb.config["num_layers"],
        context_dim=wandb.config["context_dim"],
        emb_dim=wandb.config["emb_dim"],
        number_of_fusion_layers=wandb.config["number_of_fusion_layers"],
        device=device
    ).to(device)

    return model


def main(config):
    wandb.init(config=config,
               tags=["graph", "UniversalMultiGNN", "MLP"],
               project="drug_synergy",
               entity="uoft-research-2023",
               )

    print('Hyper parameters:')
    print(wandb.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device)

    # transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    # train_graph_model(model, device, transform)

    if config["dataset_name"] == "drugcomb":
        dataset_folder = DRUGCOMB_DATA_FOLDER
    elif config["dataset_name"] == "oneil":
        dataset_folder = ONEIL_DATA_FOLDER

    train_graph_model(model, device, dataset_folder, use_scheduler=config["use_scheduler"], lr=wandb.config["lr"])

    wandb.finish()


if __name__ == '__main__':
    with open(GRAPH_BASELINE_CONFIG, 'r') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description='Run a Universal MultiGNN')
    parser.add_argument('--conv_name', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--synergy_score', type=str)
    parser.add_argument('--fold_number', type=int)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--number_of_fusion_layers', type=int, default=0)
    parser.add_argument('--context_dim', type=int, default=908)
    parser.add_argument('--use_scheduler', type=bool, default=False)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-5)

    args = vars(parser.parse_args())
    config.update(args)

    main(config)
