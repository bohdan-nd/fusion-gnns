from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch import nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import wandb
import numpy as np
import pandas as pd
import json
import argparse

from constants import *
from utils import *
from mlp import MLPBaseline

# import warnings
# warnings.filterwarnings("ignore")

seed_everything(seed=RANDOM_SEED)

num_atom_type = 119
num_chirality_tag = 3

num_bond_type = 5
num_bond_direction = 3

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def _get_drug_tokens(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    N = mol.GetNumAtoms()
    M = mol.GetNumBonds()

    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def get_mol_dict(df):
    mols = pd.concat([
        df.rename(columns={'Drug1_ID': 'id', 'Drug1': 'drug'})[['id', 'drug']],
        df.rename(columns={'Drug2_ID': 'id', 'Drug2': 'drug'})[['id', 'drug']]
    ],
        axis=0, ignore_index=True
    ).drop_duplicates(subset=['id'])

    dct = {}
    for _, x in tqdm(mols.iterrows(), total=len(mols)):
        dct[x['id']] = _get_drug_tokens(x['drug'])
    return dct


class GINEConv(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2*in_dim),
            nn.ReLU(),
            nn.Linear(2*in_dim, out_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, in_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, in_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return (x_j + edge_attr).relu()

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim, emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim//2)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)

        return h, out


class DrugCombDataset(Dataset):
    def __init__(self, drugcomb, cell_lines, mol_mapping, transform=None):
        self.drugcomb = drugcomb
        self.mol_mapping = mol_mapping
        self.cell_lines = cell_lines
        self.targets = torch.from_numpy(drugcomb['target'].values)
        self.transform = transform

    def __len__(self):
        return len(self.drugcomb)

    def __getitem__(self, idx):
        sample = self.drugcomb.iloc[idx]

        drug1 = sample[DRUG1_ID_KEY]
        drug2 = sample[DRUG2_ID_KEY]
        drug1_tokens = self.mol_mapping[drug1]
        drug2_tokens = self.mol_mapping[drug2]

        if self.transform:
            drug1_tokens = self.transform(drug1_tokens)
            drug2_tokens = self.transform(drug2_tokens)

        cell_line_name = sample[CELL_LINE_COLUMN_NAME]
        cell_line_embeddings = self.cell_lines.loc[cell_line_name].values.flatten()
        cell_line_embeddings = torch.tensor(cell_line_embeddings)

        target = self.targets[idx]  # sample["target"]

        return {
            DRUG1_KEY: drug1_tokens,
            DRUG2_KEY: drug2_tokens,
            CELL_LINE_KEY: cell_line_embeddings,
            TARGET_KEY: target
        }


class GraphMLPBaseline(MLPBaseline):
    def __init__(self, drug_encoder: str, mlp_input_dim: int, mlp_hidden_dims: list[int], dropout: float):
        super().__init__(drug_encoder, mlp_input_dim, mlp_hidden_dims, dropout)

    def _obtain_drug_embedding(self, drug):
        return self.drug_encoder(drug)[1]


def extract_data_from_batch(batch):
    drugA = batch[DRUG1_KEY]
    drugB = batch[DRUG2_KEY]
    cell_line = batch[CELL_LINE_KEY]
    target = batch[TARGET_KEY].unsqueeze(1).to(torch.float32)

    return drugA, drugB, cell_line, target


def evaluate_mlp(model, loader, loss_fn, device):
    model.eval()

    epoch_preds, epoch_labels = [], []
    epoch_loss = 0.0

    for batch in loader:
        drugA, drugB, cell_line, target = extract_data_from_batch(batch)
        drugA = drugA.to(device)
        drugB = drugB.to(device)
        cell_line = cell_line.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(drugA, drugB, cell_line)

        loss = loss_fn(output, target)
        epoch_preds.append(output.detach().cpu())
        epoch_labels.append(target.detach().cpu())
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    epoch_preds = torch.cat(epoch_preds)
    epoch_labels = torch.cat(epoch_labels)

    auprc = calculate_auprc(epoch_labels, epoch_preds)
    auc = calculate_roc_auc(epoch_labels, epoch_preds)

    wandb.log({"val_auprc": auprc, "val_auc": auc, "val_loss": epoch_loss})


def create_model(device):
    drug_encoder = GINet(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool='mean').to(device)
    state_dict = torch.load(MOLCLR_CKPT, map_location=device)
    drug_encoder.load_state_dict(state_dict)

    baseline_mlp = GraphMLPBaseline(drug_encoder=drug_encoder,
                                    mlp_input_dim=wandb.config["mlp_input_dim"],
                                    mlp_hidden_dims=wandb.config["mlp_hidden_dims"],
                                    dropout=wandb.config["dropout"])

    return baseline_mlp


def train_graph_model(model, device, data_folder, transform=None, use_scheduler=False, lr:float = 1e-5):
    drugcomb = pd.read_feather(f"{data_folder}/{wandb.config['synergy_score']}.feather")

    cell_lines = pd.read_feather(f"{data_folder}/{CELL_LINE_FILE_NAME}").set_index("cell_line_name")
    cell_lines = cell_lines.astype(np.float32)

    mol_mapping = get_mol_dict(drugcomb)

    with open(f"{data_folder}/{FOLDS_FOLDER_NAME}/{wandb.config['synergy_score']}.json") as f:
        folds = json.load(f)

    fold = folds[f"fold_{wandb.config['fold_number']}"]
    X_train, X_val, X_test = split_fold(drugcomb, fold)

    if wandb.config["double_training"]:
        X_train = get_double_df(X_train)

    train_set = DrugCombDataset(X_train, cell_lines, mol_mapping, transform)
    val_set = DrugCombDataset(X_val, cell_lines, mol_mapping, transform)

    batch_size = wandb.config["batch_size"]
    train_sampler = create_train_sampler(X_train, wandb.config["weighted_sampler"])

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=1, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=1, shuffle=False)

    # log_train_params(model)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    if use_scheduler:
        scheduler = CyclicLR(optimizer, max_lr=1e-4, base_lr=1e-5, mode="exp_range", gamma=.995, cycle_momentum=False)

    if wandb.config["weighted_loss"]:
        pos_weight = torch.tensor(drugcomb[TARGET_COLUMN_NAME].sum() / len(drugcomb[TARGET_COLUMN_NAME]))
    else:
        pos_weight = None

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()

    for _ in tqdm(range(wandb.config["number_of_epochs"])):
        epoch_preds, epoch_labels = [], []
        epoch_loss = 0.0

        for batch in train_loader:

            drugA, drugB, cell_line, target = extract_data_from_batch(batch)
            drugA = drugA.to(device)
            drugB = drugB.to(device)
            cell_line = cell_line.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(drugA, drugB, cell_line)
            loss = loss_fn(output, target)

            epoch_preds.append(output.detach().cpu())
            epoch_labels.append(target.detach().cpu())
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            if use_scheduler:
                scheduler.step()

        epoch_loss = epoch_loss / len(train_loader)
        epoch_preds = torch.cat(epoch_preds)
        epoch_labels = torch.cat(epoch_labels)

        auprc = calculate_auprc(epoch_labels, epoch_preds)
        auc = calculate_roc_auc(epoch_labels, epoch_preds)

        wandb.log({"train_auprc": auprc, "train_auc": auc, "train_loss": epoch_loss})

        evaluate_mlp(model, val_loader, loss_fn, device)


def main(config):
    wandb.init(config=config,
               tags=["graph", "MolCLR", "MLP"],
               project="drug_synergy",
               entity="uoft-research-2023",
               )

    print('Hyper parameters:')
    print(wandb.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device)
    train_graph_model(model, device)
    wandb.finish()


if __name__ == '__main__':
    with open(GRAPH_BASELINE_CONFIG, 'r') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description='Run a Graph Baseline')
    parser.add_argument('--synergy_score', type=str)
    parser.add_argument('--fold_number', type=int)
    args = vars(parser.parse_args())
    config.update(args)

    main(config)
