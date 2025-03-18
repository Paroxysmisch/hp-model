import random

import torch
import torch.nn.functional as F
import torch_scatter
from mamba_ssm import Mamba, Mamba2
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.models import GIN, GAT
from torch_geometric.nn.pool import global_add_pool


class RNN_LSTM_onlyLastHidden(nn.Module):
    """
    LSTM version that just uses the information from the last hidden state
    since the last hidden state has information from all previous states
    basis for BiDirectional LSTM
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(RNN_LSTM_onlyLastHidden, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        # change basic RNN to LSTM
        # num_layers Default: 1
        # bias Default: True
        # batch_first Default: False
        # dropout Default: 0
        # bidirectional Default: False
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # remove the sequence_length
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Get data to cuda if possible
        x = x.to(self.device)
        # print("input x.size() = ", x.size())
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # LSTM needs a separate cell state (LSTM needs both hidden and cell state)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        # need to give LSTM both hidden and cell state (h0, c0)
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        # no need to reshape the out or concat
        # out is going to take all mini-batches at the same time + last layer + all features
        out = self.fc(out[:, -1, :])
        # print("forward out = ", out)
        return out

    def sample_action(self, obs, epsilon):
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            explore_action = random.randint(0, 2)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()


class MambaModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(MambaModel, self).__init__()
        self.hidden_size = hidden_size
        self.expander = nn.Linear(input_size, self.hidden_size)
        self.mamba_layers = nn.Sequential(
            *[
                Mamba(d_model=hidden_size, d_state=64, d_conv=2, expand=2)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = self.expander(x)
        x = self.mamba_layers(x)
        out = self.fc(x[:, -1, :])

        return out

    def sample_action(self, obs, epsilon):
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            explore_action = random.randint(0, 2)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :].to(x.device)


def create_attention_mask(seq_len, max_offset=2):
    # True indicates values that should not participate in the attention calculation
    mask = torch.zeros(seq_len, seq_len).bool()
    mask |= torch.triu(torch.ones(seq_len, seq_len).bool(), diagonal=(max_offset + 1))
    mask |= torch.tril(torch.ones(seq_len, seq_len).bool(), diagonal=-(max_offset + 1))
    return mask


class TransformerModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, num_classes, device, mask=False
    ):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.expander = nn.Linear(input_size, self.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        self.positional_encoding = PositionalEncoding(d_model=hidden_size)
        self.mask = mask

    def forward(self, x):
        x = x.to(self.device)
        x = self.expander(x)
        x = self.positional_encoding(x)
        if self.mask:
            x = self.transformer(x, mask=create_attention_mask(x.size(-2)))
        else:
            x = self.transformer(x)
        out = self.fc(x[:, -1, :])

        return out

    def sample_action(self, obs, epsilon):
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            explore_action = random.randint(0, 2)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()


def fully_connected_directed_edge_index(n):
    row, col = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)
    return edge_index


class GATModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        device,
        fully_connected=False,
    ):
        super(GATModel, self).__init__()
        self.hidden_size = hidden_size
        self.expander = nn.Linear(input_size, self.hidden_size)
        self.num_layers = num_layers
        self.gnn = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    add_self_loops=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        self.positional_encoding = PositionalEncoding(d_model=hidden_size)
        self.fully_connected = fully_connected

    def forward(self, x):
        x = x.to(self.device)
        N = x.size(-2)
        if self.fully_connected:
            edge_index = fully_connected_directed_edge_index(N).to(self.device)
        else:
            edges = (
                torch.tensor([[i, i + 1] for i in range(N - 1)]).to(self.device).T
            )  # Create edges (0→1, 1→2, ..., N-2→N-1)
            # Convert to PyG edge_index format (2, num_edges)
            edge_index = torch.cat(
                [edges, edges.flip(0)], dim=1
            )  # Add reverse edges for undirected graph

        x = self.expander(x)
        x = self.positional_encoding(x)

        # Use the PyTorch Geometric Batch class if x is batched
        if x.dim() == 3:
            data_list = [Data(x=x[i], edge_index=edge_index) for i in range(len(x))]
            batch = Batch.from_data_list(data_list)
            for conv in self.gnn:
                x = conv(batch.x, batch.edge_index)
            out = global_add_pool(x, batch.batch)
        else:
            for conv in self.gnn:
                x = conv(x, edge_index)
            out = global_add_pool(x)

        out = self.fc(out)

        return out

    def sample_action(self, obs, epsilon):
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            explore_action = random.randint(0, 2)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()


class GINModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        device,
        fully_connected=False,
    ):
        super(GINModel, self).__init__()
        self.hidden_size = hidden_size
        self.expander = nn.Linear(input_size, self.hidden_size)
        self.num_layers = num_layers
        self.gnn = GIN(
            in_channels=hidden_size,
            out_channels=hidden_size,
            hidden_channels=hidden_size,
            num_layers=self.num_layers,
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        self.positional_encoding = PositionalEncoding(d_model=hidden_size)
        self.fully_connected = fully_connected

    def forward(self, x):
        x = x.to(self.device)
        N = x.size(-2)
        if self.fully_connected:
            edge_index = fully_connected_directed_edge_index(N).to(self.device)
        else:
            edges = (
                torch.tensor([[i, i + 1] for i in range(N - 1)]).to(self.device).T
            )  # Create edges (0→1, 1→2, ..., N-2→N-1)
            # Convert to PyG edge_index format (2, num_edges)
            edge_index = torch.cat(
                [edges, edges.flip(0)], dim=1
            )  # Add reverse edges for undirected graph

        x = self.expander(x)
        x = self.positional_encoding(x)

        # Use the PyTorch Geometric Batch class if x is batched
        if x.dim() == 3:
            data_list = [Data(x=x[i], edge_index=edge_index) for i in range(len(x))]
            batch = Batch.from_data_list(data_list)
            out = self.gnn(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
            out = out.reshape(-1, N, self.hidden_size)
        else:
            out = self.gnn(x=x, edge_index=edge_index)

        out = torch.sum(out, dim=-2)
        out = self.fc(out)

        return out

    def sample_action(self, obs, epsilon):
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            explore_action = random.randint(0, 2)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=100):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float()
#             * (-torch.log(torch.tensor(10000.0)) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.pe = pe  # Shape: (max_len, d_model)
#
#     def forward(self, x):
#         return self.pe[: x.size(-2), :].to(x.device)
#
#
# class GATModel(nn.Module):
#     def __init__(
#         self,
#         input_size,
#         hidden_size,
#         num_layers_gnn,
#         num_layers_lstm,
#         num_classes,
#         device,
#     ):
#         super(GATModel, self).__init__()
#         # GNN is used to capture the layout so far, and this forms the initial hidden state for the LSTM
#         # Also the LSTM loses access to the action information, as that is already captured by the GNN
#         self.hidden_size = hidden_size
#         self.num_layers_gnn = num_layers_gnn
#         self.num_layers_lstm = num_layers_lstm
#         self.device = device
#         self.expander = nn.Linear(hidden_size, hidden_size)
#         self.gnn = nn.ModuleList(
#             [
#                 GATv2Conv(
#                     in_channels=self.hidden_size,
#                     out_channels=self.hidden_size,
#                     add_self_loops=False,
#                 )
#                 for _ in range(self.num_layers_gnn)
#             ]
#         )
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers_lstm, batch_first=True)
#         self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x, graph_batch):
#         x = x.to(self.device)
#         graph_batch = graph_batch.to(
#             self.device
#         )  # [graphs, states so far, 2 for the x and y coordinates]
#
#         # prev_pos = graph_batch.x[:, -1, :]
#         # prev_prev_pos = graph_batch.x[:, -2, :]
#         # direction_order = torch.tensor([(0, 1), (1, 0), (0, -1), (-1, 0)]).float() # up, right, down, left
#         # cur_direction = prev_pos - prev_prev_pos
#         # cur_direction_idx = direction_order.index(cur_direction)
#         graph_embeddings = self.expander(graph_batch.x)
#         for conv_layer in self.gnn:
#             graph_embeddings = conv_layer(graph_embeddings, graph_batch.edge_index)
#         graph_embeddings = torch_scatter.scatter_mean(
#             graph_embeddings, graph_batch.batch, dim=0
#         )
#
#         # h0 = torch.stack([graph_embeddings for _ in range(self.num_layers_lstm)])
#         h0 = torch.zeros(self.num_layers_lstm, x.size(0), self.hidden_size).to(
#             self.device
#         )
#         c0 = torch.zeros(self.num_layers_lstm, x.size(0), self.hidden_size).to(
#             self.device
#         )
#
#         out, _ = self.lstm(
#             x, (h0, c0)
#         )  # out: tensor of shape (batch_size, seq_length, hidden_size)
#         out = torch.cat([out[:, -1, :], graph_embeddings], dim=-1)
#
#         out = F.relu(self.fc1(out))
#         out = self.fc2(out)
#         return out
#
#     def sample_action(self, obs, graph_batch, epsilon):
#         """
#         greedy epsilon choose
#         """
#         coin = random.random()
#         if coin < epsilon:
#             explore_action = random.randint(0, 2)
#             return explore_action
#         else:
#             # print("exploit")
#             out = self.forward(obs, graph_batch)
#             return out.argmax().item()
