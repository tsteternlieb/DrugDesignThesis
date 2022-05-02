from turtle import forward
from black import out
from more_itertools import last
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import RelGraphConv, GatedGraphConv
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import spectral_norm
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F

device = None


def dropout_inference(m: nn.Dropout):
    if isinstance(m, nn.Dropout):
        m.eval()


def init_weights_ortho(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)


class Linear_3(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Linear_3, self).__init__()
        self.Dense1 = nn.Linear(in_dim, hidden_dim)
        self.Dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.Dense3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, inputs):
        out = torch.tanh(self.Dense1(inputs))
        out = torch.tanh(self.Dense2(out))
        out = self.Dense3(out)
        return out


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers,
        drop_out,
        activation,
        normalization=None,
    ):
        super(LinearBlock, self).__init__()
        self.block = nn.ModuleList([])
        for i, _ in enumerate(range(num_layers)):
            block = []
            # transformation
            if i == 0:
                block.append(nn.Linear(in_dim, hidden_dim))

            else:
                if normalization == "SpectralNorm":
                    block.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)))
                    print("231")
                else:
                    block.append(nn.Linear(hidden_dim, hidden_dim))

            # normalization
            if normalization == "LayerNorm":
                block.append(nn.LayerNorm(hidden_dim))

            # activations
            if activation == "ReLU":
                block.append(nn.ReLU())
            elif activation == "ELU":
                block.append(nn.ELU())
            elif activation == "LeakyReLU":
                block.append(nn.LeakyReLU())

            # dropout
            block.append(nn.Dropout(drop_out))
            self.block.extend(block)

        self.block.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for layer in self.block:
            x = layer(x)

        return x


class CriticSqueeze(nn.Module):

    """
    Critic class for advantage estimation
    """

    def __init__(self, in_dim, hidden_dim):
        super(CriticSqueeze, self).__init__()
        self.GGN = dgl.nn.GatedGraphConv(in_dim, hidden_dim, 5, 4)
        self.Dense = Linear_3(hidden_dim + 1 + in_dim, hidden_dim, 1)

    def forward(self, graph, last_action_node, last_node):
        # print(graph.adj())
        h = F.relu(
            self.GGN(graph, graph.ndata["atomic"], graph.edata["type"].squeeze())
        )
        with graph.local_scope():
            graph.ndata["h"] = h
            hg = dgl.mean_nodes(graph, "h")
        cat = torch.cat((hg, last_action_node, last_node), dim=1)
        out = self.Dense(cat)
        return out


class EdgePredictor(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, num_layers=3, dropout=0.0, activation="ReLU"
    ):
        """torch module for edge prediction. Paramaterizes edge probability by concating
        final node feat with each other node and running that through a mlp


        Args:
            in_dim (int): input dimension
            hidden_dim (int): hidden dimension
            num_layers (int, optional): number of layers. Defaults to 3.
            dropout (float, optional): dropout level. Defaults to 0.0.
            activation (str, optional): what activation to use. Defaults to 'ReLU'.
        """
        super(EdgePredictor, self).__init__()
        self.NodeEmbed = LinearBlock(
            in_dim, hidden_dim, hidden_dim, 1, dropout, activation
        )
        self.Edges = LinearBlock(
            hidden_dim * 2, hidden_dim, 2, num_layers, dropout, activation
        )

    def forward(self, graphs, last_node_batch, h):
        """
        graph is actually a batch of graphs where len(graph.batch_num_nodes()) = len(last_node_batch)
        returns just a list of edges so its unsplit up rn, bbut
        """
        edges_per_graph = []

        node_embed_batch = self.NodeEmbed(last_node_batch)
        batch_node_stacks = []

        """looping over graphs"""
        for i in range(last_node_batch.shape[0]):
            batch_node_stacks.append(
                node_embed_batch[i].repeat(graphs.batch_num_nodes()[i], 1)
            )

        batch_node_stacks = torch.cat(batch_node_stacks, dim=0)

        stack = torch.cat((h, batch_node_stacks), dim=1)
        edges = self.Edges(stack)
        with graphs.local_scope():
            graphs.ndata["bond_pred"] = edges
            graphs = dgl.unbatch(graphs)
            for graph in graphs:
                edges_per_graph.append(graph.ndata["bond_pred"])

        return pad_sequence(
            edges_per_graph, batch_first=True, padding_value=-10000
        ).flatten(1, -1)


class Batch_Edge(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Batch_Edge, self).__init__()
        # need more theres no non linearity
        self.NodeEmbed = nn.Linear(in_dim, hidden_dim)
        self.Edges = Linear_3(hidden_dim * 2, hidden_dim * 2, 2)

    def forward(self, graphs, last_node_batch, h):
        """
        graph is actually a batch of graphs where len(graph.batch_num_nodes()) = len(last_node_batch)
        returns just a list of edges so its unsplit up rn, bbut
        """
        edges_per_graph = []

        node_embed_batch = self.NodeEmbed(last_node_batch)
        # holds number of graphs of each node embedding stacked to match node number per graph
        batch_node_stacks = []
        # so its first dimension is the batch size, and the second 'dimension' is the number of nodes per grah

        """looping over graphs"""
        for i in range(last_node_batch.shape[0]):
            batch_node_stacks.append(
                node_embed_batch[i].repeat(graphs.batch_num_nodes()[i], 1)
            )

        batch_node_stacks = torch.cat(batch_node_stacks, dim=0)

        stack = torch.cat((h, batch_node_stacks), dim=1)
        edges = self.Edges(stack)
        with graphs.local_scope():
            graphs.ndata["bond_pred"] = edges
            graphs = dgl.unbatch(graphs)
            for graph in graphs:
                edges_per_graph.append(graph.ndata["bond_pred"])

        return pad_sequence(
            edges_per_graph, batch_first=True, padding_value=-10000
        ).flatten(1, -1)


class BaseLine(nn.Module):
    """Crappy base line to check improvements against"""

    def __init__(self, in_dim, hidden_dim, num_nodes):
        super(BaseLine, self).__init__()
        self.in_dim = in_dim
        self.num_nodes = num_nodes
        self.GGN = dgl.nn.GatedGraphConv(in_dim, hidden_dim, 5, 4)
        self.AddNode = Linear_3(hidden_dim, hidden_dim, num_nodes)
        self.BatchEdge = Batch_Edge(in_dim, hidden_dim)

    def forward(self, graph, last_action_node, last_node, mask=False, softmax=True):
        # out = []
        h = F.relu(
            self.GGN(graph, graph.ndata["atomic"], graph.edata["type"].squeeze())
        )
        with graph.local_scope():
            graph.ndata["h"] = h
            hg = dgl.mean_nodes(graph, "h")

        addNode = self.AddNode(hg)

        if mask:
            node_mask = torch.unsqueeze(
                torch.cat((torch.zeros(1), torch.ones(self.num_nodes - 1))), dim=0
            ).to(device)
            mask = last_action_node * (node_mask * -100000)
            addNode += mask

        edges = self.BatchEdge(graph, last_node, h)
        out = torch.cat((addNode, edges), dim=1)
        if softmax:
            return torch.softmax(out, dim=1)
        else:
            return out


class Actor(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        drop_out,
        graph_activation,
        dense_activation,
        model,
        graph_num_layers,
        dense_num_layers,
        norm=None,
        regularizer=None,
        ortho_init=True,
    ):
        super(Actor, self).__init__()
        assert graph_num_layers > 1

        """GRAPH LAYERS"""
        if model == "RelationalGraphConv":
            init_embed = RelGraphConv(
                in_dim,
                hidden_dim,
                3,
                regularizer,
                activation=graph_activation,
                dropout=drop_out,
            )
            self.GraphConv = nn.ModuleList(
                [init_embed]
                + [
                    RelGraphConv(
                        hidden_dim,
                        hidden_dim,
                        3,
                        regularizer,
                        activation=graph_activation,
                        dropout=drop_out,
                    )
                    for _ in range(graph_num_layers)
                ]
            )

        elif model == "GatedGraphConv":
            self.GraphConv = nn.ModuleList(
                [GatedGraphConv(in_dim, hidden_dim, graph_num_layers, 4)]
            )

        else:
            print("Bad Graph Name")
            raise

        """DENSE LAYER"""
        self.dense = LinearBlock(
            hidden_dim + 1,
            hidden_dim * 2,
            out_dim,
            dense_num_layers,
            0,
            dense_activation,
            normalization=norm,
        )

        """EDGE LAYER"""
        self.BatchEdge = EdgePredictor(
            in_dim, hidden_dim, dropout=drop_out, activation=dense_activation
        )

        if ortho_init:
            self.apply(init_weights_ortho)
            
        self.out_dim = out_dim

    def forward(self, graph, last_action_node, last_node, mask=False, softmax=True):
        node_feats = graph.ndata["atomic"]

        for _, layer in enumerate(self.GraphConv):
            node_feats = layer(graph, node_feats, graph.edata["type"].squeeze())
        with graph.local_scope():
            graph.ndata["node_feats"] = node_feats
            hg = dgl.sum_nodes(graph, "node_feats")

        hg = torch.cat((hg, last_action_node), dim=1)
        addNode = self.dense(hg)
        if mask:
            node_mask = torch.unsqueeze(
                torch.cat((torch.zeros(1), torch.ones(self.out_dim - 1))), dim=0
            ).to(device)
            node_mask = node_mask.cuda()
            print(last_action_node.device,node_mask.device)
            mask = last_action_node * (node_mask * -100000)
            print(mask)
            addNode += mask
            print(addNode,'addnode')

        edges = self.BatchEdge(graph, last_node, node_feats)
        out = torch.cat((addNode, edges), dim=1)
        if softmax:
            return torch.softmax(out, dim=1)
        else:
            return out


class Critic(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        model,
        graph_num_layers,
        dense_num_layers,
        graph_activation,
        dense_activation,
        dropout,
        norm=None,
        regularizer=None,
        ortho_init=True,
    ):
        super(Critic, self).__init__()

        """GRAPH LAYERS"""
        if model == "RelationalGraphConv":
            init_embed = RelGraphConv(
                in_dim,
                hidden_dim,
                3,
                regularizer,
                activation=graph_activation,
                dropout=dropout,
            )
            self.GraphConv = nn.ModuleList(
                [init_embed]
                + [
                    RelGraphConv(
                        hidden_dim,
                        hidden_dim,
                        3,
                        regularizer,
                        activation=graph_activation,
                        dropout=dropout,
                    )
                    for _ in range(graph_num_layers)
                ]
            )

        elif model == "GatedGraphConv":
            self.GraphConv = nn.ModuleList(
                [GatedGraphConv(in_dim, hidden_dim, graph_num_layers, 4)]
            )

        else:
            print("Bad Graph Name")
            raise

        """DENSE LAYER"""
        self.dense = LinearBlock(
            hidden_dim + 1 + in_dim,
            hidden_dim * 2,
            1,
            dense_num_layers,
            0,
            dense_activation,
            normalization=norm,
        )

        # '''EDGE LAYER'''
        # self.BatchEdge = EdgePredictor(
        #     in_dim, hidden_dim, dropout=drop_out, activation=dense_activation)

        if ortho_init:
            self.apply(init_weights_ortho)

    def forward(self, graph, last_action_node, last_node, masking=False, softmax=True):
        node_feats = graph.ndata["atomic"]

        for _, layer in enumerate(self.GraphConv):
            node_feats = layer(graph, node_feats, graph.edata["type"].squeeze())
        with graph.local_scope():
            graph.ndata["node_feats"] = node_feats
            hg = dgl.sum_nodes(graph, "node_feats")

        hg = torch.cat((hg, last_action_node, last_node), dim=1)
        out = self.dense(hg)
        return out
