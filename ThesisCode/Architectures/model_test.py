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
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F

from supervised_training.sv_utils import GraphDataLoader
from models import (
    Actor,
    Critic,
    CriticSqueeze,
 
)


def model_test(model: nn.Module):
    model.cuda()
    graph_loader = GraphDataLoader("./GraphDecomp/graphData/full_chunka", 4)
    graphs = graph_loader._get_exs([0 for i in range(10)])
    print(len(graphs))
    graph, last_action_node, last_node, _ = graphs
    out = model(graph, last_action_node, last_node)
    print([out[i] == out[i + 1] for i in range(9)])


actor = Actor(54, 250, 10, 0, "ReLU", "ReLU", "GatedGraphConv", 3, 3, "SpectralNorm")
critic = Critic(54, 200, "GatedGraphConv", 3, 3, "ReLU", "ReLU", 0, "basis")
# critic_bl = CriticSqueeze(54, 300)
# model_test(actor)

s = {"aplpelfa": 123, "fravs": 134, "frava": 666}
