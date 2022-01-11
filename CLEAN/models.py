
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import torch
import torch.nn as nn
import dgl
from torch.nn.utils.rnn import pad_sequence
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F

device = None

def init_weights_recursive(m):
    
    if isinstance(m,nn.Linear):
        m.apply(init_weights)
    elif isinstance(m,Linear_3):
        m.apply(init_weights)
    elif isinstance(m, Batch_Edge):
        m.NodeEmbed.apply(init_weights)
        m.Edges.apply(init_weights)
    elif isinstance(m, BaseLine):
        m.AddNode.apply(init_weights_recursive)
        m.BatchEdge.apply(init_weights_recursive)
        m.GGN.linears[0].apply(init_weights)
        m.GGN.linears[1].apply(init_weights)
        m.GGN.linears[2].apply(init_weights)
        m.GGN.linears[3].apply(init_weights)
        m.GGN.gru.apply(init_weights)

    elif isinstance(m,torch.nn.modules.container.ModuleList):
        m[0].apply(init_weights)
        m[1].apply(init_weights)
        m[2].apply(init_weights)
        m[3].apply(init_weights)
        
    elif isinstance(m,dgl.nn.pytorch.conv.gatedgraphconv.GatedGraphConv):
        m.apply(init_weights)
    elif isinstance(m,torch.nn.modules.rnn.GRUCell):
        m.apply(init_weights)
    elif isinstance(m,CriticSqueeze):
        m.GGN.apply(init_weights_recursive)
        m.Dense.apply(init_weights_recursive)
    else:
        print(m,type(m))
def init_weights(m): 
    try:
        nn.init.orthogonal_(m.weight.data)
    except:
        print('bad init')
        pass
    
    
class Linear_3(nn.Module):
    def __init__(self, in_dim, hidden_dim,out_dim):
        super(Linear_3, self).__init__()
        self.Dense1 = nn.Linear(in_dim,hidden_dim)
        self.Dense2 = nn.Linear(hidden_dim,hidden_dim)
        self.Dense3 = nn.Linear(hidden_dim,out_dim)
        
    def forward(self, inputs):
        out = (torch.tanh(self.Dense1(inputs)))
        out = (torch.tanh(self.Dense2(out)))
        out = self.Dense3(out)
        return out
    
class CriticSqueeze(nn.Module):
    
    '''
    Critic class for advantage estimation 
    '''
    def __init__(self, in_dim, hidden_dim):
        super(CriticSqueeze, self).__init__()
        self.GGN = dgl.nn.GatedGraphConv(in_dim,hidden_dim,5,4)
        self.Dense = Linear_3(hidden_dim+1+in_dim,hidden_dim,1)
    def forward(self, graph ,last_action_node,last_node):
        #print(graph.adj())
        h = F.relu(self.GGN(graph, graph.ndata['atomic'], graph.edata['type'].squeeze()))
        with graph.local_scope():
            graph.ndata['h'] = h
            hg = dgl.mean_nodes(graph, 'h')
        cat = torch.cat((hg,last_action_node,last_node),dim = 1)
        out = self.Dense(cat)
        return out

class Batch_Edge(nn.Module):
    
    def __init__(self, in_dim, hidden_dim):
        super(Batch_Edge, self).__init__()
        self.NodeEmbed = nn.Linear(in_dim, hidden_dim)
        self.Edges = Linear_3(hidden_dim*2,hidden_dim*2,2)
        
    def forward(self, graphs, last_node_batch, h):
        '''
        graph is actually a batch of graphs where len(graph.batch_num_nodes()) = len(last_node_batch)
        returns just a list of edges so its unsplit up rn, bbut     
        '''
        edges_per_graph = []
        
        node_embed_batch = self.NodeEmbed(last_node_batch)
        batch_node_stacks = [] #holds number of graphs of each node embedding stacked to match node number per graph
                               #so its first dimension is the batch size, and the second 'dimension' is the number of nodes per grah
        
        
        '''looping over graphs'''
        for i in range(last_node_batch.shape[0]):
            batch_node_stacks.append(node_embed_batch[i].repeat(graphs.batch_num_nodes()[i],1))
        
                                    
        batch_node_stacks = torch.cat(batch_node_stacks, dim = 0)
        
        stack = torch.cat((h,batch_node_stacks),dim = 1)
        edges = self.Edges(stack)
        with graphs.local_scope():
            graphs.ndata['bond_pred'] = edges
            graphs = dgl.unbatch(graphs)
            for graph in graphs:
                edges_per_graph.append(graph.ndata['bond_pred'])
                
        return pad_sequence(edges_per_graph, batch_first = True, padding_value=-10000).flatten(1,-1)

class BaseLine(nn.Module):
    '''Crappy base line to check improvements against'''
    def __init__(self,in_dim,hidden_dim, num_nodes):
        super(BaseLine, self).__init__()
        self.in_dim = in_dim
        self.num_nodes = num_nodes
        self.GGN = dgl.nn.GatedGraphConv(in_dim, hidden_dim,5,4)
        self.AddNode = Linear_3(hidden_dim,hidden_dim,num_nodes)
        self.BatchEdge = Batch_Edge(in_dim,hidden_dim)
    
    def forward(self,graph,last_action_node, last_node, mask = False, softmax = True):
        out = []
        h = F.relu(self.GGN(graph,graph.ndata['atomic'],graph.edata['type'].squeeze()))
        with graph.local_scope():
            graph.ndata['h'] = h
            hg = dgl.mean_nodes(graph, 'h') 
            
        
        addNode = self.AddNode(hg)

        if mask:
            node_mask = torch.unsqueeze(torch.cat((torch.zeros(1), torch.ones(self.num_nodes-1))),dim=0).to(device)
            mask = last_action_node*(node_mask*-100000)
            addNode += mask
        
        
        edges = self.BatchEdge(graph,last_node,h)     
        out = torch.cat((addNode,edges), dim = 1)
        if softmax:
            return torch.softmax(out,dim = 1)
        else:
            return out
        