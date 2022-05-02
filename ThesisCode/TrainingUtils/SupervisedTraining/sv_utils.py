import sys
sys.path.append('../..')
import time
import random
import dgl
import torch
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from rdkit import Chem
from torch.utils.tensorboard import SummaryWriter
import wandb
from enviroment.ChemEnv import DefaultEnv
import matplotlib.pyplot as plt
device = None


class GraphDataLoader2():
    def __init__(self, path, batch_size, split=.8):
        self.path = path
        self.length = dgl.load_graphs(path, [1])[1]['last_action'].shape[0]

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.split = split
        self.batch_size = batch_size
        # self.indicies = list(range(self.length))
        # random.shuffle(self.indicies)

        self.train_ids, self.test_ids = self.__get_split_indices()

        self.curr_idx = 0
        self.env = DefaultEnv()
        
    def __get_split_indices(self):
        """method for splitting dataset into training and test ids

        Returns:
            _type_: _description_
        """
        train_ids, test_ids = [], []

        for idx in range(self.length):
            if random.random() < self.split:
                train_ids.append(idx)
            else:
                test_ids.append(idx)

        return train_ids, test_ids

    def __get_exs(self, indices):
        graphs, graph_dict = dgl.load_graphs(self.path, indices)
        last_action = torch.stack(
            [graph_dict['last_action'][idx] for idx in indices], dim=0)
        last_atom_feat = torch.stack(
            [graph_dict['last_atom_feats'][idx] for idx in indices], dim=0)
        action = torch.stack([graph_dict['actions'][idx]
                              for idx in indices], dim=0)
        
        graphs = dgl.batch(graphs)
        return graphs.to(self.device), torch.unsqueeze(last_action, dim=1).to(self.device), last_atom_feat.to(self.device), action.to(self.device)

    def GetValidation(self, size):
        batches = []
        for i in range(0, len(self.test_ids)-size, size):
            chunk = self.test_ids[i:i+size]
            batches.append(self.__get_exs(chunk))

        return batches

    def __next__(self):
        if self.curr_idx + self.batch_size > len(self.train_ids):
            self.curr_idx = 0
            random.shuffle(self.train_ids)
        batch_indices = self.train_ids[self.curr_idx: self.curr_idx+self.batch_size]
        self.curr_idx += self.batch_size

        return self.__get_exs(batch_indices)


class GraphDataLoader():
    def __init__(self, path, batch_size, split=.7):
        """loader for graph traning

        Args:
            path (str): locaton of graph data
            batch_size (int): batch_size
            length (int): length of data set <- change
            split (float, optional): split for traing and test. Defaults to .7.
        """
        self.path = path
        self.length = dgl.load_graphs(path, [1])[1]['last_action'].shape[0]

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.indicies = list(range(self.length))
        random.shuffle(self.indicies)

        self.curr_idx = 0

    def _get_exs(self, indices):
        graphs, graph_dict = dgl.load_graphs(self.path, indices)
        last_action = torch.stack(
            [graph_dict['last_action'][idx] for idx in indices], dim=0)
        last_atom_feat = torch.stack(
            [graph_dict['last_atom_feats'][idx] for idx in indices], dim=0)
        action = torch.stack([graph_dict['actions'][idx]
                             for idx in indices], dim=0)
        graphs = dgl.batch(graphs)
        return graphs.to(self.device), torch.unsqueeze(last_action, dim=1).to(self.device), last_atom_feat.to(self.device), action.to(self.device)

    def __next__(self):
        if self.curr_idx + self.batch_size > self.length:
            self.curr_idx = 0
            random.shuffle(self.indicies)
        batch_indices = self.indicies[self.curr_idx: self.curr_idx+self.batch_size]
        self.curr_idx += self.batch_size

        return self._get_exs(batch_indices)

class LogDistribution:
    def __init__(self) -> None:
        self.env = DefaultEnv()
        
    def GetDistribution(self,model: nn.Module,mols):
        with torch.no_grad():
            for mol in mols:
                smiles = Chem.MolToSmiles(mol)

                self.env.assignMol(mol)
                obs = self.env.getObs()
                logits = model(*obs,mask = False) 
                logits = logits.cpu()[0]
                plt.bar( [i for i in range (logits.shape[0])], list(logits))
                plt.ylabel(smiles)
                wandb.log({f'{smiles} distribution': wandb.Image(plt)})
                plt.close()
# def GetDistribution(mol, model: nn.Module):
    
    
    

class SupervisedTrainingWrapper():
    def __init__(self, model: nn.Module, batch_size, learning_rate, lr_decay, writer, path='./graph_decomp/full_chunka'):
        """Supervised Trainier

        Args:
            model (torch.nn.Module): model
            batch_size (int): batch size
            data_set_size (int): data set size
            writer (writer): torch writer
            path (str, optional): training data path. Defaults to './graph_decomp/full_chunka'.
        """
        self.path = path
        self.model = model
        self.optim = Adam(self.model.parameters(),
                          lr=learning_rate, weight_decay=1e-5)
        self.schedule = torch.optim.lr_scheduler.ExponentialLR(
            self.optim, lr_decay)
        weight = torch.tensor([.1 for i in range(70)])
        weight = weight.cuda()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.chunk_idx = 0

        self.writer = writer
        self.n_iter = 0
        self.cv_iter = 0

        self.batch_size = batch_size
        self.dataIter = GraphDataLoader2(path, batch_size)
        
        self.DistributionLog = LogDistribution()

    def CalcAccuracy(self):
        a, b, c, d = next(self.dataIter)
        return self.calc_accuracy(a, b, c, d)

    def calc_accuracy(self, graphs, last_action_nodes, last_atom_feats, actions):
        y_logits = self.model(graphs, last_action_nodes, last_atom_feats)
        values, pred_labels = y_logits.max(dim=1)
        acc = sum(pred_labels == actions)/actions.size()[0]
        return acc

    def calc_accuracy_no_call(self, logits, actions):
        values, pred_labels = logits.max(dim=1)
        acc = sum(pred_labels == actions)/actions.size()[0]
        return acc

    def calc_accuracy_no_call(self, logits, actions):
        values, pred_labels = logits.max(dim=1)
        acc = sum(pred_labels == actions)/actions.size()[0]
        return acc

    def ValidationAccuracy(self, size=512):
        self.model.eval()
        batches = self.dataIter.GetValidation(size)
        acc = 0
        for batch in batches:
            acc += self.calc_accuracy(*batch)
        self.model.train()
        return acc/len(batches)

    def Train(self, num_epochs):
        steps_per_epoch = len(self.dataIter.train_ids)//self.batch_size

        for i in (range(num_epochs)):
            
            mols = [Chem.MolFromSmiles("CC.C"),Chem.MolFromSmiles("c1ccccc1")]
            self.DistributionLog.GetDistribution(self.model,mols)
            
            
            
            acc = self.ValidationAccuracy()
            wandb.log({'validation accuracy': acc})
            t0 = time.time()
            for step in tqdm(range(steps_per_epoch)):
                self._train()
            self._train()
            t1 = time.time()
            self.schedule.step()
            print(self.optim.param_groups[0]['lr'], 'step')
            print(
                f'Time for epoch {i} is {t1 - t0} , random accuracy is {acc}')

    def _train(self, calc_accuracy=True, update=True):
        graphs, last_action_nodes, last_atom_feats, actions = next(
            self.dataIter)

        pred = self.model.forward(
            graphs, last_action_nodes, last_atom_feats, softmax=False)

        acc = self.calc_accuracy_no_call(pred, actions)
        wandb.log({'train accuracy': acc})
        weights = torch.tensor([1 for _ in range(17)] + [.5,3,.5] + [1 for _ in range(pred.shape[1]-20)])
        
        weights = weights.cuda()
        lossfn = torch.nn.CrossEntropyLoss(weights)
        classification_loss = lossfn(pred, actions.long())
        
        node_additions = torch.softmax(pred[:,1:17], dim = 1)
        #numerical stability 
        entropy = torch.special.entr(node_additions+1e-4)
        entropy = torch.mean(torch.sum(entropy,dim=1))
    
        
        total_loss = classification_loss - (1.3* entropy)

        if update:
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        self.writer.add_scalar("pre_train_loss", classification_loss.detach(), self.n_iter)
        wandb.log({'classification loss': classification_loss.detach(),
                   'entropy loss' : -entropy})
        self.n_iter += 1
