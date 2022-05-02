import sys
from matplotlib import pyplot as plt
sys.path.append('..')
import yaml, copy
from rdkit import Chem
import dgl
import networkx as nx
from enviroment.ChemEnv import DefaultEnv
from Architectures.models import Actor
from TrainingMain.config_utils import generateActor
from enviroment.utils import MolFromGraphsFULL, fixOrder, mol_to_graph_full
from dgllife.utils import complex_to_graph
from inference_utils import Handler

import torch


class Profiler(Handler):
    def __init__(self, path, load_model=True):
        super().__init__(path, load_model)
        self.curr_idx = 0
    
    
    def molSkill(self,mol_fragment, width) -> float:
        """calculates perecentage of proposed actions being legitimate for a given mol frag 

        Args:
            mol_fragment (Chem.RWMol): molecular fragment to check against
            width (int): how many actions to proposed

        Returns:
            float: accuracy
        """
        
        # self.env.assignMol(mol_fragment)
        
        actions = self._get_n_best(mol_fragment, width)
        
        successful = 0
        if actions.shape != (1,1):
            actions = torch.squeeze(actions)
        for action in actions:
            action_int = int(action)
            mol_copy = copy.deepcopy(mol_fragment)
            self.env.assignMol(mol_copy)
            
            _,_,done,reward_dict = self.env.step(action_int)
            
            if reward_dict['step_reward'] > 0 or done:
                successful += 1
        
        acc = successful/width 
        if acc != 1:
            print(actions,Chem.MolToSmiles(mol_fragment),acc)   
        return  acc
    
    def multiStepMolSkill(self,mol, length):       
        """method for checking base performance in the environment

        Args:
            mol (Chem.Mol): Starting Strecture
            length (int): how far out to check
            

        Returns:
            float: accuracy
        """
        acc = 0
        
        return acc
    
    def runProfile(self,number,width=2):
        #would be nice to have a graph over tot_accuracy with x axis being width
        tot_accuracy = 0
        seen = 0
        for _ in range(number):
            self.env.reset()
            self.curr_idx = self.env.idx
            # self.env.assignMol(Chem.MolFromSmiles('N=C1CCCC12CCCCC2.O'))
            mol = self.env.StateSpace
            g = mol_to_graph_full(mol).to_networkx().to_undirected()
            
            num_connected = nx.number_connected_components(g)
            if num_connected < 3 and mol.GetNumAtoms()>2:
                mol = self.env.StateSpace
                accuracy = self.molSkill(mol,width)
                # if accuracy != 1:
                #     print(self.curr_idx)
                tot_accuracy += accuracy
                seen += 1
            # if accuracy != 1:
            #     print(Chem.MolToSmiles(self.env.StateSpace), accuracy)
        tot_accuracy /= seen
        
        return tot_accuracy   
        
            
def checkIdx(index: int):
    graphs, graph_dict = dgl.load_graphs('../GraphDecomp/graph_decomp/graphs',[index])
    action = graph_dict['actions'][index]
    mol = MolFromGraphsFULL(graphs[0])
    print(action)
    return mol
            
graphs, graph_dict = dgl.load_graphs('../GraphDecomp/graph_decomp/graphs',[1]) 
action = graph_dict['actions']
hist = []

# for i in range(1000):
#     if i%100 == 0:
#         print(i)
#     graphs, graph_dict = dgl.load_graphs('../GraphDecomp/graph_decomp/graphs',[i]) 
#     # if graphs[0].number_of_nodes()== 2:
#     #     print(graph_dict['actions'][i])
    
#     hist.append(int(action[i]))
    
# plt.hist(hist, bins=list(range(0,34)))
# plt.savefig('distributions')
    
# # # # # # mol = Chem.MolFromSmiles("c1ccccc1") 



if __name__ == '__main__':
    profiler = Profiler('../TrainingMain/Weight13',load_model=True)     
    l = []
    s = profiler.molSkill(fixOrder(Chem.MolFromSmiles('CC.C')),2)
    print(s)
    for i in range(1,2):
        acc_profile = profiler.runProfile(250,i)
        print(acc_profile)
        l.append(acc_profile)
        print(f'completed run through {i}')
        
    print(l)
    # plt.plot(list(range(1,len(l)+1)),l)
    # plt.title('Pretrain Skill')
    # plt.xlabel('Number of Considered Actions')
    # plt.ylabel('Accuracy')
    # plt.savefig('Skill_1')
