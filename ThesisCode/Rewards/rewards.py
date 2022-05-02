from abc import ABC, abstractmethod
import heapq
import numpy as np
from .autoDock import VinaWrapper
# from autoDock import VinaWrapperi
import rdkit
from rdkit import Chem
from rdkit.Chem import Lipinski, Draw
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import qed
import matplotlib.pyplot as plt
import wandb


import torch
import math


if __name__ == '__main__':
    from SA_Score import sascorer
else:
    from .SA_Score import sascorer



# lss wandb

class SingleReward(ABC):
    @abstractmethod
    def __init__(self, Wandb = True):
        self.reward_list = []
        self.Wandb = Wandb
        if Wandb:
            wandb.define_metric(self.name(), step_metric="num_mols")
    @abstractmethod
    def _reward(self,mol):
        pass
    
    @abstractmethod
    def name(self):
        pass
    
    def rescale(self, x):
        return x
    
    def giveReward(self, x):
        reward = self.rescale(self._reward(x))
        if self.Wandb:
            wandb.log({self.name(): reward})
        self.reward_list.append(reward)
        return reward


class SynthReward(SingleReward):
    def __init__(self, Wandb = True):
        super(SynthReward,self).__init__(Wandb)
        try:
            sascorer.calculateScore(Chem.MolFromSmiles("CC"))
            print('Synth Reward Succesfully initialized')
        except:
            print('synth fail')
            raise 

    def name(self):
        return 'SynthReward'
    
    def _reward(self,mol: Chem.rdchem.Mol):
        # print(Chem.MolToSmiles(mol))
        synth = sascorer.calculateScore(mol) *  np.tanh(((mol.GetNumAtoms()-12) / 15))
        synth -= 4
        synth /= 2
        synth = np.clip(synth,a_min=-2,a_max=2)
        return synth
    
        
    
    
class DockReward(SingleReward):
    def __init__(self,receptor_path, Wandb = True):
        super(DockReward,self).__init__(Wandb)
        self.vinaWrapper = VinaWrapper(receptor_path)
        
    def name(self):
        return "DockReward"
    
    def rescale(self,value):
        if value > 2:
            return -2
        reward = (- value) - 5
        return np.clip(reward, a_min=-2,a_max=10)
    
    def _reward(self,mol):
        smile = Chem.MolToSmiles(mol)
        return self.vinaWrapper.CalculateEnergies(smile)
    
    
class SizeReward(SingleReward):
    def __init__(self, Wandb = True):
        super(SizeReward,self).__init__(Wandb)
        
    def name(self):
        return "SizeReward"
    
    def rescale(self,value):
        # norm = (value - 11)/5
        # return 3 * math.tanh(norm)
        return value * 3
    def _reward(self,mol):
        
        # return(len(mol.GetAtoms()))
        return ((20 < mol.GetNumAtoms() < 25)-.5)
        
class LogPReward(SingleReward):
    def __init__(self, Wandb = True):
        super(LogPReward,self).__init__(Wandb)
        
    def name(self):
        return 'LogPReward'
    
    def rescale(self, x):
        return x * 2

    def _reward(self, mol):
        return (MolLogP(mol) > 3.5) - .5

    
class LipinskiReward(SingleReward):
    def __init__(self, Wandb = True):
        super(LipinskiReward,self).__init__(Wandb)
        
    def name(self, Wandb = True):
        return 'LipinskiReward'

    def rescale(self, x):
        return x*2
    
    def _reward(self, mol):
        numHdonors = Lipinski.NumHDonors(mol)
        numHacceptors = Lipinski.NumHAcceptors(mol)
        
        return 2*(int(numHdonors < 5 and numHacceptors < 10)-.5)

class QEDReward(SingleReward):
    def __init__(self, Wandb=True):
        super(QEDReward,self).__init__(Wandb)
        
    def name(self):
        return 'QED Reward'
    
    def rescale(self, x):
        return (x-.8)*3 +2
    
    def _reward(self, mol):
        try: 
            return qed(mol)
        except:
            return 0

class MolBuffer():
    """Priority Queue for keeping track of best molecules seen so far
    """

    def __init__(self, length: int):
        """_summary_

        Args:
            length (int): number of molecules to hold
        """
        self.length = length
        self.__buffer = []
        self.step = 1

        # wandb.define_metric("best_mols",step_metric="best_mols_step")

    def put(self, item):
        """Put (score,smile) into buffer. Will not replace anything if its bad

        Args:
            item (tuple): score, smile tuple
        """
        if len(self.__buffer) < self.length:
            heapq.heappush(self.__buffer, item)
        else:
            heapq.heappushpop(self.__buffer, item)

    def log(self, top: int):
        """log top molecules to wandb run

        Args:
            top (int): how many molecules to log
        """
        top_n = heapq.nlargest(top,self.__buffer, lambda x: x[0])
        for item in top_n:
            try:
                molecule = Chem.MolFromSmiles(item[1])
                pil_image = Draw.MolToImage(molecule, size=(300, 300))            
                wandb.log({'best_mols': wandb.Image(pil_image)})
            except:
                pass
            

    def reset(self):
        self.__buffer = []  
    
class FinalRewardModule():
    '''reward function handling'''
    def __init__(self,writer,r_list,wandb_log = True, scaling=False):
        self.r_list = r_list
        self.writer = writer
        self.buffer = MolBuffer(1)
        self.wandb_log = wandb_log

        self.color = ['blue','orange','red','green','yellow']
        
        self.n_iter = 0
        self.scaling = scaling
        self.gamma = .99
        self.mean_discount = .95
        self.mean = 0
        self.first = True
        """Maybe put a buffer of rewards and use that to calculate std
        """
        
        if scaling:
           self.r_stats = [0]
    
    def UpdateTrainingModules(self):
        pass
    
    def PlotRewards(self):
        for idx,SingleReward in enumerate(self.r_list):
            plt.plot(SingleReward.reward_list, label = SingleReward.name(),color = self.color[idx])
        plt.show()
        
    def LogRewards(self):
        self.buffer.log(5)
        self.buffer.reset()
          
    def GiveReward(self,mol):
        if self.wandb_log:
            wandb.log({'num_mols':self.n_iter})
        self.n_iter += 1
        mol.UpdatePropertyCache()
        rewards = 0
        for rewardObject in self.r_list:
            reward = rewardObject.giveReward(mol)
            # self.writer.add_scalar(rewardObject.name(), reward, self.n_iter)
            rewards += reward

        self.buffer.put((rewards,Chem.MolToSmiles(mol)))
        
        if self.scaling:         
            if self.first:
                self.mean = 0
                self.first = False
                
            else:
                self.mean = rewards*(1-self.mean_discount) + (self.mean* self.mean_discount)
            
            rewards_centered = (rewards-self.mean)#/(np.std(self.r_stats) + 1e-10)
            if self.wandb_log:
                wandb.log({'running norm rewards': rewards_centered})
            return rewards_centered
        if self.wandb_log:
            wandb.log({'total rewards': rewards})
        return rewards
            
            
            
# mol = Chem.MolFromSmiles('Cc1c2c3c(c(NPCl)c1PC2)CN=N3')
# sr = SynthReward(False)
# print(sr.giveReward(mol))