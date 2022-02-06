from abc import ABC, abstractmethod
# from autoDock import VinaWrapper
from rdkit import Chem
import matplotlib.pyplot as plt

class SingleReward(ABC):
    @abstractmethod
    def __init__(self):
        self.reward_list = []
        
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
        self.reward_list.append(reward)
        return reward
        
    
    
# class DockReward(SingleReward):
#     def __init__(self,receptor_path):
#         self.reward_list = []
#         self.vinaWrapper = VinaWrapper(receptor_path)
#         super(DockReward,self)
        
#     def name(self):
#         return "DockReward"
    
#     def rescale(self,value):
#         return (-1*value) / 4
    
#     def _reward(self,mol):
#         smile = Chem.MolToSmile(mol)
#         return self.rescale(self.vinaWrapper.CalculateEnergies())
    
    
class SizeReward(SingleReward):
    def __init__(self):
        self.reward_list = []
        
    def name(self):
        return "SizeReward"
    
    def rescale(self,value):
        return value
    
    def _reward(self,mol):
        return(len(mol.GetAtoms()))
    
    
class FinalRewardModule():
    '''reward function handling'''
    def __init__(self,writer,r_list):
        self.r_list = r_list
        self.writer = writer
        self.color = ['blue','orange','red','green','yellow']
        self.n_iter = 0
    
    def UpdateTrainingModules(self):
        pass

    def PlotRewards(self):
        for idx,SingleReward in enumerate(self.r_list):
            plt.plot(SingleReward.reward_list, label = SingleReward.name(),color = self.color[idx])
        plt.show()
        
        
    def GiveReward(self,mol):
        self.n_iter += 1
        mol.UpdatePropertyCache()
        rewards = 0
        for rewardObject in self.r_list:
            reward = rewardObject.giveReward(mol)
            self.writer.add_scalar(rewardObject.name(), reward, self.n_iter)
            rewards += reward
        return rewards
            