import torch
import torch.nn as nn

from enviroment.ChemEnv import ChemEnv

class Handler():
    """Class for handling model. inference and that sort of stuff"""
    def __init__(self, path: str, model: nn.Module, env: ChemEnv):
        """create handler insance

        Args:
            path (str): path to saved model
            model (nn.Module): model for params to be loaded into
            env ([type]): Chem environment
        """
        self.model = model
        self.model.load(path)
        
        self.env = env

    def __get_n_best(self,n):
        obs = self.env.getObs()
        predictions = self.model(obs)
        top_actions = torch.topk(predictions, n)
        
        mol = obs[0]
        
        for action in top_actions:
            pass
            
    
    def treeSearch(self,width,height):
        for depth in range(height):
            pass
        

    def inference():
        pass