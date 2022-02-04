
from models import init_weights_recursive


import torch
from supervised_training.sv_utils import SupervisedTrainingWrapper
from torch.utils.tensorboard import SummaryWriter
from PPO import PPOTrainer
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    


class Trainer():
    def __init__(self,run_title: str, writer: SummaryWriter, sv_trainer: SupervisedTrainingWrapper, PPO: PPOTrainer):
        """class init

        Args:
            run_title (str): name of run
            writer (SummaryWriter): tensorboard writer
            sv_trainer (SupervisedTrainingWrapper): supervised training module
            PPO (PPO_MAIN): ppo module
        """
        
        self.run_title = run_title
        self.writer = writer

        self.policy = sv_trainer.model
        self.policy.apply(init_weights_recursive)
        
        
        self.PPO = PPO
        self.PPO.to_device()

        self.sv_trainer = sv_trainer

    def Train(self,sv_epochs: int, PPO_steps: int):
        
        self.sv_trainer.Train(sv_epochs)
    
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.svw.optim.state_dict()
            }, f'./{self.run_title}/supervised_model')
        
        self.PPO.learn(PPO_steps)

        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.svw.optim.state_dict()
            }, f'./{self.run_title}/fine_tuned_model')
        

# class TrainWrapper():
#     def __init__(self, )