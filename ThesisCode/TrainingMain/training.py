import subprocess
import os
import sys
sys.path.append('..')
# from models import dropout_inference
from Architectures.models import dropout_inference

import torch
# from supervised_training.sv_utils import SupervisedTrainingWrapper
from TrainingUtils.SupervisedTraining.sv_utils import SupervisedTrainingWrapper
from TrainingUtils.PPO import PPOTrainer
from torch.utils.tensorboard import SummaryWriter
# from PPO import PPOTrainer
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        run_title: str,
        writer: SummaryWriter,
        sv_trainer: SupervisedTrainingWrapper,
        PPO: PPOTrainer,
        sv_epochs: int,
        PPO_steps: int,
    ):
        """class init

        Args:
            run_title (str): name of run
            writer (SummaryWriter): tensorboard writer
            sv_trainer (SupervisedTrainingWrapper): supervised training module
            PPO (PPO_MAIN): ppo module
        """
        # if restart_path:
        #     sv_trainer.model.load_state_dict(torch.load(os.path.join(restart_path,'supervised_model'))['model_state_dict'])


        self.run_title = run_title
        self.writer = writer

        self.policy = sv_trainer.model

        self.PPO = PPO
        self.PPO.to_device()

        self.sv_trainer = sv_trainer

        self.sv_epochs = sv_epochs
        self.PPO_steps = PPO_steps

    def Train(self):
        if not os.path.isdir(f"./{self.run_title}"):
            os.mkdir(f"./{self.run_title}")
        subprocess.run(['cp', './config_small.yaml', f"./{self.run_title}/config.yaml" ])

        print(f"running supervised training for {self.sv_epochs} epochs")
        self.sv_trainer.Train(self.sv_epochs)
        print(f"saving model")

        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.sv_trainer.optim.state_dict(),
            },
            f"./{self.run_title}/supervised_model",
        )

        print(f"running RL for {self.PPO_steps} steps")

        self.PPO.to_device()

        self.policy.apply(dropout_inference)  # turn off dropout
        self.PPO.learn(self.PPO_steps)

        print("saving model")
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.PPO.actor_optim.state_dict(),
            },
            f"./{self.run_title}/fine_tuned_model",
        )

        print("complete")

