from PPO import PPOTrainer
from enviroment.ChemEnv import ChemEnv
from supervised_training.sv_utils import SupervisedTrainingWrapper
from training import Trainer
from torch.utils.tensorboard import SummaryWriter
from models import BaseLine
from Rewards.rewards import FinalRewardModule
from config_utils import generateRewardModule, generateMolFeaturizer
import torch
import yaml

import argparse
import subprocess



def config_train(path):
    with open(path) as file:
        config = yaml.safe_load(file)

    # CUDA
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # LOGGING
    RUN_TITLE = config['RUN_TITLE']
    WRITER = SummaryWriter(f'./logs/{RUN_TITLE}_logs/tb')
    # subprocess.run(['mkdir', f'./logs/'])
    subprocess.run(['cp', path, f'./logs/{RUN_TITLE}_logs/config.yaml'])

    # ENVIROMENT VARIABLES
    NUM_NODE_FEATS = int(config['ENVIROMENT_VARIABLES']['NUM_NODE_FEATS'])
    MOL_FEATURIZER = generateMolFeaturizer(
        config['ENVIROMENT_VARIABLES']['MOL_FEATURIZER'])
    REWARD_MODULE = FinalRewardModule(WRITER, generateRewardModule(
        config['ENVIROMENT_VARIABLES']['REWARD_MODULES']))

    # MODEL VARIABLES
    HIDDEN_DIM = int(config['MODEL_VARIABLES']['HIDDEN_DIM'])
    NUM_ATOM_TYPES = int(config['MODEL_VARIABLES']['NUM_ATOM_TYPES'])

    # PPO VARIABLES
    PPO_BATCH_SIZE = int(config['PPO_VARIABLES']['PPO_BATCH_SIZE'])
    TIMESTEPS_PER_ITERATION = int(
        config['PPO_VARIABLES']['TIMESTEPS_PER_ITERATION'])
    CLIP = float(config['PPO_VARIABLES']['CLIP'])
    A_LR = float(config['PPO_VARIABLES']['A_LR'])
    C_LR = float(config['PPO_VARIABLES']['C_LR'])
    NUM_UPDATED_PER_ITERATION = int(
        config['PPO_VARIABLES']['NUM_UPDATED_PER_ITERATION'])
    MAX_TIMESTEPS_PER_EPISODE = int(
        config['PPO_VARIABLES']['MAX_TIMESTEPS_PER_EPISODE'])
    GAMMA = float(config['PPO_VARIABLES']['GAMMA'])

    # SUPERVISED TRAINING VARIABLES
    SUPERVISED_BATCH_SIZE = int(
        config['SUPERVISED_TRAINING_VARIABLES']['SUPERVISED_BATCH_SIZE'])
    DATASET_SIZE = int(config['SUPERVISED_TRAINING_VARIABLES']['DATASET_SIZE'])
    PATH = config['SUPERVISED_TRAINING_VARIABLES']['PATH']

    # DEFININING MODULES
    ENV = ChemEnv(NUM_NODE_FEATS, REWARD_MODULE, MOL_FEATURIZER, WRITER)
    MODEL = BaseLine(NUM_NODE_FEATS, HIDDEN_DIM, NUM_ATOM_TYPES).cuda()
    PPO = PPOTrainer(ENV, PPO_BATCH_SIZE, TIMESTEPS_PER_ITERATION,
                     CLIP, A_LR, C_LR, NUM_UPDATED_PER_ITERATION,
                     MAX_TIMESTEPS_PER_EPISODE, GAMMA, MODEL, WRITER)

    SUPERVISED_TRAINER = SupervisedTrainingWrapper(
        MODEL, SUPERVISED_BATCH_SIZE, DATASET_SIZE, WRITER, PATH)

    # DEFINING TRAINING
    TRAINING_SESSION = Trainer(RUN_TITLE, WRITER, SUPERVISED_TRAINER, PPO)
    SV_EPOCHS = int(config['FINAL_TRAINING_VARIABLES']['SV_EPOCHS'])
    PPO_STEPS = int(config['FINAL_TRAINING_VARIABLES']['PPO_STEPS'])

    # TRAINING_SESSION.Train(SV_EPOCHS,PPO_STEPS)



def main():
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config_train(args.config)

if __name__ == '__main__':
    main()
