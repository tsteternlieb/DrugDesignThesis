import sys
sys.path.append('..')

import torch
from torch.utils.tensorboard import SummaryWriter


from TrainingUtils.PPO import PPOTrainer


from enviroment.ChemEnv import ChemEnv
# from Train.sv_utils import SupervisedTrainingWrapper
from TrainingUtils.SupervisedTraining.sv_utils import SupervisedTrainingWrapper

from training import Trainer
from Architectures.models import Actor, Critic

from Rewards.rewards import FinalRewardModule
from config_utils import generateRewardModule, generateMolFeaturizer, generateActor, generateCritic
import wandb


def make(config):
    """Generates object to handle training

    Args:
        path (str): path to config file

    Returns:
        (Trainer,dict): training session and config dict
    """
    # with open(path) as file:
    #     config = yaml.safe_load(file)

    # CUDA
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # LOGGING
    RUN_TITLE = config['RUN_TITLE']
    WRITER = SummaryWriter(f'./logs/{RUN_TITLE}_logs/tb')
    # subprocess.run(['mkdir', f'./logs/'])
    # subprocess.run(['cp', path, f'./logs/{RUN_TITLE}_logs/config.yaml'])

    if config['ACTOR_RESTART_PATH'] != 'None':
        ACTOR_RESTART_PATH = config['ACTOR_RESTART_PATH']
        print(f'Using saved actor at location: {config["ACTOR_RESTART_PATH"][0]}')
    else:
        ACTOR_RESTART_PATH = None
        
    if config['CRITIC_RESTART_PATH'] != 'None':
        CRITIC_RESTART_PATH = config['CRITIC_RESTART_PATH']
        print(f'Using saved critic at location: {config["CRITIC_RESTART_PATH"][0]}')
    else:
        CRITIC_RESTART_PATH = None
    

    # ENVIROMENT VARIABLES
    NUM_NODE_FEATS = int(config['ENVIROMENT_VARIABLES']['NUM_NODE_FEATS'])
    MOL_FEATURIZER = generateMolFeaturizer(
        config['ENVIROMENT_VARIABLES']['MOL_FEATURIZER'])
    REWARD_MODULE = FinalRewardModule(WRITER, generateRewardModule(
        config['ENVIROMENT_VARIABLES']['REWARD_MODULES']))

    # ACTOR HYPER PARAMATERS

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

    SUPERVISED_LEARNING_RATE = float(
        config['SUPERVISED_TRAINING_VARIABLES']['LR'])
    SUPERVISED_LR_DECAY = float(
        config['SUPERVISED_TRAINING_VARIABLES']['DECAY'])
    DATASET_SIZE = int(config['SUPERVISED_TRAINING_VARIABLES']['DATASET_SIZE'])
    PATH = config['SUPERVISED_TRAINING_VARIABLES']['PATH']

    # DEFININING MODULES
    ENV = ChemEnv(NUM_NODE_FEATS, REWARD_MODULE, MOL_FEATURIZER, WRITER)
    
    ACTOR = generateActor(config['ACTOR_VARIABLES'], ACTOR_RESTART_PATH).cuda()
    CRITIC = generateCritic(config['CRITIC_VARIABLES'], CRITIC_RESTART_PATH).cuda()
    

    PPO = PPOTrainer(ENV, PPO_BATCH_SIZE, TIMESTEPS_PER_ITERATION,
                     CLIP, A_LR, C_LR, NUM_UPDATED_PER_ITERATION,
                     MAX_TIMESTEPS_PER_EPISODE, GAMMA, ACTOR, CRITIC,WRITER, RUN_TITLE)

    SUPERVISED_TRAINER = SupervisedTrainingWrapper(
        ACTOR, SUPERVISED_BATCH_SIZE, SUPERVISED_LEARNING_RATE, SUPERVISED_LR_DECAY, WRITER, PATH)

    # DEFINING TRAINING
    SV_EPOCHS = int(config['FINAL_TRAINING_VARIABLES']['SV_EPOCHS'])
    PPO_STEPS = int(config['FINAL_TRAINING_VARIABLES']['PPO_STEPS'])
    TRAINING_SESSION = Trainer(
        RUN_TITLE, WRITER, SUPERVISED_TRAINER, PPO, SV_EPOCHS, PPO_STEPS)

    # TRAINING_SESSION.Train(SV_EPOCHS,PPO_STEPS)
    return TRAINING_SESSION, config


def make2(config):
    """Generates object to handle training

    Args:
        path (str): path to config file

    Returns:
        (Trainer,dict): training session and config dict
    """
    # with open(path) as file:
    #     config = yaml.safe_load(file)

    # CUDA
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(wandb.config)
    # LOGGING
    RUN_TITLE = "SWEEP"
    WRITER = SummaryWriter(f'./logs/{RUN_TITLE}_logs/tb')
    # subprocess.run(['mkdir', f'./logs/'])
    # subprocess.run(['cp', path, f'./logs/{RUN_TITLE}_logs/config.yaml'])

    # ENVIROMENT VARIABLES

    NUM_NODE_FEATS = int(config.NUM_NODE_FEATS)
    MOL_FEATURIZER = generateMolFeaturizer(
        config.MOL_FEATURIZER)
    REWARD_MODULE = FinalRewardModule(WRITER, generateRewardModule(
        config.REWARD_MODULES))

    # ACTOR HYPER PARAMATERS
    actor_kwargs = {
        'in_dim': config.NUM_NODE_FEATS,
        'hidden_dim': int(config.HIDDEN_DIM_A),
        'out_dim': int(config.NUM_NODE_FEATS),
        'drop_out': config.DROPOUT_A,
        'graph_activation': config.GRAPH_ACTIVATION_A,
        'dense_activation': config.DENSE_ACTIVATION_A,
        'model': config.MODEL_TYPE_A,
        'graph_num_layers': config.GRAPH_NUM_LAYERS_A,
        'dense_num_layers': config.DENSE_NUM_LAYERS_A,
        'norm': config.DENSE_NORMALIZATION_A

    }

    # CRITIC HYPER PARAMATERS
    critic_kwargs = {
        'in_dim': config.NUM_NODE_FEATS,
        'hidden_dim': config.HIDDEN_DIM_C,
        'model': config.MODEL_TYPE_C,
        'graph_num_layers': config.GRAPH_NUM_LAYERS_C,
        'dense_num_layers': config.DENSE_NUM_LAYERS_C,
        'graph_activation': config.GRAPH_ACTIVATION_C,
        'dense_activation': config.DENSE_ACTIVATION_C,
        'dropout': config.DROPOUT_C,
        'norm': config.DENSE_NORMALIZATION_C
    }

    # PPO VARIABLES
    PPO_BATCH_SIZE = config.PPO_BATCH_SIZE
    TIMESTEPS_PER_ITERATION = config.TIMESTEPS_PER_ITERATION
    CLIP = config.CLIP
    A_LR = config.A_LR
    C_LR = config.C_LR
    NUM_UPDATED_PER_ITERATION = config.NUM_UPDATED_PER_ITERATION
    MAX_TIMESTEPS_PER_EPISODE = config.MAX_TIMESTEPS_PER_EPISODE
    GAMMA = config.GAMMA

    # SUPERVISED TRAINING VARIABLES
    SUPERVISED_BATCH_SIZE = config.SUPERVISED_BATCH_SIZE

    SUPERVISED_LEARNING_RATE = config.SUPERVSED_LR
    SUPERVISED_LR_DECAY = config.SUPERVISED_LR_DECAY
    PATH = config.PATH

    # DEFININING MODULES
    ENV = ChemEnv(NUM_NODE_FEATS, REWARD_MODULE, MOL_FEATURIZER, WRITER)
    # MODEL = BaseLine(NUM_NODE_FEATS, HIDDEN_DIM, NUM_ATOM_TYPES).cuda()
    ACTOR = Actor(**actor_kwargs)
    CRITIC = Critic(**critic_kwargs)

    PPO = PPOTrainer(ENV, PPO_BATCH_SIZE, TIMESTEPS_PER_ITERATION,
                     CLIP, A_LR, C_LR, NUM_UPDATED_PER_ITERATION,
                     MAX_TIMESTEPS_PER_EPISODE, GAMMA, ACTOR, CRITIC, WRITER)

    SUPERVISED_TRAINER = SupervisedTrainingWrapper(
        ACTOR, SUPERVISED_BATCH_SIZE, SUPERVISED_LEARNING_RATE, SUPERVISED_LR_DECAY, WRITER, PATH)

    # DEFINING TRAINING
    SV_EPOCHS = int(config.SV_EPOCHS)
    PPO_STEPS = int(config.PPO_STEPS)
    TRAINING_SESSION = Trainer(
        RUN_TITLE, WRITER, SUPERVISED_TRAINER, PPO, SV_EPOCHS, PPO_STEPS)

    # TRAINING_SESSION.Train(SV_EPOCHS,PPO_STEPS)
    return TRAINING_SESSION
