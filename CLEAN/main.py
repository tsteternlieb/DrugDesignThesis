from PPO import PPOTrainer
from enviroment.ChemEnv import ChemEnv
from supervised_training.sv_utils import SupervisedTrainingWrapper
from training import Trainer
from torch.utils.tensorboard import SummaryWriter
from enviroment.utils import mol_to_graph_full
from models import BaseLine
from Rewards.rewards import SizeReward, FinalRewardModule
import torch



#CUDA
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#LOGGING
RUN_TITLE = 'TEST_1'
WRITER = SummaryWriter(f'./tb_logs/{RUN_TITLE}_logs')


#ENVIROMENT VARIABLES
NUM_NODE_FEATS = 54
MOL_FEATURIZER = mol_to_graph_full
REWARD_MODULE = FinalRewardModule(WRITER, [SizeReward()])

#MODEL VARIABLES
HIDDEN_DIM = 300
NUM_ATOM_TYPES = 17




#PPO VARIABLES
PPO_BATCH_SIZE = 32
TIMESTEPS_PER_ITERATION = 3000
CLIP = 0.1
A_LR = 1e-4
C_LR = 3e-4
NUM_UPDATED_PER_ITERATION = 6
MAX_TIMESTEPS_PER_EPISODE = 40
GAMMA = .95

#SUPERVISED TRAINING VARIABLES
##MODEL ALREADY DEFINED
##WRITER ALREADY DEFINED
SUPERVISED_BATCH_SIZE = 128
DATASET_SIZE = 507528




#DEFININING MODULES
ENV = ChemEnv(NUM_NODE_FEATS,REWARD_MODULE,MOL_FEATURIZER,WRITER)
MODEL = BaseLine(NUM_NODE_FEATS,HIDDEN_DIM,NUM_ATOM_TYPES)

PPO = PPOTrainer(ENV,PPO_BATCH_SIZE,TIMESTEPS_PER_ITERATION,
               CLIP,A_LR,C_LR,NUM_UPDATED_PER_ITERATION,
               MAX_TIMESTEPS_PER_EPISODE,GAMMA,MODEL,WRITER)

SUPERVISED_TRAINER = SupervisedTrainingWrapper(MODEL,SUPERVISED_BATCH_SIZE,DATASET_SIZE,WRITER)


#DEFINING TRAINING
TRAINING_SESSION = Trainer(RUN_TITLE,WRITER,SUPERVISED_TRAINER,PPO)



# ppo_test = PPO_MAIN(**PPO_kwargs)

# ppo_test.learn(100000)