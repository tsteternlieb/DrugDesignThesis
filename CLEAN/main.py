from PPO import PPO_MAIN
from enviroment.ChemEnv import ChemEnv
from torch.utils.tensorboard import SummaryWriter
from enviroment.Utils import mol_to_graph_full
from models import BaseLine

env_kwargs = chem_env_kwargs = {'max_nodes' : 12, 
                   'num_atom_types' : 17, 
                   'num_node_feats' : 54,
                   'num_edge_types' : 3, 
                   'bond_padding' : 12, 
                   'mol_featurizer': mol_to_graph_full, 
                   'RewardModule' : None, 
                   'writer' : None,
                   'num_chunks': 0}

env = ChemEnv(**env_kwargs)

PPO_kwargs = {'env' : env,
              'batch_size' : 32,
              'timesteps_per_batch' : 1200,
              'clip' : 0.08,
              'a_lr' : 1e-4,
              'c_lr' : 3e-4,
              'n_updates_per_iteration' : 6,
              'max_timesteps_per_episode' : 40,
              'gamma' : .95,
              'actor' : BaseLine(54,20,17),
              'writer': SummaryWriter(f'./tb_logs/3/3_logs')}
ppo_test = PPO_MAIN(**PPO_kwargs)

ppo_test.learn(1000)