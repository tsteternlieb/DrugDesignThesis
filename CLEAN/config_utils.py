from Rewards.rewards import SizeReward, FinalRewardModule
from enviroment.utils import mol_to_graph_full

def generateRewardModule(reward_names):
    rewards = []
    if 'SIZE' in reward_names:
        rewards.append(SizeReward())

    ### more
    return rewards

def generateMolFeaturizer(featurizer_name):
    if featurizer_name == 'mol_to_graph_full':
        return mol_to_graph_full

# def modelGenerator()