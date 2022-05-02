import os
from pdb import Restart
import yaml
from Rewards.rewards import (
    DockReward,
    SizeReward,
    SynthReward,
    LipinskiReward,
    LogPReward,
    QEDReward,
    SingleReward,
)
from enviroment.utils import mol_to_graph_full
import torch

from Architectures.models import Actor, Critic


def generateRewardModule(reward_names: "list[str]", Wandb=True):
    """tool for generating a list of rewards from list of names

    Args:
        reward_names (list[str]): list of names from SIZE, SYNTH, LIPINSKI, QED, LogP, DOCK

    Returns:
        _type_: list[SingleReward]
    """
    rewards = []
    if "SIZE" in reward_names:
        rewards.append(SizeReward(Wandb=Wandb))
        print("ADDING SIZE")
    if "SYNTH" in reward_names:
        rewards.append(SynthReward(Wandb=Wandb))
    if "LIPINSKI" in reward_names:
        print("adding Lipinski Reward")
        rewards.append(LipinskiReward(Wandb=Wandb))

    if "QED" in reward_names:
        print("adding QED Reward")
        rewards.append(QEDReward(Wandb=Wandb))

    if "LogP" in reward_names:
        print("adding LogP Reward")
        rewards.append(LogPReward(Wandb=Wandb))

    if "DOCK" in reward_names:
        rewards.append(DockReward("../Rewards/y220c_av.pdbqt", Wandb=Wandb))
        print("Adding Docking")

    return rewards


def generateMolFeaturizer(featurizer_name):
    if featurizer_name == "mol_to_graph_full":
        return mol_to_graph_full


def generateActor(vars, RestartPath=None):
    model_path = None
    if RestartPath != None and RestartPath != 'None':
        print(RestartPath,type(RestartPath))
        model_path = RestartPath[0]
        config_path = RestartPath[1]

        with open(config_path) as file:
            config = yaml.safe_load(file)["ACTOR_VARIABLES"]
        vars = config

    actor = Actor(
        54,
        vars["HIDDEN_DIM"],
        vars["NUM_ATOM_TYPES"],
        vars["DROPOUT"],
        vars["GRAPH_ACTIVATION"],
        vars["DENSE_ACTIVATION"],
        vars["MODEL_TYPE"],
        vars["GRAPH_NUM_LAYERS"],
        vars["DENSE_NUM_LAYERS"],
    )
    if model_path != None:
        actor.load_state_dict(torch.load(model_path)["model_state_dict"])

    return actor


def generateCritic(vars, restart_path=None):
    model_path = None
    if restart_path != None and restart_path != 'None':
        model_path = restart_path[0]
        config_path = restart_path[1]

        with open(config_path) as file:
            config = yaml.safe_load(file)["CRITIC_VARIABLES"]
        vars = config

    critic = Critic(
        54,
        vars["HIDDEN_DIM"],
        vars["MODEL_TYPE"],
        vars["GRAPH_NUM_LAYERS"],
        vars["DENSE_NUM_LAYERS"],
        vars["GRAPH_ACTIVATION"],
        vars["DENSE_ACTIVATION"],
        vars["DROPOUT"],
    )
    if model_path != None:
        critic.load_state_dict(torch.load(model_path)["model_state_dict"])

    return critic


# def modelGenerator()
